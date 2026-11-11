"""Helpers for the torch.export-based ONNX exporter."""

import onnx
import torch
from onnx import numpy_helper


def dynamic_shapes_from_model(model, inputs):
    """Translate the model's flattened ONNX input axes to PyTorch's input tree."""
    input_names = iter(model.get_input_names())
    axes = model.get_dynamic_axes()
    batch_dim = torch.export.Dim("batch", min=1)

    def shape_spec(value):
        if isinstance(value, torch.Tensor):
            name = next(input_names)
            return {
                axis: (
                    (
                        batch_dim
                        if axes[name][axis] == "batch"
                        else torch.export.Dim.DYNAMIC
                    )
                    if axis == 0
                    else torch.export.Dim.AUTO
                )
                for axis in axes.get(name, {})
            }
        if isinstance(value, list):
            return [shape_spec(item) for item in value]
        if isinstance(value, tuple):
            return tuple(shape_spec(item) for item in value)
        if isinstance(value, dict):
            return {key: shape_spec(item) for key, item in value.items()}
        return None

    args = inputs if isinstance(inputs, tuple) else (inputs,)
    return args, tuple(shape_spec(arg) for arg in args)


def export_onnx(model, inputs, path, *, opset_version, verbose=False):
    """Export a model with dynamic shapes matching its nested input structure."""
    args, dynamic_shapes = dynamic_shapes_from_model(model, inputs)
    training_modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if any(
            getattr(module, "requires_strict_export", False) for module in model.modules()
        ):
            exported_model = torch.export.export(
                model, args, dynamic_shapes=dynamic_shapes, strict=True
            )
            export_args = ()
            export_shapes = None
        else:
            exported_model = model
            export_args = args
            export_shapes = dynamic_shapes
        with torch.backends.mkldnn.flags(enabled=False):
            torch.onnx.export(
                exported_model,
                export_args,
                path,
                verbose=verbose,
                opset_version=opset_version,
                input_names=model.get_input_names(),
                output_names=model.get_output_names(),
                dynamic_shapes=export_shapes,
                dynamo=True,
                external_data=False,
            )
        # torch.export can specialize intermediate shapes after recurrent layers
        # even when their input sequence length is dynamic.  Stale value_info
        # makes ONNX Runtime optimize MatMul using the example length.
        graph = onnx.load_model(path, load_external_data=False)
        del graph.graph.value_info[:]
        _fix_recurrent_reshape(graph)
        _remove_unused_inputs(graph.graph)
        dynamic_axes = model.get_dynamic_axes()
        for output in graph.graph.output:
            for axis, name in dynamic_axes.get(output.name, {}).items():
                output.type.tensor_type.shape.dim[axis].dim_param = name
        onnx.save_model(graph, path)
    finally:
        for module, training in training_modes:
            module.training = training


def _fix_recurrent_reshape(model):
    """Keep the sequence dimension of exported GRU/LSTM outputs dynamic.

    PyTorch can emit a reshape with the example length after an ONNX recurrent
    operator. Infer that dimension from the runtime tensor instead.
    """
    producers = {output: node for node in model.graph.node for output in node.output}
    initializers = {value.name: value for value in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Reshape" or node.input[1] not in initializers:
            continue
        transpose = producers.get(node.input[0])
        if transpose is None or transpose.op_type != "Transpose":
            continue
        recurrent = producers.get(transpose.input[0])
        if recurrent is None or recurrent.op_type not in {"GRU", "LSTM", "RNN"}:
            continue
        if transpose.input[0] != recurrent.output[0]:
            continue
        permutation = next(
            (tuple(attribute.ints) for attribute in transpose.attribute if attribute.name == "perm"),
            None,
        )
        if not permutation or permutation[0] != 0:
            continue
        shape = numpy_helper.to_array(initializers[node.input[1]])
        if shape.ndim != 1 or len(shape) != 3 or shape[0] <= 0 or -1 in shape:
            continue
        new_shape = shape.copy()
        new_shape[0] = -1
        name = f"{node.output[0]}_dynamic_shape"
        model.graph.initializer.append(numpy_helper.from_array(new_shape, name))
        node.input[1] = name


def _remove_unused_inputs(graph):
    """Remove unused graph inputs, including those retained by torch.export."""
    used = {output.name for output in graph.output}

    def collect(subgraph):
        for node in subgraph.node:
            used.update(name for name in node.input if name)
            for attribute in node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    collect(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        collect(nested)

    collect(graph)
    kept = [value for value in graph.input if value.name in used]
    del graph.input[:]
    graph.input.extend(kept)
