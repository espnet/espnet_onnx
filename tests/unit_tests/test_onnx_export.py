import onnxruntime as ort
import pytest
import torch

from espnet_onnx.utils.onnx_export import export_onnx


class NestedInputsModel(torch.nn.Module):
    def forward(self, x, states, optional):
        return x + states[0] + states[1]

    def get_input_names(self):
        return ["x", "state_0", "state_1"]

    def get_output_names(self):
        return ["result"]

    def get_dynamic_axes(self):
        return {
            "x": {1: "length"},
            "state_0": {1: "length"},
            "state_1": {1: "length"},
            "result": {1: "length"},
        }


def test_export_nested_dynamic_inputs(tmp_path):
    model = NestedInputsModel().eval()
    args = (torch.randn(1, 4), [torch.randn(1, 4), torch.randn(1, 4)], None)
    path = tmp_path / "model.onnx"
    export_onnx(model, args, path, opset_version=18)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for length in (4, 7):
        x, first, second = (torch.randn(1, length) for _ in range(3))
        result = session.run(
            None, {"x": x.numpy(), "state_0": first.numpy(), "state_1": second.numpy()}
        )[0]
        torch.testing.assert_close(torch.from_numpy(result), model(x, [first, second], None))


def test_export_restores_training_modes_on_error(tmp_path, monkeypatch):
    model = NestedInputsModel()
    model.dropout = torch.nn.Dropout()
    model.train()
    model.dropout.eval()

    def fail_export(*args, **kwargs):
        assert not model.training
        assert not model.dropout.training
        raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", fail_export)
    args = (torch.randn(1, 4), [torch.randn(1, 4), torch.randn(1, 4)], None)
    with pytest.raises(RuntimeError, match="export failed"):
        export_onnx(model, args, tmp_path / "model.onnx", opset_version=18)

    assert model.training
    assert not model.dropout.training


def test_recurrent_reshape_only_changes_sequence_first_transposes():
    import numpy as np
    from onnx import helper, numpy_helper

    from espnet_onnx.utils.onnx_export import _fix_recurrent_reshape

    shape = numpy_helper.from_array(np.array([50, 1, 640], dtype=np.int64), "shape")
    recurrent = helper.make_node("GRU", ["x"], ["y", "state"])
    sequence_first = helper.make_node("Transpose", ["y"], ["sequence"], perm=[0, 2, 1, 3])
    batch_first = helper.make_node("Transpose", ["y"], ["batch"], perm=[2, 0, 1, 3])
    reshape_sequence = helper.make_node("Reshape", ["sequence", "shape"], ["sequence_out"])
    reshape_batch = helper.make_node("Reshape", ["batch", "shape"], ["batch_out"])
    graph = helper.make_graph(
        [recurrent, sequence_first, batch_first, reshape_sequence, reshape_batch],
        "recurrent",
        [],
        [],
        [shape],
    )
    model = helper.make_model(graph)

    _fix_recurrent_reshape(model)

    assert list(numpy_helper.to_array(model.graph.initializer[-1])) == [-1, 1, 640]
    assert model.graph.node[3].input[1] == "sequence_out_dynamic_shape"
    assert model.graph.node[4].input[1] == "shape"


def test_vits_inverse_flow_matches_espnet():
    from espnet2.gan_tts.vits.flow import ConvFlow

    from espnet_onnx.export.tts.models.tts_models.vits_flow import OnnxConvFlow

    torch.manual_seed(0)
    flow = ConvFlow(in_channels=2, hidden_channels=8, kernel_size=3, layers=2).eval()
    torch.nn.init.normal_(flow.proj.weight, std=0.05)
    torch.nn.init.normal_(flow.proj.bias, std=0.05)
    inputs = torch.randn(1, 2, 12) * 6
    mask = torch.ones(1, 1, 12)
    expected = flow(inputs, mask, inverse=True)
    actual = OnnxConvFlow(flow)(inputs, mask, inverse=True)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_export_removes_unused_onnx_inputs(tmp_path):
    class Model(torch.nn.Module):
        def forward(self, x, unused):
            return x + 1

        def get_input_names(self):
            return ["x", "unused"]

        def get_output_names(self):
            return ["result"]

        def get_dynamic_axes(self):
            return {"x": {0: "length"}}

    path = tmp_path / "unused.onnx"
    export_onnx(Model(), (torch.zeros(3), torch.ones(3)), path, opset_version=18)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [value.name for value in session.get_inputs()] == ["x"]
    result = session.run(None, {"x": torch.zeros(5).numpy()})[0]
    torch.testing.assert_close(torch.from_numpy(result), torch.ones(5))
