from pathlib import Path

import onnx


def check_op_type_count(model_path, kwargs):
    model = onnx.load(Path(model_path))
    optype2count = {}
    for op_type in kwargs.keys():
        optype2count[op_type] = 0
    for node in model.graph.node:
        if node.op_type in optype2count:
            optype2count[node.op_type] += 1
    for op_type in kwargs.keys():
        assert (
            kwargs[op_type] == optype2count[op_type]
        ), f"{op_type}: {kwargs[op_type]} - {optype2count[op_type]}"
