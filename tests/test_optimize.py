
# This test suite verifies that espnet_onnx exports
# model correctly and match the result.

import os
import pytest
from pathlib import Path
import glob
import onnx

from espnet_onnx.export.optimize.optimizer import optimize_model
from .op_test_utils import check_op_type_count

test_cases = [
    ['encoder', 'transformer', 4, 256, 3, 0, False],
    ['encoder', 'contextual_block_transformer', 4, 256, 3, 0, False],
    # ['encoder', 'transformer', 4, 256, 4, 0, True],
    # ['encoder', 'contextual_block_transformer', 4, 256, 4, 0, True],
    # ['decoder', 'transformer', 4, 256, 4, 0, True],
    # ['lm', 'transformer_pe', 4, 256, 4, 0, True],
    # ['lm', 'transformer', 4, 256, 4, 0, True],
]

@pytest.mark.parametrize('model_type, model_name, n_head, h_size, n_att, n_cross_att, use_custom_ort', test_cases)
def test_optimize(model_type, model_name, n_head, h_size, n_att, n_cross_att, use_custom_ort, model_export):
    export_dir = model_export.cache_dir / 'test' / \
        model_type / f'cache_{model_name}'
    output_dir = model_export.cache_dir / 'test' / \
        'optimize' / f'cache_{model_name}'
    
    input_model = glob.glob(os.path.join(export_dir , f'*{model_type}*'))[0]
    model_name = os.path.basename(input_model)
    
    if use_custom_ort:
        opt_model_type = 'espnet'
    else:
        opt_model_type = 'bert'
    
    optimize_model(
        input_model = str(input_model),
        output_model = str(output_dir / model_name),
        num_heads = n_head,
        hidden_size = h_size,
        model_type = opt_model_type,
    )
    
    # load the optimized model and check if the number of fused nodes is correct.
    nodes = {}
    if n_att > 0:
        nodes['Attention'] = n_att
    if n_cross_att > 0:
        nodes['CrossAttention'] = n_cross_att
        
    check_op_type_count(str(output_dir / model_name), **nodes)
    