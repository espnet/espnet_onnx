# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Convert Bert ONNX model converted from TensorFlow or exported from PyTorch to use Attention, Gelu,
# SkipLayerNormalization and EmbedLayerNormalization ops to optimize
# performance on NVidia GPU and CPU.
#
# For Bert model exported from PyTorch, OnnxRuntime has bert model optimization support internally.
# You can use the option --use_onnxruntime to check optimizations from OnnxRuntime.
# For Bert model file like name.onnx, optimized model for GPU or CPU from OnnxRuntime will output as
# name_ort_gpu.onnx or name_ort_cpu.onnx in the same directory.
#
# This script is retained for experiment purpose. Useful senarios like the following:
#  (1) Change model from fp32 to fp16 for mixed precision inference in GPU with Tensor Core.
#  (2) Change input data type from int64 to int32.
#  (3) Some model cannot be handled by OnnxRuntime, and you can modify this script to get optimized model.

# This script is modified by Masao Someki to convert espnet models.
# The Transformer and Conformer based models are supported.
# Copyright (c) 2022 Masao Someki

import argparse
import logging
import os
from typing import Dict, Optional

import coloredlogs
from onnx import ModelProto, load_model
from .fusion_options import FusionOptions
from .espnet_optimizer import ESPnetOptimizer

logger = logging.getLogger(__name__)


def optimize_by_onnxruntime(
    onnx_model_path: str,
    use_gpu: bool = False,
    optimized_model_path: Optional[str] = None,
    disabled_optimizers=[],
) -> str:
    """
    Use onnxruntime to optimize model.

    Args:
        onnx_model_path (str): the path of input onnx model.
        use_gpu (bool): whether the optimized model is targeted to run in GPU.
        optimized_model_path (str or None): the path of optimized model.
        opt_level (int): graph optimization level.
        disabled_optimizers (List[str]): a list of names of disabled optimizers
    Returns:
        optimized_model_path (str): the path of optimized model
    """
    import onnxruntime

    if use_gpu and "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        logger.error("There is no gpu for onnxruntime to do optimization.")
        return onnx_model_path

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC

    if optimized_model_path is None:
        path_prefix = onnx_model_path[:-5]  # remove .onnx suffix
        optimized_model_path = "{}_{}.onnx".format(path_prefix, "gpu" if use_gpu else "cpu")

    sess_options.optimized_model_filepath = optimized_model_path

    kwargs = {}
    if disabled_optimizers:
        kwargs["disabled_optimizers"] = disabled_optimizers

    if not use_gpu:
        session = onnxruntime.InferenceSession(
            onnx_model_path, sess_options, providers=["CPUExecutionProvider"], **kwargs
        )
    else:
        session = onnxruntime.InferenceSession(
            onnx_model_path, sess_options, providers=["CUDAExecutionProvider"], **kwargs
        )
        assert "CUDAExecutionProvider" in session.get_providers()  # Make sure there is GPU

    assert os.path.exists(optimized_model_path) and os.path.isfile(optimized_model_path)
    logger.debug("Save optimized model by onnxruntime to {}".format(optimized_model_path))
    return optimized_model_path


def optimize_by_fusion(
    model: ModelProto,
    num_heads: int = 0,
    hidden_size: int = 0,
    unidirectional: int = 0,
    optimization_options: Optional[FusionOptions] = None,
):
    """Optimize Model by graph fusion logic.

    Note that ONNXRuntime graph optimizations (like constant folding) will not be applied. So it is better to enable
    constant folding during exporting ONNX model, or run optimize_by_onnxruntime on the model first like optimize_model.

    For BERT model, num_heads and hidden_size are optional. For other model types, you need specify these parameters.

    Args:
        model (ModelProto): model object
        model_type (str, optional): model type - like bert, bert_tf, bert_keras or gpt2. Defaults to 'bert'.
        num_heads (int, optional): number of attention heads. Defaults to 0.
                                   0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int, optional): hidden size. Defaults to 0.
                                     0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.

     Returns:
        object of an optimizer class.
    """
    if optimization_options is None:
        optimization_options = FusionOptions()

    optimizer = ESPnetOptimizer(model, num_heads, hidden_size, unidirectional)

    optimizer.optimize(optimization_options)

    optimizer.topological_sort()

    optimizer.model.producer_name = "espnet_onnx.export"
    from espnet_onnx import __version__ as espnet_onnx_version

    optimizer.model.producer_version = espnet_onnx_version

    return optimizer


def optimize_model(
    input: str,
    num_heads: int = 0,
    hidden_size: int = 0,
    unidirectional: int = 0,
    optimization_options: Optional[FusionOptions] = None,
    use_gpu: bool = False,
    only_onnxruntime: bool = False,
):
    """Optimize Model by OnnxRuntime and/or python fusion logic.

    ONNX Runtime has graph optimizations (https://onnxruntime.ai/docs/resources/graph-optimizations.html).
    However, the coverage is limited. We also have graph fusions that implemented in Python to improve the coverage.

    To use ONNX Runtime only and no Python fusion logic, use only_onnxruntime flag and a positive opt_level like
        optimize_model(input, use_gpu=False, only_onnxruntime=True)

    For espnet Transformer and Conformer based models, num_heads and hidden_size are required.

    Args:
        input (str): input model path.
        num_heads (int): number of attention heads. Defaults to 0.
                        0 allows detect the parameter from graph automatically (for model_type "bert" only).
        hidden_size (int): hidden size. Defaults to 0.
                        0 allows detect the parameter from graph automatically (for model_type "bert" only).
        optimization_options (FusionOptions, optional): optimization options that turn on/off some fusions. Defaults to None.
        use_gpu (bool, optional): use gpu or not for onnxruntime. Defaults to False.
        only_onnxruntime (bool, optional): only use onnxruntime to optimize model, and no python fusion. Defaults to False.

     Returns:
        object of an optimizer class.
    """
    if num_heads == 0 or hidden_size == 0:
        raise RuntimeError('num_heads or hidden_size must be a positive value.')

    temp_model_path = None
    
    # Disable some optimizers that might cause failure in symbolic shape inference or attention fusion.
    disabled_optimizers = (
        []
        if only_onnxruntime
        else [
            "MatMulScaleFusion",
            "MatMulAddFusion",
            "SimplifiedLayerNormFusion",
            "GemmActivationFusion",
            "BiasSoftmaxFusion",
        ]
    )
    temp_model_path = optimize_by_onnxruntime(
        input,
        use_gpu=use_gpu,
        disabled_optimizers=disabled_optimizers,
    )

    if only_onnxruntime and not temp_model_path:
        logger.warning("Please specify a positive value for opt_level when only_onnxruntime is True")

    model = load_model(temp_model_path or input)

    if only_onnxruntime:
        optimizer = ESPnetOptimizer(model, num_heads, hidden_size)
    else:
        optimizer = optimize_by_fusion(model, num_heads, hidden_size, unidirectional, optimization_options)

    # Remove the temporary model.
    if temp_model_path:
        os.remove(temp_model_path)
        logger.debug("Remove tempoary model: {}".format(temp_model_path))

    return optimizer


def get_fusion_statistics(optimized_model_path: str) -> Dict[str, int]:
    """
    Get counter of fused operators in optimized model.

    Args:
        optimized_model_path (str): the path of onnx model.

    Returns:
        A dictionary with operator type as key, and count as value
    """
    model = load_model(optimized_model_path, format=None, load_external_data=True)
    optimizer = BertOnnxModel(model)
    return optimizer.get_fused_operator_statistics()
