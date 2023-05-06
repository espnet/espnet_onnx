import glob
import os

import yaml
from espnet2.bin.asr_inference import Speech2Text

from ..op_test_utils import check_op_type_count


def export_model(model_export, espnet_model, config):
    model_export.set_export_config(
        max_seq_len=5000,
        use_ort_for_espnet=config["use_ort_for_espnet"],
        use_gpu=("GPU" in config["device"]),
    )
    model_export.export(
        espnet_model,
        config["tag_name"],
        quantize=("Quantize" in config["device"]),
        optimize=(config["check_optimize"] is not None),
    )


def eval_model(espnet_model):
    espnet_model.asr_model.encoder.eval()
    for k in espnet_model.beam_search.full_scorers.keys():
        espnet_model.beam_search.full_scorers[k].eval()


def check_models(cache_dir, tag_name, check_export, check_quantize):
    file_paths = {"full": {}, "quantize": {}}
    # test full model
    for emt in check_export:
        # test if models are properly exported
        test_path = glob.glob(
            os.path.join(cache_dir, tag_name.replace(" ", "-"), "full", f"*{emt}.onnx")
        )
        assert len(test_path) == 1
        file_paths["full"][emt] = test_path[0]

    # test quantized model
    if check_quantize:
        for emt in check_export:
            # test if models are properly exported
            test_path = glob.glob(
                os.path.join(
                    cache_dir, tag_name.replace(" ", "-"), "quantize", f"*{emt}_qt.onnx"
                )
            )
            assert len(test_path) == 1
            file_paths["quantize"][emt] = test_path[0]

    return file_paths


def check_optimize(model_config, file_paths):
    for device in model_config["device"]:
        if device is not None and model_config["optimization"][device] is not None:
            if device == "Quantize":
                for k in model_config["optimization"]["Quantize"].keys():
                    check_op_type_count(
                        file_paths["quantize"][k],
                        model_config["optimization"]["Quantize"][k],
                    )
            else:
                for k in model_config["optimization"][device].keys():
                    check_op_type_count(
                        file_paths["full"][k], model_config["optimization"][device][k]
                    )


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        dic = yaml.safe_load(f)
    return dic


def build_model(model_config):
    transducer_config = None
    if model_config.use_transducer:
        transducer_config = {"search_type": "default", "score_norm": True}

    lm_train_config = None
    if model_config.use_lm:
        lm_train_config = model_config.lm_train_config

    model = Speech2Text(
        asr_train_config=model_config.asr_train_config,
        lm_train_config=lm_train_config,
        transducer_conf=transducer_config,
    )
    return model
