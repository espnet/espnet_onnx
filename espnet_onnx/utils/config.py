import json
import os
import warnings
from pathlib import Path
from types import SimpleNamespace

import yaml


def get_config(path):
    _, ext = os.path.splitext(path)
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            dic = json.load(f)
    elif ext in (".yaml", ".yml"):
        with open(path, "r", encoding="utf-8") as f:
            dic = yaml.safe_load(f)
    else:
        raise ValueError("Configuration format is not supported.")
    return Config(dic)


def save_config(config, path):
    _, ext = os.path.splitext(path)
    if ext == ".json":
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(config, Config):
                f.write(json.dumps(config.__dict__))
            else:
                f.write(json.dumps(config))
    elif ext in (".yaml", ".yml"):
        with open(path, "w", encoding="utf-8") as f:
            if isinstance(config, Config):
                yaml.dump(config.__dict__, f)
            else:
                yaml.dump(config, f)
    else:
        raise ValueError(f"File type {ext} is not supported.")


def update_model_path(tag_name, model_path):
    # get configuration of the tag name.
    tag_config_path = Path.home() / ".cache" / "espnet_onnx" / "tag_config.yaml"
    if os.path.exists(tag_config_path):
        config = get_config(tag_config_path)
    else:
        config = {}
    # if tag_name in config.keys():
    if tag_name in config.keys():
        warnings.warn(
            f'Onnx model "{tag_name}" is already saved in {config[tag_name]}. '
            + f'Update model path to "{model_path}".'
        )
    config[tag_name] = str(model_path)
    save_config(config, tag_config_path)


def get_tag_config():
    tag_config_path = Path.home() / ".cache" / "espnet_onnx" / "tag_config.yaml"
    if os.path.exists(tag_config_path):
        config = get_config(tag_config_path)
    else:
        config = {}
    return config


class Config(SimpleNamespace):
    def __init__(self, dic=None):
        super().__init__(**dic)
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                setattr(self, k, Config(v))
            elif isinstance(v, list):
                if len(v) == 0:
                    setattr(self, k, [])
                elif isinstance(v[0], dict):
                    setattr(self, k, [Config(_v) for _v in v])

    def __len__(self):
        return len(self.__dict__.keys())

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __str__(self):
        return "\n".join(
            ["%s : %s" % (str(k), str(v)) for k, v in self.__dict__.items()]
        )

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    @property
    def dic(self):
        ret = self.__dict__
        for k, v in ret.items():
            if isinstance(v, Config):
                ret[k] = v.dic
        return ret
