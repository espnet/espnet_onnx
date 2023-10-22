from collections.abc import Iterable
from importlib import import_module

import torch

from espnet_onnx.utils.config import get_config


def import_class(module):
    class_name = module.split(".")[-1]
    module_name = '.'.join(module.split(".")[:-1])
    m = import_module(module_name)
    return getattr(m, class_name)


def get_replace_modules(convert_map_path, task):
    # get all modules in the model
    # yaml must define 'from' and 'to' keys for each module name.
    convert_map = get_config(convert_map_path)
    rep = []
    if task not in convert_map.keys():
        raise ValueError(f"task {task} is not defined in {convert_map_path}")
    if not isinstance(convert_map[task], Iterable):
        raise ValueError(f"task {task} is not iterable")

    cmap = convert_map[task]
    for replacement in cmap:
        rep.append({
            'from': import_class(replacement.dic.pop('from')),
            'to': import_class(replacement.dic.pop('to')),
            **replacement.dic
        })

    return rep


def replace_modules(replacements, parent_module, **kwargs):
    # check all child modules and replace them into espnet_onnx modules
    # check if we need to replace the parent module
    if isinstance(parent_module, Iterable):
        for i, m in enumerate(parent_module):
            parent_module[i] = replace_modules(replacements, m, **kwargs)

    elif len(parent_module._modules.keys()) > 1:
        replaced = False
        for replacement in replacements:
            if isinstance(parent_module, replacement['from']):
                replaced = True
                parent_module = replacement['to'](parent_module, **replacement, **kwargs)

        if not replaced:
            for k, v in parent_module._modules.items():
                if isinstance(v, torch.nn.Module):
                    setattr(parent_module, k, replace_modules(replacements, v, **kwargs))
    else:
        for replacement in replacements:
            if isinstance(parent_module, replacement['from']):
                parent_module = replacement['to'](parent_module, **replacement, **kwargs)

    return parent_module
