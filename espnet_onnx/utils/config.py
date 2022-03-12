import os
import json
import yaml


def get_config(path):
    _, ext = os.path.splitext(path)
    if ext == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
    elif ext in ('.yaml', '.yml'):
        with open(path, 'r', encoding='utf-8') as f:
            dic = yaml.safe_load(f)
    else:
        raise ValueError('Configuration format is not supported.')
    return Config(dic)

def save_config(config, path):
    _, ext = os.path.splitext(path)
    if ext == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(config, Config):
                f.write(json.dumps(config.dic))
            elif isinstance(config, dict):
                f.write(json.dumps(config))
            else:
                raise ValueError('configuration type is not supported.')
    elif ext in ('.yaml', '.yml'):
        with open(path, 'w', encoding='utf-8') as f:
            if isinstance(config, Config):
                yaml.dump(config.dic, f)
            elif isinstance(config, dict):
                yaml.dump(config, f)
            else:
                raise ValueError('configuration type is not supported.')
    else:
        raise ValueError(f'File type {ext} is not supported.')

class Config(object):
    def __init__(self, dic):
        for j, k in dic.items():
            if isinstance(k, dict):
                setattr(self, j, Config(k))
            else:
                if k is not None:
                    setattr(self, j, k)
                else:
                    setattr(self, j, None)
        self.dic = dic

    def __len__(self):
        return len(self.dic.keys())

    def __getitem__(self, key):
        return self.dic[key]

    def __str__(self):
        return '\n'.join(['%s : %s' % (str(k), str(v)) for k, v in self.dic.items()])

    def keys(self):
        return self.dic.keys()

    def values(self):
        return self.dic.values()