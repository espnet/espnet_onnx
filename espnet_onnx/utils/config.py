import json


def get_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        dic = json.load(f)

    return Config(dic)


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