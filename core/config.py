import argparse

import yaml


def load_config(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def edit_config():
    """update configs"""
    pass


def load_config_by_log(path):
    config = {}
    file = open(path)
    for line in file:
        line = line.replace("\n", "")
        k, v = line.split(":", maxsplit=1)
        if v.isdigit():
            v = int(v)
        elif "[" in v and "]" in v:
            v = v[2:-2]
            v = v.split("\'")
            v_ = []
            for i in v:
                if ',' in i:
                    continue
                v_.append(i)
            v = v_
        config[k] = v
    config = argparse.Namespace(**config)
    return config
