import os
import pdb
import sys
import traceback

import yaml
from omegaconf import OmegaConf


def set_debug_hook():
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


def read_config_oc(config_path: str | None, cli_args: list[str] | None, config_cls):
    configs = []
    assert config_path is not None or cli_args is not None
    if config_path is not None:
        configs.append(OmegaConf.load(config_path))
    if cli_args is not None:
        configs.append(OmegaConf.from_cli(cli_args))
    merged_conf = OmegaConf.merge(*configs)
    out = config_cls(**merged_conf)
    if getattr(out, "debug", False):
        set_debug_hook()
    return out


def read_config(config_path, config_cls):
    with open(config_path) as inf:
        config = config_cls(**yaml.safe_load(inf))
    if config.debug:
        set_debug_hook()
    return config


def get_csv_path(raw_path: str, config) -> str:
    if getattr(config, "csv_prefix_to_strip", None) is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    if getattr(config, "csv_prefix_to_add", None) is not None:
        raw_path = config.csv_prefix_to_add + raw_path
    return raw_path


def get_csv_title(raw_path, config):
    if getattr(config, "csv_prefix_to_strip", None) is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    out = os.path.splitext(raw_path)[0]
    return out


def get_itos(dictionary_paths: list[str]) -> dict[str, list[str]]:
    out = {}
    for dictionary_path in dictionary_paths:
        feature_name = os.path.basename(dictionary_path).rsplit("_", maxsplit=1)[0]
        with open(dictionary_path) as inf:
            data = inf.readlines()
        contents = [
            line.split(" ", maxsplit=1)[0]
            for line in data
            if line and not line.startswith("madeupword")
        ]
        out[feature_name] = contents
    return out


def get_stoi(dictionary_paths: list[str]) -> dict[str, list[str]]:
    out = {}
    for dictionary_path in dictionary_paths:
        feature_name = os.path.basename(dictionary_path).rsplit("_", maxsplit=1)[0]
        with open(dictionary_path) as inf:
            data = inf.readlines()
        contents = [
            line.split(" ", maxsplit=1)[0]
            for line in data
            if line and not line.startswith("madeupword")
        ]
        out[feature_name] = {token: i for i, token in enumerate(contents)}
    return out
