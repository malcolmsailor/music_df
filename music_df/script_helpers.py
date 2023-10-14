import os

import yaml
from omegaconf import OmegaConf


def read_config_oc(config_path: str | None, cli_args: list[str] | None, config_cls):
    configs = []
    assert config_path is not None or cli_args is not None
    if config_path is not None:
        configs.append(OmegaConf.load(config_path))
    if cli_args is not None:
        configs.append(OmegaConf.from_cli(cli_args))
    merged_conf = OmegaConf.merge(*configs)
    return config_cls(**merged_conf)


def read_config(config_path, config_cls):
    with open(config_path) as inf:
        config = config_cls(**yaml.safe_load(inf))
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
