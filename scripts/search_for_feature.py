import argparse
from dataclasses import dataclass
from typing import Any, Iterable

from music_df.find import find_simultaneous_feature
from music_df.read import read
from music_df.script_helpers import read_config_oc


@dataclass
class Config:
    feature_name: str
    feature_values: Iterable[Any]
    # TODO: (Malcolm 2023-12-23) figure out a graceful way of supplying
    #   multiple input files
    input_file: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    # remaining passed through to omegaconf
    parser.add_argument("remaining", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)
    breakpoint()

    music_df = read(config.input_file)
    result = find_simultaneous_feature(
        music_df, config.feature_name, config.feature_values
    )
    print(result)


if __name__ == "__main__":
    main()
