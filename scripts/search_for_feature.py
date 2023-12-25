import argparse
from dataclasses import dataclass
from functools import partial
from typing import Any, Iterable

from music_df.find import find_simultaneous_feature
from music_df.read import read
from music_df.script_helpers import read_config_oc
from music_df.time import merge_contiguous_durations, time_to_bar_number_and_offset


@dataclass
class Config:
    feature_name: str
    feature_values: Iterable[Any]
    # TODO: (Malcolm 2023-12-23) figure out a graceful way of supplying
    #   multiple input files
    input_files: Iterable[str]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    parser.add_argument("--input-files", nargs="+")
    # remaining passed through to omegaconf
    # parser.add_argument("remaining", nargs=argparse.REMAINDER)

    args, remaining = parser.parse_known_args()
    return args, remaining


def print_result(result: tuple[tuple[float, int, float], ...], indent: int = 0):
    def _format(t: tuple[float, int, float]) -> str:
        return f"m{t[1]}.{t[2] + 1:.3} (offset={t[0]})"

    print(f"{' ' * indent}From {_format(result[0])} to {_format(result[1])}")


def main():
    args, remaining = parse_args()
    get_config = partial(Config, input_files=args.input_files)
    config = read_config_oc(args.config_file, remaining, get_config)

    for input_file in args.input_files:
        music_df = read(input_file)
        result = find_simultaneous_feature(
            music_df, config.feature_name, config.feature_values
        )
        result = merge_contiguous_durations(result)
        result = [
            tuple(((x, *time_to_bar_number_and_offset(music_df, x)) for x in t))
            for t in result
        ]
        if not result:
            continue
        print(input_file)
        for r in result:
            print_result(r, indent=4)


if __name__ == "__main__":
    main()
