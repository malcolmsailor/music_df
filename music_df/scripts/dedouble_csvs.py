"""
We want to process data in the following order:
1. detrill
2. quantize
3. merge notes
4. salami slice
5. dedouble
"""

import argparse
import glob
import os
import shutil
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from multiprocessing.managers import ListProxy

from tqdm import tqdm

from music_df.dedouble import dedouble
from music_df.merge_notes import merge_notes
from music_df.quantize_df import quantize_df
from music_df.read_csv import read_csv


@dataclass
class Config:
    input_folder: str
    output_folder: str
    debug: bool = False
    num_workers: int = 0
    max_files: None | int = None
    # 16 is the resolution of musicbert
    quantize: None | int = None
    merge_notes: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--quantize", type=int, default=None)
    parser.add_argument("--merge-notes", action="store_true")
    args = parser.parse_args()
    return args


def get_csv_and_other_files(input_folder):
    all_files = glob.glob(input_folder + "/**/*", recursive=True)
    csv_files = [file for file in all_files if file.endswith(".csv")]
    other_files = [
        file
        for file in all_files
        if not file.endswith(".csv") and not os.path.isdir(file)
    ]
    return csv_files, other_files


def do_csv_file(inputf: str, config: Config, output_list: list | ListProxy):
    music_df = read_csv(inputf)
    assert music_df is not None

    if config.quantize:
        music_df = quantize_df(music_df, tpq=config.quantize)
    if config.merge_notes:
        music_df = merge_notes(music_df)

    dedoubled_df = dedouble(music_df)
    relative_path = os.path.relpath(inputf, config.input_folder)
    output_path = os.path.join(config.output_folder, relative_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dedoubled_df.to_csv(output_path)
    output_list.append(output_path)


def main():
    args = parse_args()
    config = Config(**vars(args))

    input_files, other_files = get_csv_and_other_files(config.input_folder)
    if config.max_files is not None:
        input_files = input_files[: config.max_files]

    os.makedirs(config.output_folder, exist_ok=True)
    if config.num_workers > 1:
        manager = Manager()
        output_files = manager.list()
        # error_files = manager.list()
        with Pool(config.num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            do_csv_file,
                            config=config,
                            output_list=output_files,
                        ),
                        input_files,
                    ),
                    total=len(input_files),
                )
            )

    else:
        output_files = []
        # error_files = []
        for inputf in input_files:
            do_csv_file(inputf, config, output_files)

    for file in other_files:
        relative_path = os.path.relpath(file, config.input_folder)
        output_path = os.path.join(config.output_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy2(file, output_path)


if __name__ == "__main__":
    main()
