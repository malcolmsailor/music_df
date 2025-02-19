import argparse
import ast
import glob
import logging
import os
import pdb
import sys
import traceback
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from multiprocessing.managers import ListProxy

from tqdm import tqdm

from music_df.quantize_df import quantize_df
from music_df.midi_parser.parser import df_to_midi
from music_df.read_csv import read_csv
from music_df.salami_slice import salami_slice

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


@dataclass
class Config:
    input_folder: str
    output_folder: str
    debug: bool = False
    num_workers: int = 16
    max_files: None | int = None
    # 16 is the resolution of musicbert
    quantize: None | int = None
    overwrite: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("output_folder")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--quantize", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=16)
    args = parser.parse_args()
    return args


def do_csv_file(
    inputf: str,
    config: Config,
    output_list: list | ListProxy,
    error_file_list: list | ListProxy,
):
    output_path = os.path.join(config.output_folder, os.path.basename(inputf))
    if os.path.exists(output_path) and not config.overwrite:
        return
    try:
        music_df = read_csv(inputf)
        assert music_df is not None

        if config.quantize:
            music_df = quantize_df(music_df, tpq=config.quantize)

        salami_sliced = salami_slice(music_df)
        salami_sliced.to_csv(output_path)
        # print(f"Wrote {output_path}")

        output_list.append(output_path)
    except Exception as exc:
        if config.debug:
            raise
        error_file_list.append((inputf, repr(exc)))


def main():
    args = parse_args()
    config = Config(**vars(args))

    if config.debug:
        sys.excepthook = custom_excepthook

    input_files = glob.glob(f"{config.input_folder}/*.csv")
    if config.max_files is not None:
        input_files = input_files[: config.max_files]

    os.makedirs(config.output_folder, exist_ok=True)
    if config.num_workers > 1:
        manager = Manager()
        output_files = manager.list()
        error_files = manager.list()
        with Pool(config.num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            do_csv_file,
                            config=config,
                            output_list=output_files,
                            error_file_list=error_files,
                        ),
                        input_files,
                    ),
                    total=len(input_files),
                )
            )
    else:
        output_files = []
        error_files = []
        for inputf in input_files:
            do_csv_file(inputf, config, output_files, error_files)

    if error_files:
        print("Errors:")
        for xml_file, exception_str in error_files:
            print(f"{xml_file}: {exception_str}")
        print(f"{len(error_files)} total error files")


if __name__ == "__main__":
    main()
