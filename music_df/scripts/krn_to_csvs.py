"""Convenience script so we don't have to convert from krn on HPC (where we'd
have to build the TOTABLE binary.)
"""

import glob
import json
import os
import pdb
import random
import re
import sys
import traceback
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Manager, Pool
from typing import Iterable

from tqdm import tqdm

from music_df.quantize_df import quantize_df
from music_df.read_krn import read_krn
from music_df.salami_slice import salami_slice
from music_df.script_helpers import read_config_oc


@dataclass
class KRNToCSVConfig:
    output_folder: str
    input_folder: str | None = None
    input_files: list[str] | None = None
    max_files: int | None = None
    random_files: bool = False
    salami_slice: bool = True
    seed: int = 42
    event_types: Iterable[str] = field(
        default_factory=lambda: {"note", "time_signature", "bar"}
    )
    overwrite: bool = False
    num_workers: int = 8
    regex: str | None = None
    debug: bool = False
    verbose: bool = False


def get_krn_files(folder_path):
    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(os.path.join(folder_path, "**", "*.krn"), recursive=True)


def do_krn_file(krn_file, config, output_list, error_file_list, debug):
    try:
        if config.input_folder is not None:
            output_item = krn_file.replace(config.input_folder, "")
        else:
            output_item = os.path.basename(krn_file)
        output_basename = (
            output_item.lstrip(os.path.sep)
            .replace(" ", "_")
            .replace(os.path.sep, "+")
            .replace(".krn", ".csv")
        )
        output_path = os.path.join(config.output_folder, output_basename)

        if os.path.exists(output_path) and not config.overwrite:
            return

        music_df = read_krn(krn_file)
        music_df = music_df[music_df.type.isin(config.event_types)]
        if config.salami_slice:
            music_df = salami_slice(music_df)
        music_df = music_df.drop("spelling", axis=1)
        music_df.to_csv(output_path)

        if config.verbose:
            print(f"Wrote {output_path}")
        output_list.append(krn_file)
    except Exception as exc:
        if config.debug:
            raise
        error_file_list.append((krn_file, repr(exc)))


def write_json(config, output_files):
    json_path = os.path.join(config.output_folder, "source_files.json")
    if not config.overwrite and os.path.exists(json_path):
        with open(json_path, "r") as inf:
            existing_contents = json.load(inf)
        with open(json_path, "w") as outf:
            json.dump(existing_contents + list(output_files), outf)
    else:
        with open(json_path, "w") as outf:
            json.dump(list(output_files), outf, indent=2)


def main():
    config = read_config_oc(
        config_path=None, cli_args=sys.argv[1:], config_cls=KRNToCSVConfig
    )
    if config.debug:

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    assert (config.input_folder is not None) != (config.input_files is not None)

    if config.input_folder is not None:
        krn_files = get_krn_files(config.input_folder)
    else:
        krn_files = config.input_files
    assert krn_files is not None

    if config.regex is not None:
        krn_files = [f for f in krn_files if re.search(config.regex, f)]
    random.seed(config.seed)

    if config.random_files:
        random.shuffle(krn_files)

    krn_files = krn_files[: config.max_files]
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
                            do_krn_file,
                            config=config,
                            output_list=output_files,
                            error_file_list=error_files,
                            debug=config.debug,
                        ),
                        krn_files,
                    ),
                    total=len(krn_files),
                )
            )
    else:
        output_files = []
        error_files = []
        for krn_file in krn_files:
            do_krn_file(krn_file, config, output_files, error_files, debug=config.debug)

    write_json(config, output_files)

    if error_files:
        print("Errors:")
        for krn_file, exception_str in error_files:
            print(f"{krn_file}: {exception_str}")
        print(f"{len(error_files)} total error files")


if __name__ == "__main__":
    main()
