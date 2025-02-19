import glob
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from pathlib import Path

import sys
from dataclasses import dataclass
from omegaconf import OmegaConf
from tqdm import tqdm

DATASETS = os.environ.get("DATASETS", os.path.expanduser("~/datasets"))


@dataclass
class Config:
    input_folder: str
    output_folder: str
    num_workers: int = 16
    max_files: int | None = None
    overwrite: bool = False
    dryrun: bool = False


def musescore_quantize(input_path: str, output_path: str):
    command = ["mscore", input_path, "-o", output_path]
    subprocess.run(command, check=True, capture_output=True)


def get_paths(config):

    input_files = glob.glob(
        os.path.join(config.input_folder, "**", "*.mid"), recursive=True
    )

    output_files = []
    for f in input_files:
        relpath = os.path.relpath(os.path.dirname(f), config.input_folder)
        relpath = "" if relpath == "." else relpath
        output_path = os.path.join(
            config.output_folder, relpath, os.path.basename(f[:-4]) + ".xml"
        )
        output_files.append(output_path)

    return list(zip(input_files, output_files))


def filter_newer_outputs(file_pairs):
    filtered_pairs = []
    for input_file, output_file in file_pairs:
        if not os.path.exists(output_file):
            # output file does not exist, include this pair
            filtered_pairs.append((input_file, output_file))
        elif os.path.getmtime(input_file) > os.path.getmtime(output_file):
            # input file is newer than output file, include this pair
            filtered_pairs.append((input_file, output_file))
    return filtered_pairs


def do_file(paths: tuple[str, str], config, exceptions_list):
    input_file, output_file = paths
    if not config.overwrite and os.path.exists(output_file):
        return
    # print(f"Saving {output_file}")
    try:
        musescore_quantize(input_file, output_file)
    except Exception as exc:
        exceptions_list.append(f"{input_file}: {repr(exc)}")


def make_output_folders(paths):
    output_folders = {os.path.dirname(f) for _, f in paths}
    for f in output_folders:
        os.makedirs(f, exist_ok=True)


def run(config: Config):
    input_and_output_files = get_paths(config)
    input_and_output_files = filter_newer_outputs(input_and_output_files)

    if config.max_files is not None and config.max_files > 0:
        input_and_output_files = input_and_output_files[: config.max_files]
    # I'm not sure if multiprocessing will cause any issues if two processes try to
    #   create a folder at the same time, so we just create all the output folders
    #   first in the main process.
    make_output_folders(input_and_output_files)

    if config.num_workers > 1:
        manager = Manager()
        exceptions = manager.list()
        with Pool(config.num_workers) as pool:
            list(
                tqdm(
                    pool.imap_unordered(
                        partial(do_file, config=config, exceptions_list=exceptions),
                        input_and_output_files,
                    ),
                    total=len(input_and_output_files),
                )
            )
    else:
        exceptions = []
        for paths in input_and_output_files:
            do_file(paths, config, exceptions)

    for exception in exceptions:
        print(exception)
    if exceptions:
        print(f"{len(exceptions)} total errors")


if __name__ == "__main__":
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore
    run(config)
