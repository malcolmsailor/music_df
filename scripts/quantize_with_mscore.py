import glob
import os
import subprocess
from dataclasses import dataclass
from functools import partial
from multiprocessing import Manager, Pool
from pathlib import Path

DATASETS = os.environ.get("DATASETS", os.path.expanduser("~/datasets"))


@dataclass
class Config:
    input_folder: str = os.path.join(DATASETS, "/Users/malcolm/datasets/YCAC-1.0")
    output_folder: str = os.path.join(
        DATASETS, "/Users/malcolm/datasets/YCAC-1.0-quantized"
    )
    num_workers: int = 16
    max_files: int | None = None
    overwrite: bool = False


def musescore_quantize(input_path: str, output_path: str):
    command = ["mscore", input_path, "-o", output_path]
    subprocess.run(command, check=True, capture_output=True)


def get_paths(config):
    input_and_output_files = []

    for dirpath, dirnames, filenames in os.walk(config.input_folder):
        relpath = os.path.relpath(dirpath, config.input_folder)
        for filename in filenames:
            if filename.endswith(".mid"):
                input_path = os.path.join(dirpath, filename)
                output_path = os.path.join(
                    config.output_folder, relpath, filename[:-4] + ".xml"
                )
                input_and_output_files.append((input_path, output_path))

    return input_and_output_files


def do_file(paths: tuple[str, str], config, exceptions_list):
    input_file, output_file = paths
    if not config.overwrite and os.path.exists(output_file):
        return
    print(f"Saving {output_file}")
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
                pool.imap_unordered(
                    partial(do_file, config=config, exceptions_list=exceptions),
                    input_and_output_files,
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
    config = Config()
    run(config)
