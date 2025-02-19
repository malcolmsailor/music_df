import glob
import json
import os
import pdb
import random
import re
import sys
import traceback
import warnings
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Manager, Pool
from typing import Iterable

from tqdm import tqdm

from music_df.quantize_df import quantize_df
from music_df.read_midi import read_midi
from music_df.salami_slice import salami_slice
from music_df.script_helpers import read_config_oc


@dataclass
class MidiToCSVConfig:
    csv_input_folder: str
    midi_input_folder: str
    output_folder: str | None = None
    max_files: int | None = None
    random_files: bool = False
    salami_slice: bool = True
    seed: int = 42
    event_types: Iterable[str] = field(
        default_factory=lambda: {"note", "time_signature", "bar"}
    )
    overwrite: bool = False
    filter_midi_reading_warnings: bool = True
    num_workers: int = 16
    regex: str | None = None
    debug: bool = False
    # When working with files output by musescore (which I am using to quantize the
    #   midi) appears to trim this amount from the duration of every note, so we add it
    #   back in.
    release_delta: float = 0.00208333000000005
    quantize_tpq: int = 96

    def __post_init__(self):
        if self.output_folder is None:
            self.output_folder = self.csv_input_folder


def run_glob(folder_path, ext):
    # Use os.path.join to construct the search pattern
    search_pattern = os.path.join(folder_path, "**", f"*.{ext}")

    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(search_pattern, recursive=True)


def get_files(config: MidiToCSVConfig):

    csv_files = run_glob(config.csv_input_folder, "csv")
    csv_files_ids = set(os.path.splitext(os.path.basename(f))[0] for f in csv_files)
    midi_files = run_glob(config.midi_input_folder, "mid")
    midi_files_ids = set(os.path.splitext(os.path.basename(f))[0] for f in midi_files)
    missing_midi_file_ids = midi_files_ids.difference(csv_files_ids)

    missing_midi_files = [
        os.path.join(config.midi_input_folder, f"{x}.mid")
        for x in missing_midi_file_ids
    ]
    assert all(os.path.exists(x) for x in missing_midi_files)
    assert config.output_folder

    output_paths = [
        os.path.join(config.output_folder, f"{x}.csv") for x in missing_midi_file_ids
    ]
    if config.csv_input_folder == config.output_folder:
        assert not any(os.path.exists(f) for f in output_paths)
    else:
        if config.overwrite:
            for f in output_paths:
                if os.path.exists(f):
                    os.remove(f)
        else:
            output_paths = [f for f in output_paths if not os.path.exists(f)]

    return list(zip(missing_midi_files, output_paths))


def do_midi_file(path_tup, config, output_list, error_file_list):
    try:
        # output_basename = (
        #     midi_file.replace(config.input_folder, "")
        #     .lstrip(os.path.sep)
        #     .replace(" ", "_")
        #     .replace(os.path.sep, "+")
        #     .replace(".mid", ".csv")
        # )
        midi_file, output_path = path_tup
        # output_path = os.path.join(
        #     config.output_folder, os.path.basename(midi_file).replace(".mid", ".csv")
        # )
        assert not os.path.exists(output_path)

        if config.filter_midi_reading_warnings:
            warnings.filterwarnings("ignore", message="note_off event")

        music_df = read_midi(midi_file)
        music_df = music_df[music_df.type.isin(config.event_types)]
        music_df = music_df.drop("filename", axis=1)
        if config.release_delta:
            music_df["release"] = music_df["release"] + config.release_delta
        if config.quantize_tpq:
            music_df = quantize_df(music_df, tpq=config.quantize_tpq)
        if config.salami_slice:
            music_df = salami_slice(music_df)
        music_df.to_csv(output_path)

        output_list.append(midi_file)
    except Exception as exc:
        error_file_list.append((midi_file, repr(exc)))


def write_json(config, output_files):
    json_path = os.path.join(
        config.output_folder, "missing_files_to_csvs_source_files.json"
    )
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
        config_path=None, cli_args=sys.argv[1:], config_cls=MidiToCSVConfig
    )
    if config.debug:

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    input_and_output_files = get_files(config)

    if not input_and_output_files:
        print(f"No missing files")
        return

    if config.regex is not None:
        input_and_output_files = [
            (f, o) for (f, o) in input_and_output_files if re.search(config.regex, f)
        ]
    random.seed(config.seed)
    if config.random_files:
        random.shuffle(input_and_output_files)

    input_and_output_files = input_and_output_files[: config.max_files]

    assert config.output_folder
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
                            do_midi_file,
                            config=config,
                            output_list=output_files,
                            error_file_list=error_files,
                        ),
                        input_and_output_files,
                    ),
                    total=len(input_and_output_files),
                )
            )
    else:
        output_files = []
        error_files = []
        for path_tup in input_and_output_files:
            do_midi_file(path_tup, config, output_files, error_files)

    write_json(config, output_files)

    if error_files:
        print("Errors:")
        for path_tup, exception_str in error_files:
            print(f"{path_tup}: {exception_str}")
        print(f"{len(error_files)} total error files")

    assert len(output_files) + len(error_files) == len(input_and_output_files)


if __name__ == "__main__":
    main()
