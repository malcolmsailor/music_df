import glob
import json
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from multiprocessing import Manager, Pool
from typing import Iterable

from music_df.read_midi import read_midi
from music_df.script_helpers import read_config_oc

# TODO: (Malcolm 2023-10-14) round releases somehow
# TODO: (Malcolm 2023-10-14) remove doublings in orchestral scores somehow


@dataclass
class MidiToCSVConfig:
    input_folder: str
    output_folder: str
    max_files: int | None = None
    random_files: bool = False
    seed: int = 42
    event_types: Iterable[str] = field(
        default_factory=lambda: {"note", "time_signature", "bar"}
    )
    overwrite: bool = False
    filter_midi_reading_warnings: bool = True
    num_workers: int = 8


def get_midi_files(folder_path):
    # Use os.path.join to construct the search pattern
    search_pattern = os.path.join(folder_path, "**", "*.mid")

    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(search_pattern, recursive=True)


def do_midi_file(midi_file, config, output_list):
    output_basename = (
        midi_file.replace(config.input_folder, "")
        .lstrip(os.path.sep)
        .replace(" ", "_")
        .replace(os.path.sep, "+")
        .replace(".mid", ".csv")
    )
    output_path = os.path.join(config.output_folder, output_basename)

    if os.path.exists(output_path) and not config.overwrite:
        return

    if config.filter_midi_reading_warnings:
        warnings.filterwarnings("ignore", message="note_off event")

    music_df = read_midi(midi_file)
    music_df = music_df[music_df.type.isin(config.event_types)]
    music_df.to_csv(output_path)
    music_df = music_df.drop("filename", axis=1)

    print(f"Wrote {output_path}")
    output_list.append(midi_file)


def write_json(config, output_files):
    json_path = os.path.join(config.output_folder, "source_files.json")
    if not config.overwrite and os.path.exists(json_path):
        with open(json_path, "r") as inf:
            existing_contents = json.load(inf)
        with open(json_path, "w") as outf:
            json.dump(existing_contents + list(output_files), outf)
    else:
        with open(json_path, "w") as outf:
            json.dump(list(output_files), outf)


def main():
    config = read_config_oc(
        config_path=None, cli_args=sys.argv[1:], config_cls=MidiToCSVConfig
    )
    midi_files = get_midi_files(config.input_folder)
    random.seed(config.seed)
    if config.random_files:
        random.shuffle(midi_files)

    midi_files = midi_files[: config.max_files]
    os.makedirs(config.output_folder, exist_ok=True)
    if config.num_workers > 1:
        manager = Manager()
        output_files = manager.list()
        with Pool(config.num_workers) as pool:
            list(
                pool.imap_unordered(
                    partial(do_midi_file, config=config, output_list=output_files),
                    midi_files,
                )
            )
    else:
        output_files = []
        for midi_file in midi_files:
            do_midi_file(midi_file, config, output_files)


if __name__ == "__main__":
    main()
