"""For unlabeled data, we are using the YCAC midi corpus.

But midi data is often un- or strangely quantized. We can use the Musescore quantization
algorithm by reading it into musescore and then out again. The issue with that is that
musescore abbreviates the releases when writing to midi by small amounts, presumably for
performance reasons. If we save the files as xml, however, the note values are
complete. Therefore we do that, then use this script to write to csvs.
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

from music_df.quantize_df import quantize_df
from music_df.read_xml import read_xml
from music_df.salami_slice import salami_slice
from music_df.script_helpers import read_config_oc

# TODO: (Malcolm 2023-10-14) remove doublings in orchestral scores somehow


@dataclass
class XMLToCSVConfig:
    input_folder: str
    output_folder: str
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


def get_xml_files(folder_path):
    # Use os.path.join to construct the search pattern
    search_pattern = os.path.join(folder_path, "**", "*.xml")

    # Use glob.glob with recursive=True to get all matching files
    return glob.glob(search_pattern, recursive=True)


def do_xml_file(xml_file, config, output_list, error_file_list):
    try:
        output_basename = (
            xml_file.replace(config.input_folder, "")
            .lstrip(os.path.sep)
            .replace(" ", "_")
            .replace(os.path.sep, "+")
            .replace(".xml", ".csv")
        )
        output_path = os.path.join(config.output_folder, output_basename)

        if os.path.exists(output_path) and not config.overwrite:
            return

        music_df = read_xml(xml_file)
        music_df = music_df[music_df.type.isin(config.event_types)]
        if config.salami_slice:
            music_df = salami_slice(music_df)
        music_df = music_df.drop("spelling", axis=1)
        music_df.to_csv(output_path)

        print(f"Wrote {output_path}")
        output_list.append(xml_file)
    except Exception as exc:
        error_file_list.append((xml_file, repr(exc)))


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
        config_path=None, cli_args=sys.argv[1:], config_cls=XMLToCSVConfig
    )
    if config.debug:

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    xml_files = get_xml_files(config.input_folder)
    if config.regex is not None:
        xml_files = [f for f in xml_files if re.search(config.regex, f)]
    random.seed(config.seed)
    if config.random_files:
        random.shuffle(xml_files)

    xml_files = xml_files[: config.max_files]
    os.makedirs(config.output_folder, exist_ok=True)
    if config.num_workers > 1:
        manager = Manager()
        output_files = manager.list()
        error_files = manager.list()
        with Pool(config.num_workers) as pool:
            list(
                pool.imap_unordered(
                    partial(
                        do_xml_file,
                        config=config,
                        output_list=output_files,
                        error_file_list=error_files,
                    ),
                    xml_files,
                )
            )
    else:
        output_files = []
        error_files = []
        for xml_file in xml_files:
            do_xml_file(xml_file, config, output_files, error_files)

    write_json(config, output_files)

    if error_files:
        print("Errors:")
        for xml_file, exception_str in error_files:
            print(f"{xml_file}: {exception_str}")
        print(f"{len(error_files)} total error files")


if __name__ == "__main__":
    main()
