import argparse
import logging
import os
import pdb
import random
import re
import sys
import traceback
from dataclasses import dataclass, field

from music_df import read_csv
from music_df.script_helpers import read_config
from music_df.show_scores.show_score import show_score

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook

DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "plot_scores"))


@dataclass
class Config:
    feature_names: list[str] = field(default_factory=lambda: [])
    make_piano_rolls: bool = True
    make_score_pdfs: bool = True
    output_folder: str = DEFAULT_OUTPUT
    n_examples: int = 1
    random_examples: bool = True
    column_types: dict[str, str] = field(default_factory=lambda: {})


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder containing csv files",
    )
    parser.add_argument("--filter-scores", type=str, help="regex to filter score ids")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--write-csv", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def find_csv_files(directory):
    csv_files = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".csv"):
                full_path = os.path.join(dirpath, filename)
                csv_files.append(full_path)

    return csv_files


def get_csv_title(raw_path, input_folder):
    raw_path = raw_path.replace(input_folder, "", 1)
    out = os.path.splitext(raw_path)[0]
    return out


def main():
    args = parse_args()
    config = read_config(args.config_file, Config)
    if not config.make_score_pdfs or config.make_piano_rolls:
        print("Nothing to do!")
        sys.exit(1)

    csv_files = find_csv_files(args.input_folder)

    if args.filter_scores is not None:
        csv_files = [
            csv_file
            for csv_file in csv_files
            if re.search(args.filter_scores, csv_file)
        ]

    random.seed(args.seed)
    if config.random_examples:
        random.shuffle(csv_files)

    csv_files = csv_files[: config.n_examples]

    for csv_file in csv_files:
        music_df = read_csv(csv_file)
        title = get_csv_title(csv_file, args.input_folder)
        subfolder = title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_")
        for feature_name in config.feature_names:
            if config.make_score_pdfs:
                # pdf_basename = (
                #     title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_")
                # ) + f"_{feature_name}.pdf"
                pdf_basename = f"{feature_name}.pdf"
                pdf_path = os.path.join(config.output_folder, subfolder, pdf_basename)
                # pdf_path = os.path.join(config.output_folder, pdf_basename)
                return_code = show_score(music_df, feature_name, pdf_path)
                if not return_code:
                    LOGGER.info(f"Wrote {pdf_path}")
            if config.make_piano_rolls:
                raise NotImplementedError


if __name__ == "__main__":
    main()
