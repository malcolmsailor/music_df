import argparse
import ast
import logging
import os
import pdb
import random
import sys
import traceback
from dataclasses import dataclass

import pandas as pd
import yaml
from matplotlib import pyplot as plt

from music_df.plot_piano_rolls.plot_helper import plot_predictions
from music_df.read import read
from music_df.read_csv import read_csv
from music_df.show_scores.show_score import show_score_and_predictions

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook

DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "plot_predictions"))


@dataclass
class Config:
    feature_name: str = ""
    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    make_piano_rolls: bool = True
    make_score_pdfs: bool = True
    output_folder: str = DEFAULT_OUTPUT
    n_examples: int = 1
    random_examples: bool = True


def read_config(config_path):
    with open(config_path) as inf:
        config = Config(**yaml.safe_load(inf))
    return config


def get_csv_path(raw_path: str, config: Config) -> str:
    if config.csv_prefix_to_strip is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    if config.csv_prefix_to_add is not None:
        raw_path = config.csv_prefix_to_add + raw_path
    return raw_path


def get_csv_title(raw_path, config):
    if config.csv_prefix_to_strip is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    out = os.path.splitext(raw_path)[0]
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        required=True,
        help="Metadata csv containing at least the following columns: csv_path, df_indices. Rows should be in one-to-one correspondance with predictions.",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Text file containing predicted tokens, one sequence per line. Rows should be in one-to-one correspondance with metadata.",
    )
    parser.add_argument("--config-file", required=True)
    args = parser.parse_args()
    return args


def make_humdrum(df: pd.DataFrame):
    CHAR_MAPPER = {"no": "N"}
    char_map_f = lambda x: CHAR_MAPPER.get(x, "")


def handle_predictions(
    predictions_path,
    metadata_csv,
    config,
    feature_name=None,
    indices: None | list[int] = None,
):
    with open(predictions_path) as inf:
        predictions_list = inf.readlines()

    assert len(metadata_csv) == len(predictions_list)

    if indices is not None:
        # get rows pointed to by indices from metadata_csv
        metadata_csv = metadata_csv.iloc[indices]
        predictions_list = [predictions_list[i] for i in indices]

    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None
    for (_, metadata_row), preds_str in zip(metadata_csv.iterrows(), predictions_list):
        predictions = preds_str.split()

        if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
            prev_csv_path = metadata_row.csv_path
            assert isinstance(prev_csv_path, str)
            LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
            music_df = read_csv(get_csv_path(prev_csv_path, config))

        assert music_df is not None

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        title = get_csv_title(prev_csv_path, config)
        if "start_offset" in metadata_row.index:
            title += f" {metadata_row.start_offset}"
        else:
            title += f" {metadata_row.name}"

        if config.make_score_pdfs:
            pdf_basename = (
                (title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_"))
                + f"_{feature_name if feature_name is not None else config.feature_name}.pdf"
            )
            pdf_path = os.path.join(config.output_folder, pdf_basename)
            show_score_and_predictions(
                music_df, config.feature_name, predictions, df_indices, pdf_path
            )
            LOGGER.info(f"Wrote {pdf_path}")
        if config.make_piano_rolls:
            fig, ax = plt.subplots()
            # TODO: (Malcolm 2023-09-29) save to a png rather than displaying
            plot_predictions(
                music_df,
                feature_name if feature_name is not None else config.feature_name,
                predictions,
                df_indices,
                ax=ax,
                title=title,
            )
            plt.show()
        # if input("print another y/n? ").lower().strip() != "y":
        #     break
        break


def main():
    args = parse_args()
    config = read_config(args.config_file)
    if not config.make_score_pdfs or config.make_piano_rolls:
        print("Nothing to do!")
        sys.exit(1)

    metadata_csv = pd.read_csv(args.metadata)

    if config.random_examples:
        indices = random.sample(range(len(metadata_csv)), k=config.n_examples)
    else:
        indices = list(range(config.n_examples))

    # check if args.predictions is a directory
    if os.path.isdir(args.predictions):
        for predictions_path in os.listdir(args.predictions):
            feature_name = os.path.splitext(predictions_path)[0]
            predictions_path = os.path.join(args.predictions, predictions_path)
            handle_predictions(
                predictions_path,
                metadata_csv,
                config,
                feature_name=feature_name,
                indices=indices,
            )
    else:
        handle_predictions(args.predictions, metadata_csv, config, indices=indices)


if __name__ == "__main__":
    main()
