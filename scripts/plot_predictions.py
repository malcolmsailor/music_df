import argparse
import ast
import logging
import os
import pdb
import random
import sys
import traceback
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from music_df.plot_piano_rolls.plot_helper import plot_predictions
from music_df.read_csv import read_csv
from music_df.script_helpers import get_csv_path, get_csv_title, read_config_oc
from music_df.show_scores.show_score import show_score_and_predictions

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "plot_predictions"))


@dataclass
class Config:
    metadata: str
    predictions: str
    filter_scores: str | None = None
    feature_names: list[str] = field(default_factory=lambda: [])
    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    make_piano_rolls: bool = True
    make_score_pdfs: bool = True
    write_csv: bool = False
    seed: int = 42
    output_folder: str = DEFAULT_OUTPUT
    n_examples: int = 1
    random_examples: bool = True
    column_types: dict[str, str] = field(default_factory=lambda: {})
    debug: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--metadata",
    #     required=True,
    #     help="Metadata csv containing at least the following columns: csv_path, df_indices. Rows should be in one-to-one correspondance with predictions.",
    # )
    # parser.add_argument(
    #     "--predictions",
    #     required=True,
    #     help="Text file containing predicted tokens, one sequence per line. Rows should be in one-to-one correspondance with metadata.",
    # )
    # parser.add_argument("--filter-scores", type=str, help="regex to filter score ids")
    parser.add_argument("--config-file", required=True)
    parser.add_argument("remaining", nargs=argparse.REMAINDER)
    # parser.add_argument("--write-csv", action="store_true")
    # parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    return args


def handle_predictions(
    predictions_path,
    metadata_csv,
    config,
    feature_name=None,
    write_csv=False,
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

        subfolder = title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_")

        if config.make_score_pdfs:
            feature_name = (
                feature_name if feature_name is not None else config.feature_name
            )
            pdf_basename = f"{feature_name}.pdf"
            pdf_path = os.path.join(config.output_folder, subfolder, pdf_basename)
            csv_path = pdf_path[:-4] + ".csv"
            return_code = show_score_and_predictions(
                music_df,
                feature_name,
                predictions,
                df_indices,
                pdf_path,
                csv_path if write_csv else None,
                col_type=config.column_types.get(feature_name, str),
            )
            if not return_code:
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


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)
    if not config.make_score_pdfs or config.make_piano_rolls:
        print("Nothing to do!")
        sys.exit(1)

    if config.debug:

        def custom_excepthook(exc_type, exc_value, exc_traceback):
            traceback.print_exception(
                exc_type, exc_value, exc_traceback, file=sys.stdout
            )
            pdb.post_mortem(exc_traceback)

        sys.excepthook = custom_excepthook

    metadata_csv = pd.read_csv(config.metadata)

    indices = None
    if config.filter_scores is not None:
        indices = list(
            np.nonzero(
                metadata_csv.score_id.str.contains(config.filter_scores)
                | metadata_csv.score_path.str.contains(config.filter_scores)
                | metadata_csv.csv_path.str.contains(config.filter_scores)
            )[0]
        )
        if not indices:
            raise ValueError(f"No scores match pattern {config.filter_scores}")

    if indices is None:
        if config.random_examples:
            random.seed(config.seed)
            indices = random.sample(range(len(metadata_csv)), k=config.n_examples)
        else:
            indices = list(range(config.n_examples))
    else:
        if config.random_examples:
            random.seed(config.seed)
            indices = random.sample(indices, k=config.n_examples)
        else:
            indices = indices[: config.n_examples]

    # check if config.predictions is a directory
    args = []
    if os.path.isdir(config.predictions):
        for predictions_path in os.listdir(config.predictions):
            this_feature_name = os.path.splitext(predictions_path)[0]
            if config.feature_names and this_feature_name not in config.feature_names:
                continue
            predictions_path = os.path.join(config.predictions, predictions_path)
            handle_predictions(
                predictions_path,
                metadata_csv,
                config,
                feature_name=this_feature_name,
                indices=indices,
                write_csv=config.write_csv,
            )
    else:
        handle_predictions(config.predictions, metadata_csv, config, indices=indices)


if __name__ == "__main__":
    main()
