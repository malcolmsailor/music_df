import argparse
import ast
import glob
import logging
import os
import pdb
import random
import sys
import traceback
from dataclasses import dataclass, field

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from music_df.crop_df import crop_df
from music_df.plot_piano_rolls.plot_helper import plot_predictions
from music_df.read_csv import read_csv
from music_df.script_helpers import (
    get_csv_path,
    get_csv_title,
    get_itos,
    read_config_oc,
)
from music_df.show_scores.show_score import show_score_and_predictions
from music_df.sync_df import sync_array_by_df

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "plot_predictions"))

# features are either note-level (like "chord_tone") or onset-level (like "primary_degree")
ONSET_LEVEL_FEATURES = {
    "harmony_onset",
    "primary_degree",
    "primary_alteration",
    "secondary_degree",
    "secondary_alteration",
    "inversion",
    "mode",
    "quality",
    "key_pc",
}


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: path to a folder with either an .h5 file or a .txt file containing
    # predicted tokens. Rows should be in one-to-one correspondance with metadata.
    predictions: str
    dictionary_folder: str | None = None
    # regex to filter score ids
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
    sync_onsets: bool = True
    # When predicting tokens we need to subtract the number of specials
    n_specials: int = 4
    data_has_start_and_stop_tokens: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    # remaining passed through to omegaconf
    parser.add_argument("remaining", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def softmax(a):
    z = np.exp(a)
    return z / np.sum(z, axis=-1, keepdims=True)


def sync_predictions(
    h5_path,
    metadata_df,
    config,
    feature_name,
    feature_vocab,
    write_csv=False,
    indices: None | list[int] = None,
    entropy_to_transparency: bool = True,
):
    h5file = h5py.File(h5_path, mode="r")

    assert len(metadata_df) >= len(h5file)

    if indices is None:
        indices = list(range(len(h5file)))

    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None
    for i in indices:
        metadata_row = metadata_df.iloc[i]
        logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:ignore

        if config.data_has_start_and_stop_tokens:
            logits = logits[1:-1]

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

        subfolder = (
            title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_")
            + "_synced"
        )

        # This former strategy for cropping led to incorrect results sometimes:
        # cropped_df = crop_df(music_df, start_i=min(df_indices), end_i=max(df_indices))
        cropped_df = music_df.loc[df_indices]
        assert cropped_df.type.unique().tolist() == ["note"]

        notes_df = cropped_df.reset_index(drop=True)

        # In case logits were ragged, only take the logits corresponding to notes
        logits = logits[: len(notes_df)]

        if feature_name in ONSET_LEVEL_FEATURES:
            logits = sync_array_by_df(logits, notes_df, sync_col_name_or_names="onset")

        if entropy_to_transparency:
            probs = softmax(logits)
            entropy = -np.sum(probs * np.log2(probs), axis=1)
        else:
            entropy = None

        predicted_indices = logits.argmax(axis=-1)

        predicted_indices -= config.n_specials
        if predicted_indices.min() < 0:
            LOGGER.warning(
                f"Predicted at least one special token in {metadata_row.csv_path}; "
                "replacing with 0"
            )
            predicted_indices[predicted_indices < 0] = 0

        predictions = [feature_vocab[i] for i in predicted_indices]
        if config.make_score_pdfs:
            feature_name = (
                feature_name if feature_name is not None else config.feature_name
            )
            pdf_basename = f"{feature_name}.pdf"
            pdf_path = os.path.join(config.output_folder, subfolder, pdf_basename)
            csv_path = pdf_path[:-4] + ".csv"
            return_code = show_score_and_predictions(
                music_df=music_df,
                feature_name=feature_name,
                predicted_feature=predictions,
                prediction_indices=df_indices,
                pdf_path=pdf_path,
                csv_path=csv_path if write_csv else None,
                col_type=config.column_types.get(feature_name, str),
                entropy=entropy,
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
    h5file.close()


def handle_predictions(
    predictions_path,
    metadata_df,
    config,
    feature_name=None,
    write_csv=False,
    indices: None | list[int] = None,
):
    with open(predictions_path) as inf:
        predictions_list = inf.readlines()

    assert len(metadata_df) == len(predictions_list)

    if indices is not None:
        # get rows pointed to by indices from metadata_df
        metadata_df = metadata_df.iloc[indices]
        predictions_list = [predictions_list[i] for i in indices]

    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None
    for (_, metadata_row), preds_str in zip(metadata_df.iterrows(), predictions_list):
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
                music_df=music_df,
                feature_name=feature_name,
                predicted_feature=predictions,
                prediction_indices=df_indices,
                pdf_path=pdf_path,
                csv_path=csv_path if write_csv else None,
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

    metadata_df = pd.read_csv(config.metadata)

    indices = None
    if config.filter_scores is not None:
        indices = list(
            np.nonzero(
                metadata_df.score_id.str.contains(config.filter_scores)
                | metadata_df.score_path.str.contains(config.filter_scores)
                | metadata_df.csv_path.str.contains(config.filter_scores)
            )[0]
        )
        if not indices:
            raise ValueError(f"No scores match pattern {config.filter_scores}")

    if indices is None:
        if config.random_examples:
            random.seed(config.seed)
            indices = random.sample(range(len(metadata_df)), k=config.n_examples)
        else:
            indices = list(range(config.n_examples))
    else:
        if config.random_examples:
            random.seed(config.seed)
            indices = random.sample(indices, k=config.n_examples)
        else:
            indices = indices[: config.n_examples]

    args = []

    if os.path.isdir(config.predictions):
        if config.sync_onsets:
            if config.dictionary_folder is None:
                raise ValueError("must provide dictionary folder if syncing onsets")
            else:
                dictionary_paths = glob.glob(
                    os.path.join(config.dictionary_folder, "*_dictionary.txt")
                )
                vocabs = get_itos(dictionary_paths)
            for predictions_path in glob.glob(os.path.join(config.predictions, "*.h5")):
                this_feature_name = os.path.basename(
                    os.path.splitext(predictions_path)[0]
                )
                if (
                    config.feature_names
                    and this_feature_name not in config.feature_names
                ):
                    continue
                sync_predictions(
                    predictions_path,
                    metadata_df,
                    config,
                    feature_name=this_feature_name,
                    feature_vocab=vocabs[this_feature_name],
                    indices=indices,
                    write_csv=config.write_csv,
                )
        else:
            for predictions_path in glob.glob(
                os.path.join(config.predictions, "*.txt")
            ):
                if os.path.basename(predictions_path).startswith("metadata"):
                    continue
                this_feature_name = os.path.basename(
                    os.path.splitext(predictions_path)[0]
                )
                if (
                    config.feature_names
                    and this_feature_name not in config.feature_names
                ):
                    continue
                handle_predictions(
                    predictions_path,
                    metadata_df,
                    config,
                    feature_name=this_feature_name,
                    indices=indices,
                    write_csv=config.write_csv,
                )
    else:
        handle_predictions(config.predictions, metadata_df, config, indices=indices)


if __name__ == "__main__":
    main()
