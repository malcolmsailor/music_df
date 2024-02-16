import ast
import logging
import os
import pdb
import sys
import traceback

import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from music_df.plot_piano_rolls.plot_helper import plot_predictions
from music_df.read_csv import read_csv
from music_df.show_scores.show_score import show_score_and_predictions
from music_df.sync_df import sync_array_by_df

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_debug_hook():
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)

    sys.excepthook = custom_excepthook


def softmax(a):
    z = np.exp(a)
    return z / np.sum(z, axis=-1, keepdims=True)


def read_config_oc(config_path: str | None, cli_args: list[str] | None, config_cls):
    configs = []
    assert config_path is not None or cli_args is not None
    if config_path is not None:
        configs.append(OmegaConf.load(config_path))
    if cli_args is not None:
        configs.append(OmegaConf.from_cli(cli_args))
    merged_conf = OmegaConf.merge(*configs)
    out = config_cls(**merged_conf)
    if getattr(out, "debug", False):
        set_debug_hook()
    return out


def read_config(config_path, config_cls):
    with open(config_path) as inf:
        config = config_cls(**yaml.safe_load(inf))
    if config.debug:
        set_debug_hook()
    return config


def get_csv_path(raw_path: str, config) -> str:
    if getattr(config, "csv_prefix_to_strip", None) is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    if getattr(config, "csv_prefix_to_add", None) is not None:
        raw_path = config.csv_prefix_to_add + raw_path
    return raw_path


def get_csv_title(raw_path, config) -> str:
    if getattr(config, "csv_prefix_to_strip", None) is not None:
        raw_path = raw_path.replace(config.csv_prefix_to_strip, "", 1)
    out = os.path.splitext(raw_path)[0]
    return out


def get_itos(dictionary_paths: list[str] | str) -> dict[str, list[str]]:
    if isinstance(dictionary_paths, str):
        dictionary_paths = [dictionary_paths]
    out = {}
    for dictionary_path in dictionary_paths:
        feature_name = os.path.basename(dictionary_path).rsplit("_", maxsplit=1)[0]
        with open(dictionary_path) as inf:
            data = inf.readlines()
        contents = [
            line.split(" ", maxsplit=1)[0]
            for line in data
            if line and not line.startswith("madeupword")
        ]
        out[feature_name] = contents
    return out


def get_stoi(dictionary_paths: list[str]) -> dict[str, list[str]]:
    out = {}
    for dictionary_path in dictionary_paths:
        feature_name = os.path.basename(dictionary_path).rsplit("_", maxsplit=1)[0]
        with open(dictionary_path) as inf:
            data = inf.readlines()
        contents = [
            line.split(" ", maxsplit=1)[0]
            for line in data
            if line and not line.startswith("madeupword")
        ]
        out[feature_name] = {token: i for i, token in enumerate(contents)}
    return out


def plot_item_from_logits(
    metadata_row,
    logits,
    config,
    feature_vocab,
    music_df: pd.DataFrame | None = None,
    title: str | None = None,
    entropy_to_transparency: bool = False,
    keep_intermediate_files: bool = False,
    write_csv: bool = False,
    feature_name: str | None = None,
    sync: bool = False,
    label_every_nth_note: int | None = None,
):
    if getattr(config, "data_has_start_and_stop_tokens", False):
        logits = logits[1:-1]
    if music_df is None:
        music_df = read_csv(get_csv_path(metadata_row.csv_path, config))
        assert music_df is not None
    if title is None:
        title = get_csv_title(metadata_row.csv_path, config)

    df_indices = metadata_row.df_indices
    if isinstance(df_indices, str):
        df_indices = ast.literal_eval(df_indices)

    if "start_offset" in metadata_row.index:
        title += f" {metadata_row.start_offset}"
    else:
        title += f" {metadata_row.name}"

    subfolder = (
        title.strip(os.path.sep).replace(os.path.sep, "+").replace(" ", "_") + "_synced"
    )

    # This former strategy for cropping led to incorrect results sometimes:
    # cropped_df = crop_df(music_df, start_i=min(df_indices), end_i=max(df_indices))
    cropped_df = music_df.loc[df_indices]
    assert cropped_df.type.unique().tolist() == ["note"]

    notes_df = cropped_df.reset_index(drop=True)

    # In case logits were ragged, only take the logits corresponding to notes
    logits = logits[: len(notes_df)]

    # if feature_name in ONSET_LEVEL_FEATURES:
    if sync:
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
        feature_name = feature_name if feature_name is not None else config.feature_name
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
            keep_intermediate_files=keep_intermediate_files,
            label_every_nth_note=label_every_nth_note,
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
