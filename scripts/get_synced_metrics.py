import argparse
import ast
import csv
import glob
import logging
import os
from dataclasses import dataclass, field

import h5py
import pandas as pd

from music_df import quantize_df, read_csv
from music_df.crop_df import crop_df
from music_df.salami_slice import (
    appears_salami_sliced,
    get_unique_salami_slices,
    slice_into_uniform_steps,
)
from music_df.script_helpers import get_csv_path, get_itos, get_stoi, read_config_oc
from music_df.sync_df import get_unique_from_array_by_df, sync_array_by_df

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT = os.path.expanduser(os.path.join("~", "output", "metrics"))


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: h5 file containing predicted tokens. Rows
    # should be in one-to-one correspondance with metadata.
    predictions: str
    dictionary_folder: str | None = None
    debug: bool = False
    # When predicting tokens we need to subtract the number of specials
    n_specials: int = 4
    data_has_start_and_stop_tokens: bool = False
    features: tuple[str, ...] = (
        "harmony_onset",
        "primary_degree",
        "primary_alteration",
        "secondary_degree",
        "secondary_alteration",
        "inversion",
        "mode",
        "quality",
        "key_pc",
    )
    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    column_types: dict[str, str] = field(default_factory=lambda: {})
    output_folder: str = DEFAULT_OUTPUT
    uniform_step: None | int = 8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    # remaining passed through to omegaconf
    parser.add_argument("remaining", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    return args


def process_h5(h5_path, metadata_df, config, feature_name, stoi):
    h5file = h5py.File(h5_path, mode="r")

    output_path = os.path.join(config.output_folder, f"{feature_name}.csv")
    outf = open(output_path, "w", newline="")
    writer = csv.writer(outf)
    writer.writerow(["path", "indices", "uniform_steps", "labels", "predicted"])

    assert len(metadata_df) >= len(h5file)
    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None

    for i in range(len(h5file)):
        metadata_row = metadata_df.iloc[i]
        logits: np.ndarray = (h5file[f"logits_{i}"])[:]  # type:ignore
        # TODO: (Malcolm 2023-11-27) aggregate logits across entire score

        if config.data_has_start_and_stop_tokens:
            logits = logits[1:-1]

        if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
            prev_csv_path = metadata_row.csv_path
            assert isinstance(prev_csv_path, str)
            print(".", end="", flush=True)
            # LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
            music_df = read_csv(get_csv_path(prev_csv_path, config))
            if music_df is None:
                continue
            assert appears_salami_sliced(music_df)

        if music_df is None:
            continue

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        if min(df_indices) != df_indices[0]:
            continue

        # This former strategy for cropping led to incorrect results sometimes:
        # cropped_df = crop_df(music_df, start_i=min(df_indices), end_i=max(df_indices))
        cropped_df = music_df.loc[df_indices]
        assert cropped_df.type.unique().tolist() == ["note"]

        notes_df = cropped_df.reset_index(drop=True)
        unique_slices = get_unique_salami_slices(notes_df)
        labels = unique_slices[feature_name]
        label_indices = [stoi[str(label)] for label in labels]

        # Trim pad tokens
        logits = logits[: len(notes_df)]

        if len(logits) < len(notes_df):
            LOGGER.error(f"{metadata_row.csv_path} length of logits < length of notes")
            continue
        logits = get_unique_from_array_by_df(
            logits,
            notes_df,
            unique_col_name_or_names="onset",
            sync_col_name_or_names="onset",
        )

        predicted_indices = logits.argmax(axis=-1)

        predicted_indices -= config.n_specials
        if predicted_indices.min() < 0:
            LOGGER.warning(
                f"Predicted at least one special token in {metadata_row.csv_path}; "
                "replacing with 0"
            )
            predicted_indices[predicted_indices < 0] = 0

        if config.uniform_step:
            # notes_df["predicted_indices"] = predicted_indices
            unique_slices = quantize_df(
                unique_slices,
                tpq=config.uniform_step,
                ticks_out=True,
                zero_dur_action="preserve",
            )
            uniform_steps = (unique_slices["release"] - unique_slices["onset"]).tolist()
            assert len(uniform_steps) == len(label_indices) == len(predicted_indices)
        else:
            uniform_steps = None
            assert len(label_indices) == len(predicted_indices)

        writer.writerow(
            [
                prev_csv_path,
                df_indices,
                uniform_steps,
                label_indices,
                predicted_indices.tolist(),
            ]
        )


def main():
    args = parse_args()
    config = read_config_oc(args.config_file, args.remaining, Config)
    metadata_df = pd.read_csv(config.metadata)
    os.makedirs(config.output_folder, exist_ok=True)
    if os.path.isdir(config.predictions):
        if config.dictionary_folder is None:
            raise ValueError
        else:
            dictionary_paths = glob.glob(
                os.path.join(config.dictionary_folder, "*_dictionary.txt")
            )
            stoi_vocabs = get_stoi(dictionary_paths)
            for predictions_path in glob.glob(os.path.join(config.predictions, "*.h5")):
                this_feature_name = os.path.basename(
                    os.path.splitext(predictions_path)[0]
                )
                if this_feature_name not in config.features:
                    continue
                # if (
                #     config.feature_names
                #     and this_feature_name not in config.feature_names
                # ):
                #     continue
                process_h5(
                    predictions_path,
                    metadata_df,
                    config,
                    this_feature_name,
                    stoi_vocabs[this_feature_name],
                )


if __name__ == "__main__":
    main()
