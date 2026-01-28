import argparse
import glob
import os
import shutil
from ast import literal_eval
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Literal

try:
    import h5py
except ImportError as e:
    raise ImportError(
        "h5py is required for this script. "
        "Install with: pip install music_df[scripts]"
    ) from e
import numpy as np
import pandas as pd
from tqdm import tqdm

from music_df.script_helpers import read_config_oc


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # predictions.
    metadata: str
    # predictions: path to a folder with either an .h5 file or a .txt file containing
    # predicted tokens. Rows should be in one-to-one correspondance with metadata.
    predictions: str
    output_folder: str
    feature_names: list[str] = field(default_factory=lambda: [])
    overwrite: bool = False
    # column_types: dict[str, str] = field(default_factory=lambda: {})
    debug: bool = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    args, remaining = parser.parse_known_args()
    return args, remaining


def handle_metadata(metadata_rows, reference_df: pd.DataFrame | None, config: Config):
    out_metadata_df = pd.DataFrame(metadata_rows)
    if reference_df is None:
        df_path = os.path.join(config.output_folder, os.path.basename(config.metadata))
        out_metadata_df.to_csv(df_path)
        print(f"Wrote {df_path}")
        return out_metadata_df
    else:
        assert out_metadata_df.equals(reference_df)
        return reference_df


def merge_logits(logits_list: list[np.ndarray], indices: list[str]):
    weights = np.array([i.count(",") + 1 for i in indices])
    weights = weights / weights.sum()
    all_logits = np.stack(logits_list, axis=-1)
    logits = (all_logits * weights).sum(axis=-1)
    return logits


def main():
    args, remaining = parse_args()
    config = read_config_oc(args.config_file, remaining, Config)
    metadata_df = pd.read_csv(config.metadata, index_col=0).reset_index(drop=True)

    # (Malcolm 2024-01-08) There's no reason to be predicting on augmented
    #   data, which might lead to headaches.
    if "transpose" in metadata_df.columns:
        assert (metadata_df["transpose"] == 0).all()
    if "scaled_by" in metadata_df.columns:
        assert (metadata_df["scaled_by"] == 1.0).all()

    if os.path.exists(config.output_folder):
        if config.overwrite:
            shutil.rmtree(config.output_folder)
        else:
            raise ValueError(f"Output folder {config.output_folder} already exists")
    os.makedirs(config.output_folder)

    unique_scores = metadata_df.score_id.unique()

    reference_out_metadata_df = None
    predictions_paths = glob.glob(os.path.join(config.predictions, "*.h5"))

    for predictions_path in predictions_paths:
        print(f"Handling {predictions_path}")

        metadata_rows = []
        out_predictions = []
        h5file = h5py.File(predictions_path, mode="r")
        for score in tqdm(unique_scores):
            score_rows = metadata_df[metadata_df.score_id == score]

            score_predictions: list[np.ndarray] = [
                (h5file[f"logits_{i}"])[:] for i in score_rows.index  # type:ignore
            ]

            merged_logits = merge_logits(
                score_predictions,
                score_rows.df_indices.tolist(),
            )

            metadata_row = score_rows.iloc[0].copy()
            metadata_rows.append(metadata_row)
            out_predictions.append(merged_logits)
            if config.debug:
                break
        # TODO: (Malcolm 2024-01-16) is this necessary?
        reference_out_metadata_df = handle_metadata(
            metadata_rows, reference_out_metadata_df, config
        )

        out_preds_path = os.path.join(
            config.output_folder, "predictions", os.path.basename(predictions_path)
        )
        os.makedirs(os.path.dirname(out_preds_path), exist_ok=True)
        h5outf = h5py.File(out_preds_path, "w")
        for logit_i, example in enumerate(out_predictions):
            h5outf.create_dataset(f"logits_{logit_i}", data=example)
        print(f"Wrote {out_preds_path}")
        if config.debug:
            break


if __name__ == "__main__":
    main()
