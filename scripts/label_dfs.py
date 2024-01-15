import argparse
import ast
import logging
import os
import pdb
import sys
import traceback
from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from music_df.label_df import label_df
from music_df.read_csv import read_csv
from music_df.script_helpers import get_csv_path, read_config_oc

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


@dataclass
class Config:
    # metadata: path to metadata csv containing at least the following columns:
    # csv_path, df_indices. Rows should be in one-to-one correspondance with
    # labels_path.
    metadata_path: str
    # labels_path: path or paths to .txt file(s) containing labels
    labels_path: str | Sequence[str]
    output_folder: str
    dictionary_folder: str | None = None
    # regex to filter score ids
    filter_scores: str | None = None
    csv_prefix_to_strip: None | str = None
    csv_prefix_to_add: None | str = None
    feature_name: str = "label"
    debug: bool = False


def handle_labels(
    metadata_df: pd.DataFrame,
    config: Config,
    feature_name: str | None = None,
    indices: None | list[int] = None,
):
    labels_paths = (
        [config.labels_path]
        if isinstance(config.labels_path, str)
        else config.labels_path
    )
    labels_lists = []
    for labels_path in labels_paths:
        with open(labels_path) as inf:
            labels_list = inf.readlines()

        assert len(metadata_df) == len(labels_list)

        if indices is not None:
            # get rows pointed to by indices from metadata_df
            metadata_df = metadata_df.iloc[indices]
            labels_list = [labels_list[i] for i in indices]

        labels_lists.append(labels_list)

    prev_csv_path: None | str = None
    music_df: pd.DataFrame | None = None

    assert isinstance(metadata_df.index, pd.RangeIndex) and metadata_df.index.start == 0
    for row_i, metadata_row in metadata_df.iterrows():
        if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
            prev_csv_path = metadata_row.csv_path
            assert isinstance(prev_csv_path, str)
            LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
            music_df = read_csv(get_csv_path(prev_csv_path, config))

        assert music_df is not None

        df_indices = metadata_row.df_indices
        if isinstance(df_indices, str):
            df_indices = ast.literal_eval(df_indices)

        feature_name = feature_name if feature_name is not None else config.feature_name
        for label_i, labels_list in enumerate(labels_lists):
            labels_str = labels_list[row_i]
            labels = labels_str.split()

            if isinstance(feature_name, str):
                if len(labels_lists) > 1:
                    this_feature_name = f"{feature_name}_{label_i}"
                else:
                    this_feature_name = feature_name
            else:
                this_feature_name = feature_name[label_i]

            music_df = label_df(
                music_df,
                labels=labels,
                label_indices=df_indices,
                label_col_name=this_feature_name,
            )
        os.makedirs(config.output_folder, exist_ok=True)
        output_path = os.path.join(
            config.output_folder, os.path.basename(metadata_row.csv_path)
        )
        music_df.to_csv(output_path)
        print(f"Wrote {output_path}")
        if config.debug:
            break
    # for (_, metadata_row), labels_str in zip(metadata_df.iterrows(), labels_list):
    #     labels = labels_str.split()

    #     if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
    #         prev_csv_path = metadata_row.csv_path
    #         assert isinstance(prev_csv_path, str)
    #         LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
    #         music_df = read_csv(get_csv_path(prev_csv_path, config))

    #     assert music_df is not None

    #     df_indices = metadata_row.df_indices
    #     if isinstance(df_indices, str):
    #         df_indices = ast.literal_eval(df_indices)
    #     feature_name = feature_name if feature_name is not None else config.feature_name
    #     labeled_df = label_df(
    #         music_df,
    #         labels=labels,
    #         label_indices=df_indices,
    #         label_col_name=feature_name,
    #     )
    #     os.makedirs(config.output_folder, exist_ok=True)
    #     output_path = os.path.join(
    #         config.output_folder, os.path.basename(metadata_row.csv_path)
    #     )
    #     labeled_df.to_csv(output_path)
    #     print(f"Wrote {output_path}")
    #     if config.debug:
    #         break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file")
    args, remaining = parser.parse_known_args()
    return args, remaining


def main():
    args, remaining = parse_args()
    config = read_config_oc(args.config_file, remaining, Config)
    if config.debug:
        sys.excepthook = custom_excepthook
    metadata_df = pd.read_csv(config.metadata_path)
    handle_labels(metadata_df, config)


if __name__ == "__main__":
    main()
