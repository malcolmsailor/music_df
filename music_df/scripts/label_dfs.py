import argparse
import ast
from functools import partial
import logging
import os
import pdb
import random
import sys
import traceback
from dataclasses import dataclass
from typing import Sequence
import multiprocessing
import pandas as pd
from tqdm import tqdm

from music_df.label_df import label_df
from music_df.read_csv import read_csv
from music_df.script_helpers import get_csv_path, read_config_oc

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type != KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


def any_input_is_newer(input_paths, output_paths, verbose=False):
    newest_input = max(os.path.getmtime(p) for p in input_paths)

    # We take max(os.path.getmtime(p), os.path.getctime(p)) because in the case
    #   where the file is copied, its modification time will not be updated
    existing_output_paths = [p for p in output_paths if os.path.exists(p)]

    if not existing_output_paths:
        return True

    oldest_output = min(
        max(os.path.getmtime(p), os.path.getctime(p)) for p in existing_output_paths
    )

    if newest_input > oldest_output:
        newest_file = sorted(input_paths, key=os.path.getmtime)[-1]
        oldest_output_file = sorted(
            output_paths, key=lambda x: max(os.path.getmtime(x), os.path.getctime(x))
        )[0]
        if verbose:
            print(f"{newest_file} is newer than {oldest_output_file}")

    return newest_input > oldest_output


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
    max_rows: int | None = None  # TODO: (Malcolm 2024-01-27) implement
    row_p: float | None = 1.0
    verbose: bool = False
    num_workers: int = 8
    multiprocess_chunk_size: int = 8


def process_row(metadata_tup, *, config, labels_paths, feature_name, labels_lists):
    row_i, metadata_row = metadata_tup
    if config.row_p is not None and random.random() > config.row_p:
        return
    output_path = os.path.join(
        config.output_folder, os.path.basename(metadata_row.csv_path)
    )
    if not any_input_is_newer(labels_paths + [metadata_row.csv_path], [output_path]):
        return

    # if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
    #     prev_csv_path = metadata_row.csv_path
    #     assert isinstance(prev_csv_path, str)
    #     if config.verbose:
    #         LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
    music_df = read_csv(get_csv_path(metadata_row.csv_path, config))

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

    music_df.to_csv(output_path)

    if config.verbose:
        print(f"Wrote {output_path}")

    if config.debug:
        return


def handle_labels(
    metadata_df: pd.DataFrame,
    config: Config,
    feature_name: str | None = None,
    indices: None | list[int] = None,
):
    labels_paths = list(
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

    if config.row_p is None:
        config.row_p = 1.0
    random.seed(42)

    with multiprocessing.Pool(config.num_workers) as pool:
        partial_handler = partial(
            process_row,
            config=config,
            labels_paths=labels_paths,
            feature_name=feature_name,
            labels_lists=labels_lists,
        )
        list(
            tqdm(
                pool.imap_unordered(
                    partial_handler,
                    metadata_df.iterrows(),
                    chunksize=config.multiprocess_chunk_size,
                ),
                total=len(metadata_df),
            )
        )
    # for row_i, metadata_row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
    #     if config.row_p is not None and random.random() > config.row_p:
    #         continue
    #     output_path = os.path.join(
    #         config.output_folder, os.path.basename(metadata_row.csv_path)
    #     )
    #     if not any_input_is_newer(
    #         labels_paths + [metadata_row.csv_path], [output_path]
    #     ):
    #         continue

    #     if prev_csv_path is None or metadata_row.csv_path != prev_csv_path:
    #         prev_csv_path = metadata_row.csv_path
    #         assert isinstance(prev_csv_path, str)
    #         if config.verbose:
    #             LOGGER.info(f"Reading {get_csv_path(prev_csv_path, config)}")
    #         music_df = read_csv(get_csv_path(prev_csv_path, config))

    #     assert music_df is not None

    #     df_indices = metadata_row.df_indices
    #     if isinstance(df_indices, str):
    #         df_indices = ast.literal_eval(df_indices)

    #     feature_name = feature_name if feature_name is not None else config.feature_name
    #     for label_i, labels_list in enumerate(labels_lists):
    #         labels_str = labels_list[row_i]
    #         labels = labels_str.split()

    #         if isinstance(feature_name, str):
    #             if len(labels_lists) > 1:
    #                 this_feature_name = f"{feature_name}_{label_i}"
    #             else:
    #                 this_feature_name = feature_name
    #         else:
    #             this_feature_name = feature_name[label_i]

    #         music_df = label_df(
    #             music_df,
    #             labels=labels,
    #             label_indices=df_indices,
    #             label_col_name=this_feature_name,
    #         )
    #     os.makedirs(config.output_folder, exist_ok=True)

    #     music_df.to_csv(output_path)

    #     if config.verbose:
    #         print(f"Wrote {output_path}")

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
