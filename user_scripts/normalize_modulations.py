import os
import pdb
import shutil
import sys
import traceback
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from music_df.chord_df import (
    drop_harmony_columns,
    extract_chord_df_from_music_df,
    label_music_df_with_chord_df,
    single_degree_to_split_degrees,
    split_degrees_to_single_degree,
)
from music_df.harmony.modulation import (
    remove_long_tonicizations,
    remove_short_modulations,
)


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class RemoveLongTonicizationsConfig:
    max_tonicization_duration: float | None = 16.0
    min_removal_duration: float | None = 4.0
    max_tonicization_num_chords: int | None = 2
    min_removal_num_chords: int | None = 2


@dataclass
class RemoveShortModulationsConfig:
    min_modulation_duration: float | None = 2.0
    max_removal_duration: float | None = 16.0
    min_modulation_num_chords: int | None = 3
    max_removal_num_chords: int | None = 8


@dataclass
class Config(RemoveLongTonicizationsConfig, RemoveShortModulationsConfig):
    input_folder: str = (
        "/Users/malcolm/datasets/chord_tones/salami_slice_dedoubled_no_suspensions_q16"
    )
    output_folder: str = os.path.expanduser("~/output/normalize_modulations")
    max_files: int | None = None


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    os.makedirs(config.output_folder, exist_ok=True)

    n_files = 0
    for root, dirnames, files in os.walk(config.input_folder):
        new_root = os.path.join(
            config.output_folder, os.path.relpath(root, config.input_folder)
        )
        for dirname in dirnames:
            os.makedirs(
                os.path.join(new_root, dirname),
                exist_ok=True,
            )

        for file in tqdm(files):
            if config.max_files is not None and n_files >= config.max_files:
                break
            if not file.endswith(".csv"):
                shutil.copy(os.path.join(root, file), os.path.join(new_root, file))
            else:
                input_path = os.path.join(root, file)
                output_path = os.path.join(new_root, file)
                music_df = pd.read_csv(input_path, index_col=0)
                music_df = split_degrees_to_single_degree(music_df)
                chord_df = extract_chord_df_from_music_df(music_df)

                chord_df = remove_long_tonicizations(
                    chord_df,
                    max_tonicization_duration=config.max_tonicization_duration,
                    min_removal_duration=config.min_removal_duration,
                    max_tonicization_num_chords=config.max_tonicization_num_chords,
                    min_removal_num_chords=config.min_removal_num_chords,
                )
                chord_df = remove_short_modulations(
                    chord_df,
                    min_modulation_duration=config.min_modulation_duration,
                    max_removal_duration=config.max_removal_duration,
                    min_modulation_num_chords=config.min_modulation_num_chords,
                    max_removal_num_chords=config.max_removal_num_chords,
                )

                updated_music_df = drop_harmony_columns(music_df)
                updated_music_df = label_music_df_with_chord_df(
                    updated_music_df, chord_df
                )
                updated_music_df = single_degree_to_split_degrees(updated_music_df)
                updated_music_df.to_csv(output_path)

                n_files += 1
    print(
        f"Processed {n_files} files from {config.input_folder} to {config.output_folder}"
    )


if __name__ == "__main__":
    main()
