import glob
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from music_df.chord_df import (
    extract_chord_df_from_music_df,
    split_degrees_to_single_degree,
)
from music_df.harmony.modulation import modulation_census, tonicization_census
from music_df.read_csv import read_csv


@dataclass
class Config:
    input_folder: str
    output_folder: str = os.path.join(os.path.expanduser("~"), "output")
    max_files: int | None = None


def get_primary_with_secondary(chord_df: pd.DataFrame):
    has_secondary = chord_df["degree"].str.contains("/")
    if not has_secondary.any():
        return []
    subset_df = chord_df.loc[has_secondary]
    return subset_df["degree"].str.split("/", expand=True)[0].to_list()


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    input_files = glob.glob(f"{config.input_folder}/**/*.csv", recursive=True)

    input_files = input_files[: config.max_files]

    mod_census_accumulator = []
    ton_census_accumulator = []

    primary_with_secondary_counter = Counter()

    for input_file in tqdm(input_files):
        music_df = read_csv(input_file)
        assert music_df is not None, f"Failed to read {input_file}"
        music_df = split_degrees_to_single_degree(music_df)
        chord_df = extract_chord_df_from_music_df(music_df)
        primary_with_secondary_counter.update(get_primary_with_secondary(chord_df))
        this_mod_census = modulation_census(chord_df)
        this_ton_census = tonicization_census(chord_df)
        mod_census_accumulator.append(
            this_mod_census.drop(columns=["chord_df_index", "onset"])
        )
        ton_census_accumulator.append(
            this_ton_census.drop(columns=["chord_df_index", "onset"])
        )

    print(primary_with_secondary_counter)

    mod_census_df = pd.concat(mod_census_accumulator, ignore_index=True)
    ton_census_df = pd.concat(ton_census_accumulator, ignore_index=True)

    output_folder = os.path.join(
        config.output_folder, config.input_folder.replace(os.path.sep, "+").strip("+")
    )
    os.makedirs(output_folder, exist_ok=True)
    mod_census_df.to_csv(f"{output_folder}/modulation_census.csv", index=False)
    ton_census_df.to_csv(f"{output_folder}/tonicization_census.csv", index=False)

    with open(f"{output_folder}/secondary_primary_degree_counts.json", "w") as outf:
        json.dump(primary_with_secondary_counter, outf, indent=2)

    print(f"Saved modulation census to {output_folder}/modulation_census.csv")
    print(f"Saved tonicization census to {output_folder}/tonicization_census.csv")
    print(
        f"Saved secondary chord counts to {output_folder}/secondary_primary_degree_counts.json"
    )


if __name__ == "__main__":
    main()
