import os
import sys
from dataclasses import dataclass

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from music_df.chord_df import (
    add_chord_pcs,
    add_key_pcs,
    extract_chord_df_from_music_df,
    merge_annotations,
    single_degree_to_split_degrees,
    split_degrees_to_single_degree,
)
from music_df.harmony.matching import label_pc_matches, percent_chord_df_match
from music_df.plot_piano_rolls.plot import (
    plot_piano_roll_and_continuous_feature,
    plot_piano_roll_and_feature,
)
from music_df.read import read
from music_df.slice_df import slice_df


@dataclass
class Config:
    input_folder: str = (
        "/Users/malcolm/datasets/chord_tones/salami_slice_dedoubled_no_suspensions_q16"
    )
    output_folder: str = "/Users/malcolm/output/plot_harmony_annotations"
    max_files: int | None = 5
    plot_width: float = 12
    plot_height: float = 6


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    os.makedirs(config.output_folder, exist_ok=True)

    paths = []
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
        for file in files:
            if config.max_files is not None and n_files >= config.max_files:
                break
            if not file.endswith(".csv"):
                continue
            else:
                input_path = os.path.join(root, file)
                output_basepath = os.path.join(new_root, file[:-4])
                paths.append((input_path, output_basepath))

    for input_path, output_basepath in tqdm(paths):
        music_df = read(input_path)

        # music_df["rn"] = merge_annotations(music_df)

        chord_df = extract_chord_df_from_music_df(music_df)
        chord_df = add_chord_pcs(chord_df)
        chord_df = add_key_pcs(chord_df)

        music_df = slice_df(music_df, chord_df.onset)

        chord_result = percent_chord_df_match(
            music_df, chord_df, is_sliced=True, match_col="percent_chord_match"
        )
        key_result = percent_chord_df_match(
            chord_result["music_df"],
            chord_df,
            chord_df_pc_key="key_pcs",
            is_sliced=True,
            match_col="percent_key_match",
        )

        chord_result_df = label_pc_matches(
            chord_result["music_df"], chord_df, is_sliced=True
        )
        key_result_df = label_pc_matches(
            key_result["music_df"],
            chord_df,
            chord_df_pc_key="key_pcs",
            is_sliced=True,
            match_col="is_key_match",
        )

        chord_note_df = chord_result_df[chord_result_df["type"] == "note"]
        key_note_df = key_result_df[key_result["music_df"]["type"] == "note"]

        fig, ax = plt.subplots(figsize=(config.plot_width, config.plot_height))
        plot_piano_roll_and_continuous_feature(
            chord_note_df,
            "percent_chord_match",
            show=True,
            title="Chord match %",
        )
        plt.savefig(f"{output_basepath}_chord_match_percent.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(config.plot_width, config.plot_height))
        plot_piano_roll_and_continuous_feature(
            key_note_df,
            "percent_key_match",
            show=True,
            title="Key match %",
        )
        plt.savefig(f"{output_basepath}_key_match_percent.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(config.plot_width, config.plot_height))
        plot_piano_roll_and_feature(
            chord_note_df,
            "is_chord_match",
            colormapping={True: "blue", False: "red"},
            show=True,
            label_notes=False,
            title="Chord match",
        )
        plt.savefig(f"{output_basepath}_chord_match.png")
        plt.close()

        fig, ax = plt.subplots(figsize=(config.plot_width, config.plot_height))
        plot_piano_roll_and_feature(
            key_note_df,
            "is_key_match",
            show=True,
            colormapping={True: "blue", False: "red"},
            label_notes=False,
            title="Key match",
        )
        plt.savefig(f"{output_basepath}_key_match.png")
        plt.close()


if __name__ == "__main__":
    main()
