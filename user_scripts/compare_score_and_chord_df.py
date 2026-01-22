import pdb
import sys
import traceback
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf

from music_df.chord_df import add_chord_pcs, add_key_pcs, extract_key_df_from_music_df
from music_df.harmony.matching import label_pc_matches, percent_chord_df_match
from music_df.plot_piano_rolls.plot import (
    plot_piano_roll_and_continuous_feature,
    plot_piano_roll_and_feature,
)
from music_df.read import read
from music_df.slice_df import slice_df


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    midi_path: str = "/Users/malcolm/datasets/ClassicalMusicArchivesClean/Zipoli,_Domenico-Gavotta_in_B-.mid"
    chord_df_path: str = "/Volumes/Taneyev/musicbert/chord_tables_0.3/CMA-dedoubled/chained_rn/45951812_cond_on_39958320/Zipoli,_Domenico-Gavotta_in_B-.csv"


CHORD_COLS_TO_KEEP = ["onset", "release", "rn", "key"]


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore
    score_df = read(config.midi_path)
    chord_df = pd.read_csv(config.chord_df_path)[CHORD_COLS_TO_KEEP]
    chord_df = add_chord_pcs(chord_df)

    key_df = extract_key_df_from_music_df(chord_df)
    key_df = add_key_pcs(key_df)

    score_df = slice_df(score_df, chord_df.onset)

    chord_result = percent_chord_df_match(
        score_df, chord_df, is_sliced=True, match_col="percent_chord_match"
    )
    key_result = percent_chord_df_match(
        chord_result["music_df"],
        key_df,
        chord_df_pc_key="key_pcs",
        is_sliced=True,
        match_col="percent_key_match",
    )

    chord_result_df = label_pc_matches(
        chord_result["music_df"], chord_df, is_sliced=True
    )
    chord_note_df = chord_result_df[chord_result_df["type"] == "note"]
    key_result_df = label_pc_matches(
        key_result["music_df"],
        key_df,
        chord_df_pc_key="key_pcs",
        is_sliced=True,
        match_col="is_key_match",
    )
    key_note_df = key_result_df[key_result["music_df"]["type"] == "note"]

    plot_piano_roll_and_continuous_feature(
        chord_note_df,
        "percent_chord_match",
        show=True,
        title="Chord match %",
    )
    plot_piano_roll_and_continuous_feature(
        key_note_df,
        "percent_key_match",
        show=True,
        title="Key match %",
    )
    plot_piano_roll_and_feature(
        chord_note_df,
        "is_chord_match",
        colormapping={True: "blue", False: "red"},
        show=True,
        label_notes=False,
        title="Chord match",
    )
    plot_piano_roll_and_feature(
        key_note_df,
        "is_key_match",
        show=True,
        colormapping={True: "blue", False: "red"},
        label_notes=False,
        title="Key match",
    )
    print(f"Chord pc macroaverage: {chord_result['macroaverage']}")
    print(f"Chord pc microaverage: {chord_result['microaverage']}")
    print(f"Key pc macroaverage: {key_result['macroaverage']}")
    print(f"Key pc microaverage: {key_result['microaverage']}")


if __name__ == "__main__":
    main()
