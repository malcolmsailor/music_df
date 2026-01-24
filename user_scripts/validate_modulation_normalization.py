import os
import pdb
import random
import sys
import traceback
from dataclasses import dataclass

import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

from music_df.chord_df import (
    add_chord_pcs,
    add_key_pcs,
    extract_chord_df_from_music_df,
    extract_key_df_from_music_df,
    merge_annotations,
    split_degrees_to_single_degree,
)
from music_df.harmony.chords import get_key_pc_cache, get_rn_pc_cache
from music_df.harmony.matching import percent_chord_df_match


def custom_excepthook(exc_type, exc_value, exc_traceback):
    if exc_type is not KeyboardInterrupt:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
        pdb.post_mortem(exc_traceback)


sys.excepthook = custom_excepthook


@dataclass
class Config:
    original_folder: str = (
        "/Users/malcolm/datasets/chord_tones/salami_slice_dedoubled_no_suspensions_q16"
    )
    normalized_folder: str = "/Users/malcolm/output/normalize_modulations"
    max_files: int | None = None
    seed: int = 42
    diff_output_path: str = "~/tmp/chord_diff_notes.csv"


def get_music_df_with_match(
    music_df: pd.DataFrame, rn_pc_cache, key_pc_cache
) -> pd.DataFrame:
    """Process music_df and add percent_chord_match column."""
    music_df = split_degrees_to_single_degree(music_df)
    chord_df = extract_chord_df_from_music_df(music_df)
    chord_df["rn"] = merge_annotations(chord_df, include_key=False)
    chord_df = add_chord_pcs(chord_df, rn_pc_cache=rn_pc_cache)

    chord_result = percent_chord_df_match(
        music_df,
        chord_df,
        is_sliced=True,
        match_col="percent_chord_match",
    )
    return chord_result["music_df"]


def find_differing_notes(
    original_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
    rn_pc_cache,
    key_pc_cache,
) -> pd.DataFrame:
    """Find notes where percent_chord_match differs between original and normalized."""
    original_with_match = get_music_df_with_match(
        original_df.copy(), rn_pc_cache, key_pc_cache
    )
    normalized_with_match = get_music_df_with_match(
        normalized_df.copy(), rn_pc_cache, key_pc_cache
    )
    original_with_match["rn"] = merge_annotations(
        original_with_match, include_key=False
    )
    normalized_with_match["rn"] = merge_annotations(
        normalized_with_match, include_key=False
    )

    original_with_match = original_with_match.loc[
        original_with_match.type == "note",
        ["key", "rn", "percent_chord_match", "onset", "pitch", "chord_pcs"],
    ]
    normalized_with_match = normalized_with_match.loc[
        normalized_with_match.type == "note",
        ["key", "rn", "percent_chord_match", "onset", "pitch", "chord_pcs"],
    ]

    original_with_match = original_with_match.rename(
        columns={
            "percent_chord_match": "original_percent_chord_match",
            "key": "original_key",
            "rn": "original_rn",
            "chord_pcs": "original_chord_pcs",
        }
    )
    normalized_with_match = normalized_with_match.rename(
        columns={
            "percent_chord_match": "normalized_percent_chord_match",
            "key": "normalized_key",
            "rn": "normalized_rn",
            "chord_pcs": "normalized_chord_pcs",
        }
    )

    merged = original_with_match.merge(
        normalized_with_match,
        on=["onset", "pitch"],
        how="outer",
        indicator=True,
    )

    diff_mask = (
        merged["original_percent_chord_match"]
        != merged["normalized_percent_chord_match"]
    )
    return merged[diff_mask]


def get_match_percentages(
    music_df: pd.DataFrame, rn_pc_cache, key_pc_cache
) -> dict[str, float]:
    music_df = split_degrees_to_single_degree(music_df)
    chord_df = extract_chord_df_from_music_df(music_df)
    chord_df["rn"] = merge_annotations(chord_df, include_key=False)
    chord_df = add_chord_pcs(chord_df, rn_pc_cache=rn_pc_cache)
    key_df = extract_key_df_from_music_df(music_df)
    key_df = add_key_pcs(key_df, key_pc_cache=key_pc_cache)

    chord_result = percent_chord_df_match(
        music_df,
        chord_df,
        is_sliced=True,
        match_col="percent_chord_match",
    )

    key_result = percent_chord_df_match(
        chord_result["music_df"],
        key_df,
        chord_df_pc_key="key_pcs",
        is_sliced=True,
        match_col="percent_key_match",
    )

    return {
        "chord_macroaverage": chord_result["macroaverage"],
        "key_macroaverage": key_result["macroaverage"],
    }


def main():
    conf = OmegaConf.from_cli(sys.argv[1:])
    config = Config(**conf)  # type:ignore

    csv_paths = []
    for root, _, files in os.walk(config.original_folder):
        for file in files:
            if file.endswith(".csv"):
                rel_path = os.path.relpath(
                    os.path.join(root, file), config.original_folder
                )
                csv_paths.append(rel_path)

    if config.max_files is not None:
        random.seed(config.seed)
        random.shuffle(csv_paths)
        csv_paths = csv_paths[: config.max_files]

    results = []
    all_diff_notes = []

    rn_pc_cache = get_rn_pc_cache(rn_format="rnbert", hex_str=True)
    key_pc_cache = get_key_pc_cache(hex_str=True)

    for rel_path in tqdm(csv_paths):
        original_path = os.path.join(config.original_folder, rel_path)
        normalized_path = os.path.join(config.normalized_folder, rel_path)

        if not os.path.exists(normalized_path):
            print(f"Warning: normalized file not found: {normalized_path}")
            continue

        original_df = pd.read_csv(original_path, index_col=0)
        normalized_df = pd.read_csv(normalized_path, index_col=0)

        original_matches = get_match_percentages(original_df, rn_pc_cache, key_pc_cache)
        normalized_matches = get_match_percentages(
            normalized_df, rn_pc_cache, key_pc_cache
        )

        original_chord = original_matches["chord_macroaverage"]
        normalized_chord = normalized_matches["chord_macroaverage"]

        results.append(
            {
                "file": rel_path,
                "original_chord": original_chord,
                "original_key": original_matches["key_macroaverage"],
                "normalized_chord": normalized_chord,
                "normalized_key": normalized_matches["key_macroaverage"],
            }
        )

        if original_chord != normalized_chord:
            diff_notes = find_differing_notes(
                original_df, normalized_df, rn_pc_cache, key_pc_cache
            )
            if not diff_notes.empty:
                diff_notes = diff_notes.copy()
                diff_notes["file"] = rel_path
                all_diff_notes.append(diff_notes)

    results_df = pd.DataFrame(results)
    print(results_df.to_string())
    print("\nMeans:")
    print(
        results_df[
            ["original_chord", "original_key", "normalized_chord", "normalized_key"]
        ].mean()
    )

    if all_diff_notes:
        diff_notes_df = pd.concat(all_diff_notes, ignore_index=True)
        output_path = os.path.expanduser(config.diff_output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        diff_notes_df.to_csv(output_path)
        print(f"\nSaved {len(diff_notes_df)} differing notes to {output_path}")


if __name__ == "__main__":
    main()
