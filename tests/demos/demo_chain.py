"""Demo: apply a chain of transforms to a folder of music files.

Reads files, applies transforms in the user-specified order (all with
default parameters), reports summary statistics, and saves sampled
before/after passage excerpts as CSVs.

Usage:
    python demo_chain.py input/ output/ --transforms merge_repeated_notes,dedouble_octaves
"""

from __future__ import annotations

import argparse
from typing import Callable

import pandas as pd

from music_df.dedouble_instruments import (
    dedouble_instruments,
    dedouble_octaves,
    dedouble_octaves_within_instrument,
)
from music_df.detremolo import merge_repeated_notes
from music_df.add_feature import infer_barlines
from music_df.split_notes import split_notes_at_barlines

from _demo_helpers import add_common_args, run_demo
from demo_dedouble_instruments import _find_partner_indices
from demo_dedouble_octaves import _find_octave_partner_indices
from demo_dedouble_octaves_within import _find_within_octave_partner_indices


def _wrap_dedouble_instruments(df: pd.DataFrame) -> pd.DataFrame:
    result = dedouble_instruments(df)
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def _wrap_dedouble_octaves(df: pd.DataFrame) -> pd.DataFrame:
    result = dedouble_octaves(df)
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_octave_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def _wrap_dedouble_octaves_within(df: pd.DataFrame) -> pd.DataFrame:
    result = dedouble_octaves_within_instrument(df)
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_within_octave_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def _wrap_merge_repeated_notes(df: pd.DataFrame) -> pd.DataFrame:
    n_undedoubled_notes = int((df["type"] == "note").sum())

    result = merge_repeated_notes(df)

    if "original_index" not in result.columns:
        result = result.reset_index(names="original_index")
    else:
        result = result.reset_index(drop=True)

    n_dedoubled_notes = int((result["type"] == "note").sum())
    result.attrs["n_undedoubled_notes"] = n_undedoubled_notes
    result.attrs["n_dedoubled_notes"] = n_dedoubled_notes

    dropped = set(df.index) - set(result["original_index"])
    modified: set[int] = set()
    for _, row in result.iterrows():
        orig_idx = row["original_index"]
        if orig_idx in df.index and row["type"] == "note":
            if row["release"] != df.loc[orig_idx, "release"]:
                modified.add(orig_idx)
    result.attrs["involved_indices"] = dropped | modified
    return result


def _wrap_split_notes_at_barlines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Tag each row with its original index before splitting
    df["original_index"] = df.index

    result = split_notes_at_barlines(df)
    # original_index propagates through the split; child rows inherit
    # the parent's original_index
    n_undedoubled_notes = int((df["type"] == "note").sum())
    n_dedoubled_notes = int((result["type"] == "note").sum())
    result = result.reset_index(drop=True)
    result.attrs["n_undedoubled_notes"] = n_undedoubled_notes
    result.attrs["n_dedoubled_notes"] = n_dedoubled_notes
    result.attrs["involved_indices"] = set()
    return result


TRANSFORMS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "merge_repeated_notes": _wrap_merge_repeated_notes,
    "split_notes_at_barlines": _wrap_split_notes_at_barlines,
    "dedouble_octaves": _wrap_dedouble_octaves,
    "dedouble_octaves_within": _wrap_dedouble_octaves_within,
    "dedouble_instruments": _wrap_dedouble_instruments,
}


def _make_chain(
    steps: list[Callable[[pd.DataFrame], pd.DataFrame]],
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Return a transform that applies steps in sequence, tracking original indices."""

    def chained(df: pd.DataFrame) -> pd.DataFrame:
        n_undedoubled_notes = int((df["type"] == "note").sum())

        if "bar" not in df["type"].values:
            # Tag rows with their original index before infer_barlines
            # adds bar rows and resets the index
            df = df.copy()
            df["_chain_orig_idx"] = df.index
            df = infer_barlines(df)

        all_involved: set[int] = set()

        # Build initial mapping from post-barline positions to true
        # original indices. Bar rows added by infer_barlines have no
        # original and are excluded.
        current_to_original: dict[int, int] = {}
        if "_chain_orig_idx" in df.columns:
            for pos in range(len(df)):
                val = df.iloc[pos]["_chain_orig_idx"]
                if pd.notna(val):
                    current_to_original[pos] = int(val)
            df = df.drop(columns=["_chain_orig_idx"])
        else:
            current_to_original = dict(enumerate(df.index))

        current = df.copy()
        for step in steps:
            # Strip any leftover original_index from previous step
            if "original_index" in current.columns:
                current = current.drop(columns=["original_index"])
            current = current.reset_index(drop=True)

            result = step(current)

            # Remap original_index through saved mapping
            result["original_index"] = result["original_index"].map(
                current_to_original
            )

            # Remap involved_indices through saved mapping
            step_involved = result.attrs.get("involved_indices", set())
            remapped_involved = {
                current_to_original[i]
                for i in step_involved
                if i in current_to_original
            }
            all_involved |= remapped_involved

            # Build new mapping for the next step (skip bar rows
            # that have no original_index)
            current_to_original = {}
            for new_pos in result.index:
                val = result.at[new_pos, "original_index"]
                if pd.notna(val):
                    current_to_original[new_pos] = int(val)

            current = result

        n_dedoubled_notes = int((current["type"] == "note").sum())
        current.attrs["n_undedoubled_notes"] = n_undedoubled_notes
        current.attrs["n_dedoubled_notes"] = n_dedoubled_notes
        current.attrs["involved_indices"] = all_involved
        return current

    return chained


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: apply chained transforms and save before/after excerpts."
    )
    add_common_args(parser)
    parser.add_argument(
        "--transforms",
        type=str,
        required=True,
        help=(
            "Comma-separated transform names in order. "
            f"Available: {', '.join(TRANSFORMS)}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    names = [t.strip() for t in args.transforms.split(",")]
    for name in names:
        if name not in TRANSFORMS:
            raise SystemExit(
                f"Unknown transform: {name!r}. "
                f"Available: {', '.join(TRANSFORMS)}"
            )

    steps = [TRANSFORMS[name] for name in names]
    transform = _make_chain(steps)

    meta = {
        "input_folder": str(args.input_folder),
        "max_files": args.max_files,
        "samples": args.samples,
        "bars": args.bars,
        "quarter_notes": args.quarter_notes,
        "transforms": names,
        "seed": args.seed,
    }
    run_demo(args, transform, meta)


if __name__ == "__main__":
    main()
