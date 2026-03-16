"""Demo: apply merge_repeated_notes to a folder of music files.

Reads files, merges repeated notes, reports summary statistics,
and saves sampled before/after passage excerpts as CSVs.
"""

from __future__ import annotations

import argparse
from functools import partial

import pandas as pd

from music_df.detremolo import merge_repeated_notes

from _demo_helpers import add_common_args, run_demo


def _adapt_for_demo(
    df: pd.DataFrame,
    max_note_duration: float | None,
    max_gap: float,
) -> pd.DataFrame:
    """Wrap merge_repeated_notes to produce the interface expected by
    _demo_helpers (original_index column and note-count attrs)."""
    n_undedoubled_notes = int((df["type"] == "note").sum())
    n_non_notes = int((df["type"] != "note").sum())

    result = merge_repeated_notes(
        df, max_note_duration=max_note_duration, max_gap=max_gap
    )

    if "original_index" not in result.columns:
        result = result.reset_index(names="original_index")
    else:
        result = result.reset_index(drop=True)

    n_dedoubled_notes = int((result["type"] == "note").sum())
    assert (result["type"] != "note").sum() == n_non_notes, (
        "Non-note row count changed during merging"
    )
    assert n_dedoubled_notes <= n_undedoubled_notes, (
        "Merged notes exceed original notes"
    )

    result.attrs["n_undedoubled_notes"] = n_undedoubled_notes
    result.attrs["n_dedoubled_notes"] = n_dedoubled_notes

    # Compute involved_indices: dropped notes + notes whose release was extended
    dropped = set(df.index) - set(result["original_index"])
    modified: set[int] = set()
    for _, row in result.iterrows():
        orig_idx = row["original_index"]
        if orig_idx in df.index and row["type"] == "note":
            if row["release"] != df.loc[orig_idx, "release"]:
                modified.add(orig_idx)
    result.attrs["involved_indices"] = dropped | modified

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: merge repeated notes and save before/after excerpts."
    )
    add_common_args(parser)
    parser.add_argument(
        "--max-note-duration",
        type=float,
        default=None,
        help="Max note duration eligible for merging (default: no limit)",
    )
    parser.add_argument(
        "--max-gap",
        type=float,
        default=0.125,
        help="Max gap between note release and next onset (default: 0.125)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    transform = partial(
        _adapt_for_demo,
        max_note_duration=args.max_note_duration,
        max_gap=args.max_gap,
    )
    meta = {
        "input_folder": str(args.input_folder),
        "max_files": args.max_files,
        "samples": args.samples,
        "bars": args.bars,
        "quarter_notes": args.quarter_notes,
        "min_length": args.min_length,
        "max_note_duration": args.max_note_duration,
        "max_gap": args.max_gap,
        "seed": args.seed,
    }
    run_demo(args, transform, meta)


if __name__ == "__main__":
    main()
