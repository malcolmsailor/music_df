"""Demo: apply octave dedoubling to a folder of music files.

Reads files, applies dedouble_octaves(), reports summary statistics,
and saves sampled before/after passage excerpts as CSVs.
"""

from __future__ import annotations

import argparse
from functools import partial

import pandas as pd
from _demo_helpers import TICKS_PER_QUARTER, add_common_args, run_demo

from music_df.dedouble_instruments import DEFAULT_PITCH_THRESHOLD, dedouble_octaves


def _find_octave_partner_indices(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    dropped_indices: set[int],
) -> set[int]:
    """Find kept notes that are octave partners of dropped notes.

    Matches by quantized (onset, release, pitch % 12) instead of exact pitch.
    """
    tpq = TICKS_PER_QUARTER
    dropped_notes = original_df.loc[sorted(dropped_indices)]
    kept_notes = result_df[result_df["type"] == "note"]
    partners: set[int] = set()
    for _, dn in dropped_notes.iterrows():
        match_mask = (
            ((kept_notes["onset"] * tpq).round() == round(dn["onset"] * tpq))
            & ((kept_notes["release"] * tpq).round() == round(dn["release"] * tpq))
            & (kept_notes["pitch"] % 12 == dn["pitch"] % 12)
        )
        partners.update(kept_notes.loc[match_mask, "original_index"])
    return partners


def _transform(df, min_length, pitch_threshold):
    result = dedouble_octaves(
        df, min_length=min_length, pitch_threshold=pitch_threshold
    )
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_octave_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: dedouble octaves and save before/after excerpts."
    )
    add_common_args(parser)
    parser.add_argument(
        "--pitch-threshold",
        type=float,
        default=DEFAULT_PITCH_THRESHOLD,
        help=f"MIDI pitch threshold for melody/bass register (default: {DEFAULT_PITCH_THRESHOLD})",
    )
    # Override min-length default to 3 (octave matching needs more notes
    # to avoid false positives from contrary motion)
    parser.set_defaults(min_length=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    transform = partial(
        _transform, min_length=args.min_length, pitch_threshold=args.pitch_threshold
    )
    meta = {
        "input_folder": str(args.input_folder),
        "max_files": args.max_files,
        "samples": args.samples,
        "bars": args.bars,
        "quarter_notes": args.quarter_notes,
        "min_length": args.min_length,
        "pitch_threshold": args.pitch_threshold,
        "seed": args.seed,
    }
    run_demo(args, transform, meta)


if __name__ == "__main__":
    main()
