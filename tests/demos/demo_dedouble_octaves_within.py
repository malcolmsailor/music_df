"""Demo: apply within-instrument octave dedoubling to a folder of music files.

Reads files, applies dedouble_octaves_within_instrument(), reports summary
statistics, and saves sampled before/after passage excerpts as CSVs.
"""

from __future__ import annotations

import argparse
from functools import partial

import pandas as pd

from music_df.dedouble_instruments import (
    DEFAULT_PITCH_THRESHOLD,
    dedouble_octaves_within_instrument,
)

from _demo_helpers import TICKS_PER_QUARTER, add_common_args, run_demo


def _find_within_octave_partner_indices(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    dropped_indices: set[int],
) -> set[int]:
    """Find kept notes that are octave partners of dropped notes.

    For each dropped note, finds the kept note at the same onset with
    a pitch difference in {12, 24, 36}.
    """
    tpq = TICKS_PER_QUARTER
    dropped_notes = original_df.loc[sorted(dropped_indices)]
    kept_notes = result_df[result_df["type"] == "note"]
    partners: set[int] = set()
    octave_intervals = {12, 24, 36}
    for _, dn in dropped_notes.iterrows():
        match_mask = (
            (kept_notes["onset"] * tpq).round() == round(dn["onset"] * tpq)
        ) & (kept_notes["pitch"] % 12 == dn["pitch"] % 12)
        for orig_idx in kept_notes.loc[match_mask, "original_index"]:
            kept_pitch = kept_notes.loc[
                kept_notes["original_index"] == orig_idx, "pitch"
            ].iloc[0]
            if abs(kept_pitch - dn["pitch"]) in octave_intervals:
                partners.add(orig_idx)
    return partners


def _transform(df, min_length, pitch_threshold, match_releases, max_gap_onsets,
               max_streak_pitch_distance):
    result = dedouble_octaves_within_instrument(
        df,
        min_length=min_length,
        pitch_threshold=pitch_threshold,
        match_releases=match_releases,
        max_gap_onsets=max_gap_onsets,
        max_streak_pitch_distance=max_streak_pitch_distance,
    )
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_within_octave_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: dedouble within-instrument octaves and save before/after excerpts."
    )
    add_common_args(parser)
    parser.add_argument(
        "--pitch-threshold", type=float, default=DEFAULT_PITCH_THRESHOLD,
        help=f"MIDI pitch threshold for melody/bass register (default: {DEFAULT_PITCH_THRESHOLD})",
    )
    parser.add_argument(
        "--no-match-releases", action="store_true",
        help="Don't require matching releases for octave pairs",
    )
    parser.add_argument(
        "--max-gap-onsets", type=int, default=0,
        help="Allow up to N consecutive onsets without the pair before breaking the streak (default: 0)",
    )
    parser.add_argument(
        "--max-streak-pitch-distance", type=int, default=12,
        help="Max lower-pitch distance between consecutive pairs in a streak (default: 12)",
    )
    parser.set_defaults(min_length=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    transform = partial(
        _transform,
        min_length=args.min_length,
        pitch_threshold=args.pitch_threshold,
        match_releases=not args.no_match_releases,
        max_gap_onsets=args.max_gap_onsets,
        max_streak_pitch_distance=args.max_streak_pitch_distance,
    )
    meta = {
        "input_folder": str(args.input_folder),
        "max_files": args.max_files,
        "samples": args.samples,
        "bars": args.bars,
        "quarter_notes": args.quarter_notes,
        "min_length": args.min_length,
        "pitch_threshold": args.pitch_threshold,
        "match_releases": not args.no_match_releases,
        "max_gap_onsets": args.max_gap_onsets,
        "max_streak_pitch_distance": args.max_streak_pitch_distance,
        "seed": args.seed,
    }
    run_demo(args, transform, meta)


if __name__ == "__main__":
    main()
