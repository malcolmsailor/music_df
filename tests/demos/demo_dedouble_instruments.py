"""Demo: apply instrument dedoubling to a folder of music files.

Reads files, applies dedouble_instruments(), reports summary statistics,
and saves sampled before/after passage excerpts as CSVs.
"""

from __future__ import annotations

import argparse

import pandas as pd

from music_df.dedouble_instruments import dedouble_instruments

from _demo_helpers import TICKS_PER_QUARTER, add_common_args, run_demo


def _find_partner_indices(
    original_df: pd.DataFrame,
    result_df: pd.DataFrame,
    dropped_indices: set[int],
) -> set[int]:
    """Find original indices of kept notes that are partners of dropped notes.

    A "partner" is a kept note with the same quantized (onset, release, pitch)
    as a dropped note.
    """
    tpq = TICKS_PER_QUARTER
    dropped_notes = original_df.loc[sorted(dropped_indices)]
    kept_notes = result_df[result_df["type"] == "note"]
    partners: set[int] = set()
    for _, dn in dropped_notes.iterrows():
        match_mask = (
            (kept_notes["onset"] * tpq).round() == round(dn["onset"] * tpq)
        ) & (
            (kept_notes["release"] * tpq).round() == round(dn["release"] * tpq)
        ) & (kept_notes["pitch"] == dn["pitch"])
        partners.update(kept_notes.loc[match_mask, "original_index"])
    return partners


def _transform(df, min_length):
    result = dedouble_instruments(df, min_length=min_length)
    dropped = set(df.index) - set(result["original_index"])
    partners = _find_partner_indices(df, result, dropped)
    result.attrs["involved_indices"] = dropped | partners
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Demo: dedouble instruments and save before/after excerpts."
    )
    add_common_args(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    transform = lambda df: _transform(df, args.min_length)  # noqa: E731
    meta = {
        "input_folder": str(args.input_folder),
        "max_files": args.max_files,
        "samples": args.samples,
        "bars": args.bars,
        "quarter_notes": args.quarter_notes,
        "min_length": args.min_length,
        "seed": args.seed,
    }
    run_demo(args, transform, meta)


if __name__ == "__main__":
    main()
