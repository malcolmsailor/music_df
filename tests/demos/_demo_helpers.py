"""Shared infrastructure for demo scripts.

Provides file discovery, processing, sampling, and CLI argument helpers
so each demo script only defines its transform and script-specific args.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Callable

import pandas as pd
from tqdm import tqdm

from music_df.add_feature import infer_barlines
from music_df.crop_df import crop_df, get_context_bars
from music_df.read import read
from music_df.slice_df import slice_df
from music_df.sort_df import sort_df

SUPPORTED_EXTENSIONS = {".csv", ".mid", ".midi", ".xml", ".mxl", ".mscx", ".mscz", ".krn"}
# Match the default grid used by dedouble_instruments()
TICKS_PER_QUARTER = 16


def quantize_df(df: pd.DataFrame, tpq: int = TICKS_PER_QUARTER) -> pd.DataFrame:
    """Round note onsets/releases to the same grid as dedouble_instruments()."""
    df = df.copy()
    note_mask = df["type"] == "note"
    df.loc[note_mask, "onset"] = (df.loc[note_mask, "onset"] * tpq).round() / tpq
    df.loc[note_mask, "release"] = (df.loc[note_mask, "release"] * tpq).round() / tpq
    return df


@dataclass
class FileResult:
    path: Path
    notes_before: int
    notes_after: int
    dropped_indices: set[int]
    involved_indices: set[int]
    original_df: pd.DataFrame
    dedoubled_df: pd.DataFrame


@dataclass
class Candidate:
    filename: str
    bar_onset: float
    n_removed: int
    file_result: FileResult


def read_file(path: Path) -> pd.DataFrame | None:
    try:
        return sort_df(read(str(path)))
    except Exception as exc:
        print(f"  WARNING: failed to read {path.name}: {exc}", file=sys.stderr)
        return None


def discover_files(input_folder: Path, max_files: int | None, seed: int) -> list[Path]:
    paths = [
        p for p in input_folder.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_files is not None:
        paths = paths[:max_files]
    return paths


def process_files(
    paths: list[Path],
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> list[FileResult]:
    """Process files, applying transform_fn to each.

    transform_fn receives a sorted df and must return a dedoubled df with
    ``original_index`` column and ``n_undedoubled_notes`` / ``n_dedoubled_notes``
    attrs.
    """
    results: list[FileResult] = []
    for path in tqdm(paths, desc="Processing files"):
        df = read_file(path)
        if df is None:
            continue
        dedoubled = transform_fn(df)
        original_indices = set(df.index)
        kept_indices = set(dedoubled["original_index"])
        dropped = original_indices - kept_indices
        involved = dedoubled.attrs.get("involved_indices", dropped)
        results.append(
            FileResult(
                path=path,
                notes_before=int((df["type"] == "note").sum()),
                notes_after=dedoubled.attrs["n_dedoubled_notes"],
                dropped_indices=dropped,
                involved_indices=involved,
                original_df=df,
                dedoubled_df=dedoubled,
            )
        )
    return results


def print_summary(results: list[FileResult]) -> None:
    files_with_removals = [r for r in results if r.notes_before > r.notes_after]
    total_before = sum(r.notes_before for r in results)
    total_after = sum(r.notes_after for r in results)
    total_removed = total_before - total_after

    print("\n=== Summary ===")
    print(f"Files processed:            {len(results)}")
    print(f"Files with doublings found: {len(files_with_removals)}")
    print(f"Total notes before:         {total_before}")
    print(f"Total notes after:          {total_after}")
    print(f"Total notes removed:        {total_removed}")

    if files_with_removals:
        removals = [r.notes_before - r.notes_after for r in files_with_removals]
        print(f"Mean notes removed/file:    {mean(removals):.1f}")
        print(f"Median notes removed/file:  {median(removals):.1f}")


def collect_candidates(
    results: list[FileResult], passage_bars: int | None, passage_qn: float | None
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for result in results:
        if result.notes_before == result.notes_after:
            continue
        df = result.original_df
        if "bar" not in df["type"].values:
            # Preserve original indices as a column so that
            # dropped_indices/involved_indices (computed against the
            # pre-barline index) can still be matched after
            # infer_barlines resets the DataFrame index.
            df["_src_idx"] = df.index
            df = infer_barlines(df)
            result.original_df = df
        bar_rows = df[df["type"] == "bar"]
        if bar_rows.empty:
            continue
        bar_onsets = sorted(bar_rows["onset"].unique())
        dropped = result.dropped_indices

        for bar_onset in bar_onsets:
            if passage_bars is not None:
                bar_idx = bar_onsets.index(bar_onset)
                if bar_idx + passage_bars > len(bar_onsets):
                    continue
                end_onset = (
                    bar_onsets[bar_idx + passage_bars]
                    if bar_idx + passage_bars < len(bar_onsets)
                    else df["onset"].max() + 1
                )
            else:
                assert passage_qn is not None
                end_onset = bar_onset + passage_qn

            passage_mask = (
                (df["type"] == "note")
                & (df["onset"] >= bar_onset)
                & (df["onset"] < end_onset)
            )
            passage_notes = df[passage_mask]
            if "_src_idx" in passage_notes.columns:
                passage_indices = set(
                    passage_notes["_src_idx"].dropna().astype(int)
                )
            else:
                passage_indices = set(passage_notes.index)
            removed_in_passage = passage_indices & dropped
            if removed_in_passage:
                candidates.append(
                    Candidate(
                        filename=result.path.name,
                        bar_onset=bar_onset,
                        n_removed=len(removed_in_passage),
                        file_result=result,
                    )
                )
    return candidates


def crop_passage(
    df: pd.DataFrame,
    bar_onset: float,
    passage_bars: int | None,
    passage_qn: float | None,
) -> pd.DataFrame:
    if passage_bars is not None:
        return get_context_bars(df, bar_onset, n_before=0, n_after=passage_bars - 1)
    assert passage_qn is not None
    end_onset = bar_onset + passage_qn
    cropped = crop_df(df, start_time=bar_onset)
    note_mask = cropped["type"] == "note"
    return cropped[~note_mask | (cropped["onset"] < end_onset)]


def _slice_to_excerpt(
    df: pd.DataFrame, start_time: float, end_time: float
) -> pd.DataFrame:
    """Slice notes at excerpt boundaries and return notes within range.

    Notes crossing start_time or end_time are split so the within-excerpt
    portion is preserved rather than being lost to onset-based filtering.
    """
    sliced = slice_df(df, [start_time, end_time])
    note_mask = sliced["type"] == "note"
    in_range = (sliced["onset"] >= start_time) & (sliced["onset"] < end_time)
    result = sliced[note_mask & in_range]
    return result.drop(columns=["sliced"])


def save_samples(
    candidates: list[Candidate],
    output_folder: Path,
    n_samples: int,
    seed: int,
    passage_bars: int | None,
    passage_qn: float | None,
    meta: dict,
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sampled = rng.sample(candidates, min(n_samples, len(candidates)))

    lookup_rows = []
    for i, cand in enumerate(sampled):
        excerpt_id = f"{i:03d}"

        start_time = cand.bar_onset
        orig_df = cand.file_result.original_df
        bar_onsets = sorted(
            orig_df.loc[orig_df["type"] == "bar", "onset"].unique()
        )
        if passage_bars is not None:
            bar_idx = bar_onsets.index(start_time)
            if bar_idx + passage_bars < len(bar_onsets):
                end_time = bar_onsets[bar_idx + passage_bars]
            else:
                end_time = orig_df.loc[
                    orig_df["type"] == "note", "release"
                ].max()
        else:
            assert passage_qn is not None
            end_time = start_time + passage_qn

        # Get metadata rows (time_signature, bar) via crop_passage
        before_full = crop_passage(
            orig_df, cand.bar_onset, passage_bars, passage_qn
        )
        before_non_notes = before_full[before_full["type"] != "note"]

        # Slice notes at excerpt boundaries so notes crossing edges
        # keep their within-excerpt portion
        before_notes = _slice_to_excerpt(orig_df, start_time, end_time)
        before = sort_df(
            pd.concat([before_non_notes, before_notes], ignore_index=True)
        )

        after_notes = _slice_to_excerpt(
            cand.file_result.dedoubled_df, start_time, end_time
        )
        after_crop = sort_df(
            pd.concat([before_non_notes, after_notes], ignore_index=True)
        )

        before = quantize_df(before)
        involved = cand.file_result.involved_indices
        if "_src_idx" in before.columns:
            before["target"] = before["_src_idx"].isin(involved)
        else:
            before["target"] = before.index.isin(involved)
        before.drop(columns=["_src_idx"], errors="ignore").to_csv(
            output_folder / f"{excerpt_id}_before.csv", index=False
        )
        after_crop = quantize_df(after_crop)
        if "original_index" in after_crop.columns:
            after_crop["target"] = after_crop["original_index"].isin(involved)
        else:
            after_crop["target"] = False
        after_crop.drop(columns=["_src_idx"], errors="ignore").to_csv(
            output_folder / f"{excerpt_id}_after.csv", index=False
        )

        lookup_rows.append(
            {
                "id": excerpt_id,
                "source_file": cand.filename,
                "bar_onset": cand.bar_onset,
                "notes_removed": cand.n_removed,
            }
        )

    lookup_df = pd.DataFrame(lookup_rows)
    lookup_df.to_csv(output_folder / "lookup.csv", index=False)

    meta["n_candidates"] = len(candidates)
    meta["n_sampled"] = len(sampled)
    (output_folder / "sample_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved {len(sampled)} excerpt pairs to {output_folder}/")


def add_common_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add CLI args shared by all demo scripts."""
    parser.add_argument("input_folder", type=Path, help="Folder of music files")
    parser.add_argument("output_folder", type=Path, help="Where to write output CSVs")
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N files")
    parser.add_argument("--samples", type=int, default=10, help="Sample up to M passages")

    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument("--bars", type=int, default=None, help="Passage length in bars")
    length_group.add_argument(
        "--quarter-notes", type=float, default=None, help="Passage length in quarter notes"
    )

    parser.add_argument("--min-length", type=int, default=2, help="Min doubling run length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser


def run_demo(
    args: argparse.Namespace,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    meta: dict,
) -> None:
    """Shared main orchestration: discover -> process -> summarize -> collect -> sample."""
    paths = discover_files(args.input_folder, args.max_files, args.seed)
    if not paths:
        print(f"No supported files found in {args.input_folder}")
        sys.exit(1)
    print(f"Found {len(paths)} files to process")

    results = process_files(paths, transform_fn)
    print_summary(results)

    # Default to 4 bars if neither specified
    passage_bars = args.bars
    passage_qn = args.quarter_notes
    if passage_bars is None and passage_qn is None:
        passage_bars = 4

    candidates = collect_candidates(results, passage_bars, passage_qn)
    print(f"\nCandidate passages with removals: {len(candidates)}")

    if not candidates:
        print("No passages with doublings found; nothing to sample.")
        return

    save_samples(
        candidates, args.output_folder, args.samples, args.seed,
        passage_bars, passage_qn, meta,
    )
