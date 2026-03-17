"""Demo: apply a pipeline of registered transforms to a folder of music files.

Reads files, applies the specified transforms in order, reports summary
statistics, and saves sampled before/after passage excerpts as CSVs with
changed notes labeled with the responsible transform name in a ``label`` column.

Usage:
    python demo_transforms.py input/ output/ --transforms quantize_df,salami_slice,dedouble

    # With per-transform parameters (JSON or YAML):
    python demo_transforms.py input/ output/ --transforms quantize_df,dedouble \\
        --params params.yaml

    # List available transforms:
    python demo_transforms.py --list

Where params.yaml looks like:

    quantize_df:
      tpq: 16
    dedouble:
      match_releases: false
"""

from __future__ import annotations

import argparse
import json
import random
import sys

import yaml
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from statistics import mean, median

import pandas as pd
from tqdm import tqdm

from music_df.add_feature import infer_barlines
from music_df.crop_df import crop_df, get_context_bars
from music_df.read import read
from music_df.slice_df import slice_df
from music_df.sort_df import sort_df
from music_df.transforms import TRANSFORMS, _ensure_transforms_loaded, apply_transforms

SUPPORTED_EXTENSIONS = {".csv", ".mid", ".midi", ".xml", ".mxl", ".mscx", ".mscz", ".krn"}

# Columns that together identify a "note identity" for symmetric diffing
_NOTE_KEY_COLS = ("onset", "release", "pitch")

TICKS_PER_QUARTER = 16


def _quantize_for_comparison(df: pd.DataFrame, tpq: int = TICKS_PER_QUARTER) -> pd.DataFrame:
    """Round note onsets/releases to a grid for stable comparisons."""
    df = df.copy()
    note_mask = df["type"] == "note"
    df.loc[note_mask, "onset"] = (df.loc[note_mask, "onset"] * tpq).round() / tpq
    df.loc[note_mask, "release"] = (df.loc[note_mask, "release"] * tpq).round() / tpq
    return df


def _note_tuples(df: pd.DataFrame) -> set[tuple]:
    """Extract the set of (onset, release, pitch) tuples from note rows."""
    notes = df[df["type"] == "note"]
    cols = [c for c in _NOTE_KEY_COLS if c in notes.columns]
    return set(notes[cols].itertuples(index=False, name=None))


def _label_notes(
    df: pd.DataFrame, label_map: dict[tuple, str],
) -> pd.DataFrame:
    """Add a ``label`` column with the transform name for changed notes, NA otherwise."""
    df = df.copy()
    if not label_map:
        df["label"] = pd.NA
        return df
    notes = df["type"] == "note"
    cols = [c for c in _NOTE_KEY_COLS if c in df.columns]
    tuples = df[cols].apply(tuple, axis=1)
    df["label"] = tuples.map(label_map).where(notes, other=pd.NA)
    return df


# ---------------------------------------------------------------------------
# File processing
# ---------------------------------------------------------------------------


@dataclass
class FileResult:
    path: Path
    notes_before: int
    notes_after: int
    before_df: pd.DataFrame
    after_df: pd.DataFrame
    # Per-transform diffs: maps note tuple -> transform name
    removed_by: dict[tuple, str]
    added_by: dict[tuple, str]


def _ensure_barlines(df: pd.DataFrame) -> pd.DataFrame:
    if "bar" not in df["type"].values:
        df = infer_barlines(df)
    return df


def _process_one(path: Path, steps: list[dict[str, dict]]) -> FileResult | None:
    try:
        df = sort_df(read(str(path)))
    except Exception as exc:
        print(f"  WARNING: failed to read {path.name}: {exc}", file=sys.stderr)
        return None

    df = _ensure_barlines(df)
    notes_before = int((df["type"] == "note").sum())

    # Apply transforms one at a time, tracking which notes each step changes.
    # Earlier transforms take priority (a note is attributed to the first
    # transform that touches it).
    removed_by: dict[tuple, str] = {}
    added_by: dict[tuple, str] = {}
    current = df
    for step in steps:
        (name, kwargs), = step.items()
        before_q = _quantize_for_comparison(current)
        current = apply_transforms(current, [step])
        after_q = _quantize_for_comparison(current)

        custom_diff = getattr(TRANSFORMS[name], "diff_func", None)
        if custom_diff is not None:
            removed, added = custom_diff(before_q, after_q)
        else:
            removed = _note_tuples(before_q) - _note_tuples(after_q)
            added = _note_tuples(after_q) - _note_tuples(before_q)

        for t in removed:
            removed_by.setdefault(t, name)
        for t in added:
            added_by.setdefault(t, name)

    notes_after = int((current["type"] == "note").sum())

    return FileResult(
        path=path,
        notes_before=notes_before,
        notes_after=notes_after,
        before_df=df,
        after_df=current,
        removed_by=removed_by,
        added_by=added_by,
    )


# ---------------------------------------------------------------------------
# Excerpt extraction
# ---------------------------------------------------------------------------

def _slice_to_excerpt(
    df: pd.DataFrame, start_time: float, end_time: float,
) -> pd.DataFrame:
    """Slice notes at excerpt boundaries and return notes within range."""
    sliced = slice_df(df, [start_time, end_time])
    note_mask = sliced["type"] == "note"
    in_range = (sliced["onset"] >= start_time) & (sliced["onset"] < end_time)
    result = sliced[note_mask & in_range]
    return result.drop(columns=["sliced"])


def _get_passage_end(
    bar_onsets: list[float],
    bar_idx: int,
    passage_bars: int | None,
    passage_qn: float | None,
    df: pd.DataFrame,
) -> float:
    if passage_bars is not None:
        if bar_idx + passage_bars < len(bar_onsets):
            return bar_onsets[bar_idx + passage_bars]
        return df.loc[df["type"] == "note", "release"].max()
    assert passage_qn is not None
    return bar_onsets[bar_idx] + passage_qn


@dataclass
class Candidate:
    filename: str
    bar_onset: float
    n_changed: int
    file_result: FileResult


def _collect_candidates(
    results: list[FileResult],
    passage_bars: int | None,
    passage_qn: float | None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for result in results:
        if not result.removed_by and not result.added_by:
            continue
        df = result.before_df
        bar_rows = df[df["type"] == "bar"]
        if bar_rows.empty:
            continue
        bar_onsets = sorted(bar_rows["onset"].unique())

        changed = set(result.removed_by) | set(result.added_by)

        for bar_idx, bar_onset in enumerate(bar_onsets):
            end_onset = _get_passage_end(
                bar_onsets, bar_idx, passage_bars, passage_qn, df,
            )
            if passage_bars is not None and bar_idx + passage_bars > len(bar_onsets):
                continue

            n_changed = sum(
                1 for onset, release, pitch in changed
                if bar_onset <= onset < end_onset
            )
            if n_changed > 0:
                candidates.append(
                    Candidate(
                        filename=result.path.name,
                        bar_onset=bar_onset,
                        n_changed=n_changed,
                        file_result=result,
                    )
                )
    return candidates


def _crop_passage(
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


def _save_samples(
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

        orig_df = cand.file_result.before_df
        after_df = cand.file_result.after_df
        bar_onsets = sorted(
            orig_df.loc[orig_df["type"] == "bar", "onset"].unique()
        )
        bar_idx = bar_onsets.index(start_time)
        end_time = _get_passage_end(
            bar_onsets, bar_idx, passage_bars, passage_qn, orig_df,
        )

        # Get non-note metadata rows (time_signature, bar, etc.)
        before_full = _crop_passage(orig_df, cand.bar_onset, passage_bars, passage_qn)
        non_notes = before_full[before_full["type"] != "note"]

        # Slice notes at excerpt boundaries
        before_notes = _slice_to_excerpt(orig_df, start_time, end_time)
        before = sort_df(pd.concat([non_notes, before_notes], ignore_index=True))
        before = _quantize_for_comparison(before)

        after_notes = _slice_to_excerpt(after_df, start_time, end_time)
        after = sort_df(pd.concat([non_notes, after_notes], ignore_index=True))
        after = _quantize_for_comparison(after)

        before = _label_notes(before, cand.file_result.removed_by)
        after = _label_notes(after, cand.file_result.added_by)

        before.to_csv(output_folder / f"{excerpt_id}_before.csv", index=False)
        after.to_csv(output_folder / f"{excerpt_id}_after.csv", index=False)

        lookup_rows.append(
            {
                "id": excerpt_id,
                "source_file": cand.filename,
                "bar_onset": cand.bar_onset,
                "notes_changed": cand.n_changed,
            }
        )

    pd.DataFrame(lookup_rows).to_csv(output_folder / "lookup.csv", index=False)

    meta["n_candidates"] = len(candidates)
    meta["n_sampled"] = len(sampled)
    (output_folder / "sample_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nSaved {len(sampled)} excerpt pairs to {output_folder}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_steps(
    names: list[str],
    params: dict[str, dict],
) -> list[dict[str, dict]]:
    return [{name: params.get(name, {})} for name in names]


def _handle_list(argv: list[str] | None = None) -> bool:
    """If --list is in argv, print available transforms and return True."""
    args = argv if argv is not None else sys.argv[1:]
    if "--list" not in args:
        return False
    _ensure_transforms_loaded()
    print("Available transforms:")
    for name in sorted(TRANSFORMS):
        print(f"  {name}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a pipeline of transforms to a folder of music files.",
    )
    parser.add_argument("input_folder", type=Path, help="Folder of music files")
    parser.add_argument("output_folder", type=Path, help="Where to write output CSVs")
    parser.add_argument(
        "--transforms", type=str, required=True,
        help="Comma-separated transform names, applied in order.",
    )
    parser.add_argument(
        "--params", type=Path, default=None,
        help="JSON or YAML file with per-transform parameters.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Process at most N files")
    parser.add_argument("--samples", type=int, default=10, help="Sample up to M passages")

    length_group = parser.add_mutually_exclusive_group()
    length_group.add_argument("--bars", type=int, default=None, help="Passage length in bars")
    length_group.add_argument(
        "--quarter-notes", type=float, default=None,
        help="Passage length in quarter notes",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if _handle_list(argv):
        return

    args = parse_args(argv)
    _ensure_transforms_loaded()

    names = [t.strip() for t in args.transforms.split(",")]
    for name in names:
        if name not in TRANSFORMS:
            print(
                f"Unknown transform: {name!r}. "
                f"Available: {', '.join(sorted(TRANSFORMS))}",
                file=sys.stderr,
            )
            sys.exit(1)

    params: dict[str, dict] = {}
    if args.params is not None:
        text = args.params.read_text()
        if args.params.suffix in (".yaml", ".yml"):
            params = yaml.safe_load(text)
        else:
            params = json.loads(text)

    steps = _build_steps(names, params)

    paths = [
        p for p in args.input_folder.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    rng = random.Random(args.seed)
    rng.shuffle(paths)
    if args.max_files is not None:
        paths = paths[: args.max_files]

    if not paths:
        print(f"No supported files found in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(paths)} files to process")
    print(f"Transforms: {' -> '.join(names)}")

    results: list[FileResult] = []
    for path in tqdm(paths, desc="Processing"):
        result = _process_one(path, steps)
        if result is not None:
            results.append(result)

    if not results:
        print("No files were successfully processed.", file=sys.stderr)
        sys.exit(1)

    # Summary
    total_before = sum(r.notes_before for r in results)
    total_after = sum(r.notes_after for r in results)
    files_changed = [r for r in results if r.notes_before != r.notes_after]

    print(f"\n=== Summary ===")
    print(f"Files processed:   {len(results)}")
    print(f"Files changed:     {len(files_changed)}")
    print(f"Notes before:      {total_before}")
    print(f"Notes after:       {total_after}")
    print(f"Notes removed:     {total_before - total_after}")
    if files_changed:
        deltas = [r.notes_before - r.notes_after for r in files_changed]
        print(f"Mean delta/file:   {mean(deltas):.1f}")
        print(f"Median delta/file: {median(deltas):.1f}")

    # Collect and save excerpt samples
    passage_bars = args.bars
    passage_qn = args.quarter_notes
    if passage_bars is None and passage_qn is None:
        passage_bars = 4

    candidates = _collect_candidates(results, passage_bars, passage_qn)
    print(f"\nCandidate passages with changes: {len(candidates)}")

    if not candidates:
        print("No passages with changes found; nothing to sample.")
        return

    meta = {
        "input_folder": str(args.input_folder),
        "transforms": names,
        "params": params,
        "max_files": args.max_files,
        "seed": args.seed,
        "files_processed": len(results),
        "notes_before": total_before,
        "notes_after": total_after,
    }
    _save_samples(
        candidates, args.output_folder, args.samples, args.seed,
        passage_bars, passage_qn, meta,
    )


if __name__ == "__main__":
    main()
