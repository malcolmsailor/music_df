"""Demo: apply a pipeline of registered transforms to a folder of music files.

Reads files, applies the specified transforms in order, reports summary
statistics, and saves sampled before/after passage excerpts as CSVs with
changed notes labeled with the responsible transform name in a ``label`` column.

Usage:
    python demo_transforms.py input/ output/ --transforms quantize_df,salami_slice,dedouble

    # With per-transform parameters (JSON or YAML):
    python demo_transforms.py input/ output/ --transforms quantize_df,dedouble \\
        --params params.yaml

    # Or pass a YAML file specifying transforms as a list of dicts:
    python demo_transforms.py input/ output/ --transforms transforms.yaml

    # List available transforms:
    python demo_transforms.py --list

Where params.yaml looks like:

    quantize_df:
      tpq: 16
    dedouble:
      match_releases: false

And transforms.yaml (list-of-dicts format) looks like:

    - quantize_df:
        tpq: 16
    - dedouble:
        match_releases: false
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import sys
import time
import traceback
from collections import Counter

import yaml
from dataclasses import dataclass, field
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

# Temporary column used to track note identity through transforms
_NOTE_ID_COL = "_note_id"

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


def _label_notes_by_id(
    df: pd.DataFrame, label_map: dict[int, str],
) -> pd.DataFrame:
    """Add a ``label`` column using _note_id -> transform name mapping."""
    df = df.copy()
    if not label_map:
        df["label"] = pd.NA
        return df
    notes = df["type"] == "note"
    ids = df[_NOTE_ID_COL].where(notes)
    df["label"] = ids.map(label_map).where(notes, other=pd.NA)
    return df


def _label_notes_by_tuple(
    df: pd.DataFrame, label_map: dict[tuple, str],
) -> pd.DataFrame:
    """Add a ``label`` column using (onset, release, pitch) -> transform name mapping."""
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
    # Per-transform diffs: maps _note_id -> transform name
    removed_by: dict[int, str]
    # Per-transform diffs: maps note tuple -> transform name
    added_by: dict[tuple, str]
    # Time intervals that should be shown intact in excerpts.
    # Each bound is (before_start, before_end) or
    # (before_start, before_end, after_start, after_end) for transforms
    # that shift the timeline.
    diff_bounds: list[tuple] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    # Maps original bar onsets to shifted bar onsets (from remove_repeated_bars)
    bar_onset_map: dict[float, float] = field(default_factory=dict)


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

    # Assign stable IDs to note rows so we can track them through transforms
    # that modify onset/release values.
    note_mask = df["type"] == "note"
    df.loc[note_mask, _NOTE_ID_COL] = range(int(note_mask.sum()))

    # Apply transforms one at a time, tracking which notes each step changes.
    # Earlier transforms take priority (a note is attributed to the first
    # transform that touches it).
    removed_by: dict[int, str] = {}
    added_by: dict[tuple, str] = {}
    timings: dict[str, float] = {}
    all_diff_bounds: list[tuple[float, float]] = []
    current = df
    for step in steps:
        (name, kwargs), = step.items()
        # Track which note IDs exist before this step
        ids_before = set(
            current.loc[current["type"] == "note", _NOTE_ID_COL]
            .dropna().astype(int)
        )
        before_q = _quantize_for_comparison(current)

        t0 = time.perf_counter()
        current = apply_transforms(current, [step])
        timings[name] = time.perf_counter() - t0

        after_notes = current[current["type"] == "note"]
        ids_after = set(after_notes[_NOTE_ID_COL].dropna().astype(int))

        # Removals: IDs that disappeared
        for nid in ids_before - ids_after:
            if nid in removed_by:
                removed_by[nid] += "+" + name
            else:
                removed_by[nid] = name

        # Additions: notes without an ID were created by this step;
        # track by (onset, release, pitch) tuple in the after state.
        after_q = _quantize_for_comparison(current)
        new_notes = after_q[(after_q["type"] == "note") & after_q[_NOTE_ID_COL].isna()]
        if not new_notes.empty:
            cols = [c for c in _NOTE_KEY_COLS if c in new_notes.columns]
            for t in new_notes[cols].itertuples(index=False, name=None):
                if t in added_by:
                    added_by[t] += "+" + name
                else:
                    added_by[t] = name
            # Give new notes IDs so later transforms can track them
            max_id = after_notes[_NOTE_ID_COL].dropna().max()
            next_id = int(max_id) + 1 if not pd.isna(max_id) else 0
            new_mask = (current["type"] == "note") & current[_NOTE_ID_COL].isna()
            current.loc[new_mask, _NOTE_ID_COL] = range(
                next_id, next_id + int(new_mask.sum())
            )

        # Extract diff_bounds from custom diff_func if available
        custom_diff = getattr(TRANSFORMS[name], "diff_func", None)
        if custom_diff is not None:
            diff_result = custom_diff(before_q, after_q)
            if len(diff_result) > 2:
                all_diff_bounds.extend(diff_result[2])

    notes_after = int((current["type"] == "note").sum())

    bar_onset_map = current.attrs.get("bar_onset_map", {})

    return FileResult(
        path=path,
        notes_before=notes_before,
        notes_after=notes_after,
        before_df=df,
        after_df=current,
        removed_by=removed_by,
        added_by=added_by,
        diff_bounds=all_diff_bounds,
        timings=timings,
        bar_onset_map=bar_onset_map,
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

        # Collect onsets of all changed notes for passage counting.
        # removed_by keys are _note_ids; look up their onsets from before_df.
        removed_onsets = []
        if result.removed_by:
            notes = df[df["type"] == "note"]
            for nid in result.removed_by:
                matches = notes[notes[_NOTE_ID_COL] == nid]
                if not matches.empty:
                    removed_onsets.append(matches.iloc[0]["onset"])
        added_onsets = [onset for onset, release, pitch in result.added_by]
        changed_onsets = removed_onsets + added_onsets

        for bar_idx, bar_onset in enumerate(bar_onsets):
            end_onset = _get_passage_end(
                bar_onsets, bar_idx, passage_bars, passage_qn, df,
            )
            if passage_bars is not None and bar_idx + passage_bars > len(bar_onsets):
                continue

            n_changed = sum(
                1 for onset in changed_onsets
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

        # Expand window to cover any overlapping diff bounds so that
        # the full repeat pattern is visible in the excerpt.
        # Track after-coordinate bounds separately for transforms that
        # shift the timeline (e.g. remove_repeated_bars).
        after_start_time = None
        after_end_time = None
        for bound in cand.file_result.diff_bounds:
            bound_start, bound_end = bound[0], bound[1]
            if bound_start < end_time and bound_end > start_time:
                start_time = min(start_time, bound_start)
                end_time = max(end_time, bound_end)
                if len(bound) >= 4:
                    ab_start, ab_end = bound[2], bound[3]
                    if after_start_time is None:
                        after_start_time = ab_start
                        after_end_time = ab_end
                    else:
                        after_start_time = min(after_start_time, ab_start)
                        after_end_time = max(after_end_time, ab_end)

        # Snap to bar boundaries
        bar_idx_start = bisect.bisect_right(bar_onsets, start_time) - 1
        start_time = bar_onsets[max(0, bar_idx_start)]
        bar_idx_end = bisect.bisect_left(bar_onsets, end_time)
        if bar_idx_end < len(bar_onsets):
            end_time = bar_onsets[bar_idx_end]
        else:
            end_time = orig_df.loc[orig_df["type"] == "note", "release"].max()

        # Compute effective passage size for cropping (may differ from
        # the original passage_bars/passage_qn if diff bounds expanded
        # the window).
        eff_bars: int | None = None
        eff_qn: float | None = None
        if passage_bars is not None:
            start_idx = bar_onsets.index(start_time)
            end_idx = bisect.bisect_left(bar_onsets, end_time)
            eff_bars = max(1, end_idx - start_idx)
        else:
            eff_qn = end_time - start_time

        # If no after-coordinate bounds were provided, use bar_onset_map
        # to translate before coordinates to after coordinates (handles
        # timeline shifts from remove_repeated_bars). Falls back to the
        # same window if no mapping is available.
        if after_start_time is None:
            bom = cand.file_result.bar_onset_map
            if bom:
                after_start_time = bom.get(start_time, start_time)
                after_end_time = bom.get(end_time, end_time)
            else:
                after_start_time = start_time
                after_end_time = end_time
        # Snap after window to bar boundaries in the after_df
        after_bar_onsets = sorted(
            after_df.loc[after_df["type"] == "bar", "onset"].unique()
        )
        if after_bar_onsets:
            abi_start = bisect.bisect_right(after_bar_onsets, after_start_time) - 1
            after_start_time = after_bar_onsets[max(0, abi_start)]
            abi_end = bisect.bisect_left(after_bar_onsets, after_end_time)
            if abi_end < len(after_bar_onsets):
                after_end_time = after_bar_onsets[abi_end]
            else:
                after_end_time = after_df.loc[
                    after_df["type"] == "note", "release"
                ].max()

        # Compute effective after passage size
        after_eff_bars: int | None = None
        after_eff_qn: float | None = None
        if passage_bars is not None and after_bar_onsets:
            a_start_idx = bisect.bisect_left(after_bar_onsets, after_start_time)
            a_end_idx = bisect.bisect_left(after_bar_onsets, after_end_time)
            after_eff_bars = max(1, a_end_idx - a_start_idx)
        else:
            after_eff_qn = after_end_time - after_start_time

        # Get non-note metadata rows (time_signature, bar, etc.)
        before_full = _crop_passage(orig_df, start_time, eff_bars, eff_qn)
        before_non_notes = before_full[before_full["type"] != "note"]

        # Slice notes at excerpt boundaries
        before_notes = _slice_to_excerpt(orig_df, start_time, end_time)
        before = sort_df(pd.concat([before_non_notes, before_notes], ignore_index=True), force=True)
        before = _quantize_for_comparison(before)

        after_full = _crop_passage(
            after_df, after_start_time, after_eff_bars, after_eff_qn,
        )
        after_non_notes = after_full[after_full["type"] != "note"]
        after_notes = _slice_to_excerpt(after_df, after_start_time, after_end_time)
        after = sort_df(pd.concat([after_non_notes, after_notes], ignore_index=True), force=True)
        after = _quantize_for_comparison(after)

        before = _label_notes_by_id(before, cand.file_result.removed_by)
        after = _label_notes_by_tuple(after, cand.file_result.added_by)

        before.drop(columns=[_NOTE_ID_COL], errors="ignore").to_csv(
            output_folder / f"{excerpt_id}_before.csv", index=False,
        )
        after.drop(columns=[_NOTE_ID_COL], errors="ignore").to_csv(
            output_folder / f"{excerpt_id}_after.csv", index=False,
        )

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

def _resolve_transforms(raw: str) -> list[dict[str, dict]]:
    """Parse --transforms value: either comma-separated names or a YAML file path.

    When *raw* points to an existing YAML/YML file, load it and expect a list
    of single-key dicts (same format that :func:`apply_transforms` accepts).
    Otherwise treat *raw* as comma-separated transform names.
    """
    path = Path(raw)
    if path.suffix.lower() in (".yaml", ".yml") and path.is_file():
        result = yaml.safe_load(path.read_text())
        assert isinstance(result, list), (
            f"Expected list of transforms in {path}, got {type(result).__name__}"
        )
        return result
    return [{name.strip(): {}} for name in raw.split(",")]


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
        help="Comma-separated transform names, or path to a YAML file "
             "containing a list of {name: params} dicts.",
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

    steps = _resolve_transforms(args.transforms)
    params = {}

    # If loaded from a YAML file, --params is ignored (params are inline).
    # Otherwise merge in any --params file.
    transforms_from_file = Path(args.transforms).suffix.lower() in (".yaml", ".yml") and Path(args.transforms).is_file()
    if transforms_from_file:
        params = {list(s.keys())[0]: list(s.values())[0] or {} for s in steps}
    elif args.params is not None:
        text = args.params.read_text()
        if args.params.suffix in (".yaml", ".yml"):
            params = yaml.safe_load(text)
        else:
            params = json.loads(text)
        names = [list(s.keys())[0] for s in steps]
        steps = _build_steps(names, params)

    # Validate all transform names
    for step in steps:
        (name, _), = step.items()
        if name not in TRANSFORMS:
            print(
                f"Unknown transform: {name!r}. "
                f"Available: {', '.join(sorted(TRANSFORMS))}",
                file=sys.stderr,
            )
            sys.exit(1)

    names = [list(s.keys())[0] for s in steps]

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
        try:
            result = _process_one(path, steps)
        except Exception:
            print(f"\nError processing {path}:", file=sys.stderr)
            traceback.print_exc()
            continue
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

    # Per-transform breakdown
    per_transform_removed: Counter[str] = Counter()
    per_transform_added: Counter[str] = Counter()
    per_transform_deltas: dict[str, list[int]] = {n: [] for n in names}
    for r in results:
        file_removed: Counter[str] = Counter()
        file_added: Counter[str] = Counter()
        for _tuple, tname in r.removed_by.items():
            per_transform_removed[tname] += 1
            file_removed[tname] += 1
        for _tuple, tname in r.added_by.items():
            per_transform_added[tname] += 1
            file_added[tname] += 1
        for n in names:
            per_transform_deltas[n].append(file_removed[n] - file_added[n])

    name_w = max(len(n) for n in names)
    header = (
        f"  {'transform':<{name_w}}  {'removed':>8}  {'added':>8}"
        f"  {'net':>8}  {'mean/file':>10}  {'median/file':>12}"
    )
    print(f"\n{header}")
    print(f"  {'-' * (len(header) - 2)}")
    for n in names:
        removed = per_transform_removed[n]
        added = per_transform_added[n]
        net = removed - added
        deltas_n = per_transform_deltas[n]
        mu = mean(deltas_n) if deltas_n else 0.0
        med = median(deltas_n) if deltas_n else 0.0
        print(
            f"  {n:<{name_w}}  {removed:>8}  {added:>8}"
            f"  {net:>8}  {mu:>10.1f}  {med:>12.1f}"
        )

    # Per-transform timing breakdown
    per_transform_times: dict[str, list[float]] = {n: [] for n in names}
    for r in results:
        for n in names:
            per_transform_times[n].append(r.timings.get(n, 0.0))

    t_header = (
        f"  {'transform':<{name_w}}  {'total(s)':>9}  {'mean(ms)':>9}"
        f"  {'median(ms)':>11}  {'max(ms)':>9}"
    )
    print(f"\n{t_header}")
    print(f"  {'-' * (len(t_header) - 2)}")
    for n in names:
        times = per_transform_times[n]
        total_s = sum(times)
        times_ms = [t * 1000 for t in times]
        mu = mean(times_ms) if times_ms else 0.0
        med = median(times_ms) if times_ms else 0.0
        mx = max(times_ms) if times_ms else 0.0
        print(
            f"  {n:<{name_w}}  {total_s:>9.2f}  {mu:>9.1f}"
            f"  {med:>11.1f}  {mx:>9.1f}"
        )

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
