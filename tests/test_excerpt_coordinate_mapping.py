"""Test that before/after excerpts select the same musical passage.

Regression test: when remove_repeated_bars shifts the timeline, the
after excerpt must use mapped coordinates, not the original ones.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from tests.demos.demo_transforms import (
    FileResult,
    _NOTE_ID_COL,
    _collect_candidates,
    _save_samples,
)
from music_df.remove_repeated_bars import remove_repeated_bars
from music_df.sort_df import sort_df
from music_df.transforms import TRANSFORMS, _ensure_transforms_loaded


def _make_piece_with_repeat() -> pd.DataFrame:
    """Piece: bars A A B C D E F G — 8 bars, each 4 beats, 2 tracks.

    Bar 0 (onset 0):  A, pitches 60, 72 (track 0, 1)
    Bar 1 (onset 4):  A, pitches 60, 72 (repeat — will be removed)
    Bar 2 (onset 8):  B, pitches 62, 74
    Bar 3 (onset 12): C, pitches 64, 76
    Bar 4 (onset 16): D, pitches 65, 77
    Bar 5 (onset 20): E, pitches 67, 79
    Bar 6 (onset 24): F, pitches 69, 81
    Bar 7 (onset 28): G, pitches 72, 84

    After remove_repeated_bars, bar 1 is removed and everything shifts
    left by 4. Bar D ends up at onset 12 in the after DF.
    """
    rows = []
    rows.append({"type": "time_signature", "onset": 0.0, "release": np.nan,
                 "pitch": np.nan, "other": {"numerator": 4, "denominator": 4},
                 "track": np.nan})

    base_pitches = [60, 60, 62, 64, 65, 67, 69, 72]
    for i, pitch in enumerate(base_pitches):
        bar_onset = i * 4.0
        bar_release = (i + 1) * 4.0
        rows.append({"type": "bar", "onset": bar_onset, "release": bar_release,
                      "pitch": np.nan, "other": None, "track": np.nan})
        # Track 0 note
        rows.append({"type": "note", "onset": bar_onset, "release": bar_onset + 2.0,
                      "pitch": float(pitch), "other": None, "track": 0.0})
        # Track 1 note (octave higher)
        rows.append({"type": "note", "onset": bar_onset, "release": bar_onset + 2.0,
                      "pitch": float(pitch + 12), "other": None, "track": 1.0})
    # Trailing bar
    rows.append({"type": "bar", "onset": 32.0, "release": 36.0,
                  "pitch": np.nan, "other": None, "track": np.nan})

    return pd.DataFrame(rows)


def test_after_excerpt_uses_mapped_coordinates():
    """Simulates the real pipeline: notes are removed by dedouble (simulated
    via removed_by), then remove_repeated_bars shifts the timeline. The
    after excerpt at the shifted passage should match the before excerpt.

    Bar D is at onset 16 (before) and onset 12 (after). We mark one note
    at bar D as removed (as if by dedouble) so it becomes a candidate.
    Without the fix, the after excerpt uses onset 16 in the shifted
    timeline and shows bar E (pitch 67) instead of bar D (pitch 65).
    """
    _ensure_transforms_loaded()

    df = _make_piece_with_repeat()
    before = sort_df(df.copy())

    note_mask = before["type"] == "note"
    before.loc[note_mask, _NOTE_ID_COL] = range(int(note_mask.sum()))

    after = remove_repeated_bars(before.copy())

    after_notes = after[after["type"] == "note"]
    before_note_ids = set(before.loc[note_mask, _NOTE_ID_COL].dropna().astype(int))
    after_note_ids = set(after_notes[_NOTE_ID_COL].dropna().astype(int))

    # Notes removed by remove_repeated_bars (bar 1 = second A)
    removed_by_rrb = {nid: "remove_repeated_bars" for nid in before_note_ids - after_note_ids}

    # Simulate dedouble removing the track 1 note at bar D (onset 16).
    # Find its _note_id in the before df.
    bar_d_notes = before[
        (before["type"] == "note") & (before["onset"] == 16.0)
    ]
    track1_note = bar_d_notes[bar_d_notes["track"] == 1.0]
    assert not track1_note.empty
    dedouble_nid = int(track1_note.iloc[0][_NOTE_ID_COL])

    # Also remove it from after_df to simulate the dedouble transform
    after = after[
        ~((after[_NOTE_ID_COL] == dedouble_nid) & (after["type"] == "note"))
    ].copy()

    removed_by = {**removed_by_rrb, dedouble_nid: "dedouble"}

    # Get diff_bounds from remove_repeated_bars
    diff_func = TRANSFORMS["remove_repeated_bars"].diff_func
    diff_result = diff_func(before, after)
    diff_bounds = diff_result[2] if len(diff_result) > 2 else []

    bar_onset_map = after.attrs.get("bar_onset_map", {})

    result = FileResult(
        path=Path("test_piece.mid"),
        notes_before=int(note_mask.sum()),
        notes_after=int((after["type"] == "note").sum()),
        before_df=before,
        after_df=after,
        removed_by=removed_by,
        added_by={},
        diff_bounds=diff_bounds,
        bar_onset_map=bar_onset_map,
    )

    candidates = _collect_candidates([result], passage_bars=2, passage_qn=None)
    cand_at_16 = [c for c in candidates if c.bar_onset == 16.0]
    assert cand_at_16, (
        f"Expected candidate at onset 16.0, "
        f"got: {[c.bar_onset for c in candidates]}"
    )
    cand = cand_at_16[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir)
        _save_samples(
            [cand], output, n_samples=1, seed=42,
            passage_bars=2, passage_qn=None,
            meta={"test": True},
        )

        before_csv = pd.read_csv(output / "000_before.csv")
        after_csv = pd.read_csv(output / "000_after.csv")

        before_pitches = set(
            before_csv.loc[before_csv["type"] == "note", "pitch"]
        )
        after_pitches = set(
            after_csv.loc[after_csv["type"] == "note", "pitch"]
        )

        # Before excerpt at bar D should have pitch 65 (and 77 from track 1).
        # After excerpt should also have pitch 65 at its mapped location (onset 12).
        # Without the fix, after uses onset 16 in the shifted timeline and
        # shows bar E (67, 79) instead.
        assert 65.0 in before_pitches, f"Before should contain bar D (65): {before_pitches}"
        assert 65.0 in after_pitches, (
            f"After excerpt should contain bar D (pitch 65) at mapped coordinates, "
            f"but got pitches {sorted(after_pitches)}. "
            f"The after excerpt likely shows the wrong passage."
        )
