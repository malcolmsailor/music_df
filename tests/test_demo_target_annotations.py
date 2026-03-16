"""Test that demo helper target annotations are correct.

Regression test for a bug where infer_barlines() in collect_candidates()
reset the DataFrame index, causing involved_indices (computed against the
original index) to mismatch the cropped excerpt's index.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from music_df.dedouble_instruments import dedouble_octaves
from music_df.sort_df import sort_df

# demo helpers live under tests/demos/
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "demos"))
from _demo_helpers import (
    FileResult,
    collect_candidates,
    crop_passage,
    save_samples,
)


def _make_octave_doubling_df_without_barlines() -> pd.DataFrame:
    """Build a small music_df with octave doubling and NO bar rows.

    Two tracks play a 4-note melody in octaves.  A time signature is
    present but barlines are absent, so collect_candidates will need to
    call infer_barlines().
    """
    rows = [
        # time signature (4/4)
        {"type": "time_signature", "onset": 0.0, "release": np.nan,
         "track": np.nan, "channel": np.nan, "pitch": np.nan,
         "velocity": np.nan, "other": "{'numerator': 4, 'denominator': 4}"},
    ]
    # 8-note melody on two tracks an octave apart (covers 2 bars of 4/4)
    for i in range(8):
        onset = float(i)
        release = onset + 1.0
        rows.append({"type": "note", "onset": onset, "release": release,
                      "track": 1.0, "channel": 0.0, "pitch": 60.0 + i,
                      "velocity": 80.0, "other": np.nan})
        rows.append({"type": "note", "onset": onset, "release": release,
                      "track": 2.0, "channel": 0.0, "pitch": 72.0 + i,
                      "velocity": 80.0, "other": np.nan})
    df = pd.DataFrame(rows)
    df = sort_df(df)
    assert "bar" not in df["type"].values
    return df


class TestBeforeTargetAnnotation:
    """Verify that 'target' column in before excerpts marks the right notes."""

    def _process_df(self, df: pd.DataFrame) -> FileResult:
        """Mimic what the demo pipeline does: transform, compute indices."""
        result = dedouble_octaves(
            df, instrument_columns=["track"], min_length=2
        )
        original_indices = set(df.index)
        kept_indices = set(result["original_index"])
        dropped = original_indices - kept_indices

        # Compute involved_indices the same way demo_dedouble_octaves does
        from demo_dedouble_octaves import _find_octave_partner_indices

        partners = _find_octave_partner_indices(df, result, dropped)
        involved = dropped | partners

        return FileResult(
            path=Path("test_file.mid"),
            notes_before=int((df["type"] == "note").sum()),
            notes_after=result.attrs["n_dedoubled_notes"],
            dropped_indices=dropped,
            involved_indices=involved,
            original_df=df,
            dedoubled_df=result,
        )

    def test_before_targets_match_involved_notes(self):
        """Target=True notes in 'before' must be exactly the involved notes."""
        df = _make_octave_doubling_df_without_barlines()
        file_result = self._process_df(df)

        # Sanity: there should be some dropped notes
        assert file_result.dropped_indices, "Expected some notes to be dropped"

        # Record the original involved indices for later comparison
        original_involved = file_result.involved_indices.copy()
        # Record which (onset, pitch) tuples are involved
        notes = df[df["type"] == "note"]
        involved_notes_key = set()
        for idx in original_involved:
            if idx in notes.index:
                row = notes.loc[idx]
                involved_notes_key.add((row["onset"], row["pitch"]))

        # Run collect_candidates (this is where infer_barlines may reset index)
        candidates = collect_candidates(
            [file_result], passage_bars=None, passage_qn=8.0
        )
        assert candidates, "Expected at least one candidate passage"

        cand = candidates[0]

        # Crop the 'before' excerpt the same way save_samples does
        before = crop_passage(
            cand.file_result.original_df, cand.bar_onset,
            passage_bars=None, passage_qn=8.0,
        )

        involved = cand.file_result.involved_indices
        if "_src_idx" in before.columns:
            before["target"] = before["_src_idx"].isin(involved)
        else:
            before["target"] = before.index.isin(involved)

        # Check: every note marked target=True should be an involved note
        before_notes = before[before["type"] == "note"]
        target_true = before_notes[before_notes["target"]]
        target_keys = set(zip(target_true["onset"], target_true["pitch"]))
        assert target_keys <= involved_notes_key, (
            f"target=True notes not in involved set: "
            f"{target_keys - involved_notes_key}"
        )

        # Check: every involved note in the excerpt should be target=True
        excerpt_involved_keys = set()
        for _, row in before_notes.iterrows():
            if (row["onset"], row["pitch"]) in involved_notes_key:
                excerpt_involved_keys.add((row["onset"], row["pitch"]))
        assert excerpt_involved_keys <= target_keys, (
            f"Involved notes missing target=True: "
            f"{excerpt_involved_keys - target_keys}"
        )
