"""Tests for the to_absolute_time transform."""

import numpy as np
import pandas as pd

from music_df.to_absolute_time import TARGET_BPM, from_absolute_time, to_absolute_time
from music_df.transforms import apply_transforms


def _note(onset, release, pitch=60, other=None):
    return {
        "type": "note",
        "pitch": float(pitch),
        "onset": float(onset),
        "release": float(release),
        "other": other,
    }


def _tempo(onset, bpm):
    return {
        "type": "tempo",
        "pitch": float("nan"),
        "onset": float(onset),
        "release": float("nan"),
        "other": {"tempo": float(bpm)},
    }


def _bar(onset, release):
    return {
        "type": "bar",
        "pitch": float("nan"),
        "onset": float(onset),
        "release": float(release),
        "other": None,
    }


def _time_sig(onset, numerator, denominator):
    return {
        "type": "time_signature",
        "pitch": float("nan"),
        "onset": float(onset),
        "release": float("nan"),
        "other": {"numerator": int(numerator), "denominator": int(denominator)},
    }


def _only_tempo_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"] == "tempo"]


def test_no_tempo_rows_prepends_synthetic_120():
    # If the input has no tempo rows, onsets stay put but a synthetic 120 row
    # is prepended so the output schema matches the general case.
    df = pd.DataFrame([_note(0.0, 1.0, 60), _note(1.0, 2.0, 62)])
    out = to_absolute_time(df)

    tempo_rows = _only_tempo_rows(out)
    assert len(tempo_rows) == 1
    assert tempo_rows.iloc[0]["onset"] == 0.0
    assert tempo_rows.iloc[0]["other"] == {"tempo": TARGET_BPM}

    # Notes are unchanged.
    notes = out[out["type"] == "note"].reset_index(drop=True)
    assert list(notes["onset"]) == [0.0, 1.0]
    assert list(notes["release"]) == [1.0, 2.0]
    assert out.attrs["absolute_time"] is True


def test_single_120_tempo_row_is_effectively_identity():
    # The onsets/releases should be unchanged and there should still be only
    # one tempo row (the synthetic one, which replaces the input's 120 row).
    df = pd.DataFrame([_tempo(0.0, 120.0), _note(0.0, 1.0, 60), _note(1.0, 2.0, 62)])
    out = to_absolute_time(df)
    notes = out[out["type"] == "note"].reset_index(drop=True)
    assert list(notes["onset"]) == [0.0, 1.0]
    assert list(notes["release"]) == [1.0, 2.0]
    assert len(_only_tempo_rows(out)) == 1


def test_90_bpm_user_example():
    # User's motivating example: at 90 bpm, quarter notes become length 4/3.
    df = pd.DataFrame([_tempo(0.0, 90.0), _note(0.0, 1.0, 60), _note(1.0, 2.0, 62)])
    out = to_absolute_time(df)
    notes = out[out["type"] == "note"].reset_index(drop=True)
    expected_slope = TARGET_BPM / 90.0  # 4/3
    assert np.allclose(notes["onset"], [0.0, expected_slope])
    assert np.allclose(notes["release"], [expected_slope, 2 * expected_slope])


def test_multi_segment_piecewise_remap():
    # 120 bpm for [0, 2), then 60 bpm for [2, 4), then 240 bpm for [4, 6).
    # Slopes: 1, 2, 0.5. Segment new-starts: 0, 2, 2 + 2*2 = 6. At original
    # onset 6 the new onset is 6 + 0.5 * 0 = 6. Wait: the last note at 5 maps
    # to 6 + 0.5 * (5 - 4) = 6.5.
    df = pd.DataFrame(
        [
            _tempo(0.0, 120.0),
            _tempo(2.0, 60.0),
            _tempo(4.0, 240.0),
            _note(0.0, 2.0, 60),  # slope 1: 0 -> 0, 2 -> 2
            _note(2.0, 4.0, 62),  # slope 2: 2 -> 2, 4 -> 6
            _note(4.0, 6.0, 64),  # slope 0.5: 4 -> 6, 6 -> 7
            _note(4.5, 5.5, 65),  # slope 0.5: 4.5 -> 6.25, 5.5 -> 6.75
        ]
    )
    out = to_absolute_time(df)
    notes = out[out["type"] == "note"].reset_index(drop=True)
    assert np.allclose(notes["onset"], [0.0, 2.0, 6.0, 6.25])
    assert np.allclose(notes["release"], [2.0, 6.0, 7.0, 6.75])


def test_bars_remap_in_lockstep_with_notes():
    # A bar covering [0, 4) at 60 bpm should end at new beat 8, matching a
    # note that spans the same range.
    df = pd.DataFrame(
        [
            _tempo(0.0, 60.0),
            _bar(0.0, 4.0),
            _note(0.0, 4.0, 60),
        ]
    )
    out = to_absolute_time(df)
    bar = out[out["type"] == "bar"].iloc[0]
    note = out[out["type"] == "note"].iloc[0]
    assert bar["onset"] == note["onset"] == 0.0
    assert bar["release"] == note["release"] == 8.0


def test_tempo_rows_dropped_from_output():
    df = pd.DataFrame(
        [
            _tempo(0.0, 120.0),
            _tempo(2.0, 60.0),
            _tempo(4.0, 120.0),
            _note(0.0, 6.0, 60),
        ]
    )
    out = to_absolute_time(df)
    assert len(_only_tempo_rows(out)) == 1
    # The one remaining tempo row is the synthetic 120 one.
    assert _only_tempo_rows(out).iloc[0]["other"] == {"tempo": TARGET_BPM}


def test_duration_column_recomputed_when_present():
    df = pd.DataFrame(
        [
            {**_tempo(0.0, 60.0), "duration": float("nan")},
            {**_note(0.0, 1.0, 60), "duration": 1.0},
            {**_note(1.0, 3.0, 62), "duration": 2.0},
        ]
    )
    out = to_absolute_time(df)
    notes = out[out["type"] == "note"].reset_index(drop=True)
    # slope = 2 (60 bpm), so new durations should be 2 and 4.
    assert np.allclose(notes["duration"], [2.0, 4.0])
    assert np.allclose(notes["duration"], notes["release"] - notes["onset"])


def test_first_tempo_after_zero_keeps_opening_at_120():
    # Opening region [0, 2) has no tempo event → defaults to 120.
    # Then 60 bpm takes over. The two opening notes should be unchanged,
    # and the 2nd-segment note should scale by 2.
    df = pd.DataFrame(
        [
            _note(0.0, 1.0, 60),
            _note(1.0, 2.0, 62),
            _tempo(2.0, 60.0),
            _note(2.0, 3.0, 64),
        ]
    )
    out = to_absolute_time(df)
    notes = out[out["type"] == "note"].reset_index(drop=True)
    assert np.allclose(notes["onset"], [0.0, 1.0, 2.0])
    assert np.allclose(notes["release"], [1.0, 2.0, 4.0])


def test_registered_via_apply_transforms():
    df = pd.DataFrame([_tempo(0.0, 90.0), _note(0.0, 1.0, 60)])
    out = apply_transforms(df, [{"to_absolute_time": {}}])
    note = out[out["type"] == "note"].iloc[0]
    assert np.isclose(note["release"], TARGET_BPM / 90.0)


def test_input_df_not_mutated():
    df = pd.DataFrame([_tempo(0.0, 60.0), _note(0.0, 1.0, 60), _note(1.0, 2.0, 62)])
    snapshot = df.copy(deep=True)
    _ = to_absolute_time(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_round_trip_simple_90_bpm():
    df = pd.DataFrame([_tempo(0.0, 90.0), _note(0.0, 1.0, 60), _note(1.0, 2.0, 62)])
    restored = from_absolute_time(to_absolute_time(df))
    notes_in = df[df["type"] == "note"].reset_index(drop=True)
    notes_out = restored[restored["type"] == "note"].reset_index(drop=True)
    np.testing.assert_allclose(notes_out["onset"], notes_in["onset"])
    np.testing.assert_allclose(notes_out["release"], notes_in["release"])
    tempos_out = restored[restored["type"] == "tempo"].reset_index(drop=True)
    assert len(tempos_out) == 1
    assert tempos_out.iloc[0]["onset"] == 0.0
    assert tempos_out.iloc[0]["other"] == {"tempo": 90.0}
    assert "absolute_time" not in restored.attrs
    assert "to_absolute_time_metadata" not in restored.attrs


def test_round_trip_multi_segment():
    df = pd.DataFrame(
        [
            _tempo(0.0, 120.0),
            _tempo(2.0, 60.0),
            _tempo(4.0, 240.0),
            _note(0.0, 2.0, 60),
            _note(2.0, 4.0, 62),
            _note(4.0, 6.0, 64),
            _note(4.5, 5.5, 65),
        ]
    )
    restored = from_absolute_time(to_absolute_time(df))
    notes_in = df[df["type"] == "note"].reset_index(drop=True)
    notes_out = restored[restored["type"] == "note"].reset_index(drop=True)
    np.testing.assert_allclose(notes_out["onset"], notes_in["onset"])
    np.testing.assert_allclose(notes_out["release"], notes_in["release"])
    tempos_out = restored[restored["type"] == "tempo"].reset_index(drop=True)
    assert list(tempos_out["onset"]) == [0.0, 2.0, 4.0]
    assert [t["tempo"] for t in tempos_out["other"]] == [120.0, 60.0, 240.0]


def test_round_trip_preserves_time_signature_rows():
    df = pd.DataFrame(
        [
            _tempo(0.0, 90.0),
            _time_sig(0.0, 3, 4),
            _note(0.0, 1.0, 60),
            _time_sig(3.0, 6, 8),
            _note(3.0, 4.0, 62),
        ]
    )
    abs_df = to_absolute_time(df)
    # Forward must drop the time_signature rows from the output frame.
    assert (abs_df["type"] == "time_signature").sum() == 0

    restored = from_absolute_time(abs_df)
    ts_out = restored[restored["type"] == "time_signature"].reset_index(drop=True)
    assert list(ts_out["onset"]) == [0.0, 3.0]
    assert ts_out.iloc[0]["other"] == {"numerator": 3, "denominator": 4}
    assert ts_out.iloc[1]["other"] == {"numerator": 6, "denominator": 8}


def test_round_trip_preserves_bar_rows():
    df = pd.DataFrame(
        [
            _tempo(0.0, 60.0),
            _bar(0.0, 4.0),
            _note(0.0, 4.0, 60),
            _bar(4.0, 8.0),
        ]
    )
    restored = from_absolute_time(to_absolute_time(df))
    bars = restored[restored["type"] == "bar"].reset_index(drop=True)
    assert list(bars["onset"]) == [0.0, 4.0]
    assert list(bars["release"]) == [4.0, 8.0]


def test_round_trip_no_tempo_rows():
    df = pd.DataFrame(
        [
            _time_sig(0.0, 4, 4),
            _note(0.0, 1.0, 60),
            _note(1.0, 2.0, 62),
        ]
    )
    restored = from_absolute_time(to_absolute_time(df))
    notes_in = df[df["type"] == "note"].reset_index(drop=True)
    notes_out = restored[restored["type"] == "note"].reset_index(drop=True)
    np.testing.assert_allclose(notes_out["onset"], notes_in["onset"])
    np.testing.assert_allclose(notes_out["release"], notes_in["release"])
    ts_out = restored[restored["type"] == "time_signature"].reset_index(drop=True)
    assert len(ts_out) == 1
    assert ts_out.iloc[0]["other"] == {"numerator": 4, "denominator": 4}
    assert (restored["type"] == "tempo").sum() == 0


def test_from_absolute_time_errors_without_stash():
    df = pd.DataFrame([_tempo(0.0, 90.0), _note(0.0, 1.0, 60)])
    import pytest

    with pytest.raises(ValueError, match="absolute_time"):
        from_absolute_time(df)


def test_to_absolute_time_is_idempotent():
    df = pd.DataFrame(
        [_tempo(0.0, 90.0), _note(0.0, 1.0, 60), _note(1.0, 2.0, 62)]
    )
    once = to_absolute_time(df)
    twice = to_absolute_time(once)
    pd.testing.assert_frame_equal(once, twice)
    # The second call must not have wiped the stash.
    assert twice.attrs["to_absolute_time_metadata"]["tempos"]
    assert twice.attrs["absolute_time"] is True


def test_forward_drops_time_signature_rows():
    df = pd.DataFrame(
        [
            _tempo(0.0, 60.0),
            _time_sig(0.0, 3, 4),
            _note(0.0, 1.0, 60),
        ]
    )
    out = to_absolute_time(df)
    assert (out["type"] == "time_signature").sum() == 0
    # And the stash should hold the original.
    stashed = out.attrs["to_absolute_time_metadata"]["time_signatures"]
    assert len(stashed) == 1
    assert stashed[0]["other"] == {"numerator": 3, "denominator": 4}


def test_from_absolute_time_via_apply_transforms():
    df = pd.DataFrame([_tempo(0.0, 90.0), _note(0.0, 1.0, 60)])
    abs_df = apply_transforms(df, [{"to_absolute_time": {}}])
    restored = apply_transforms(abs_df, [{"from_absolute_time": {}}])
    notes_out = restored[restored["type"] == "note"].reset_index(drop=True)
    np.testing.assert_allclose(notes_out["onset"], [0.0])
    np.testing.assert_allclose(notes_out["release"], [1.0])
