import pandas as pd

from music_df.split_notes import split_notes_at_barlines


def test_split_note_crossing_first_bar_no_preceding_bar():
    """Notes starting before the first bar's onset should be split at that barline."""
    df = pd.DataFrame(
        {
            "type": ["note", "note", "note", "bar", "note"],
            "onset": [0.0, 1.5, 1.75, 1.9375, 2.0],
            "release": [1.5, 1.75, 2.0, 4.9375, 2.3125],
            "pitch": [50, 53, 58, None, 62],
            "tie_to_next": [False] * 5,
            "tie_to_prev": [False] * 5,
        }
    )
    result = split_notes_at_barlines(df, min_overhang_dur=1 / 16)

    note58 = result[result["pitch"] == 58]
    assert len(note58) == 2, (
        f"Expected note 58 to be split into 2 rows, got {len(note58)}"
    )

    before_bar = note58[note58["onset"] == 1.75]
    assert len(before_bar) == 1
    assert before_bar.iloc[0]["release"] == 1.9375
    assert before_bar.iloc[0]["tie_to_next"] == True  # noqa: E712
    assert before_bar.iloc[0]["tie_to_prev"] == False  # noqa: E712

    after_bar = note58[note58["onset"] == 1.9375]
    assert len(after_bar) == 1
    assert after_bar.iloc[0]["release"] == 2.0
    assert after_bar.iloc[0]["tie_to_prev"] == True  # noqa: E712


def test_split_note_normal_case_bar_at_onset_zero():
    """Normal case: bar at onset 0, notes crossing later barlines are split correctly."""
    df = pd.DataFrame(
        {
            "type": ["bar", "note", "bar"],
            "onset": [0.0, 0.0, 4.0],
            "release": [4.0, 6.0, 8.0],
            "pitch": [None, 60, None],
            "tie_to_next": [False] * 3,
            "tie_to_prev": [False] * 3,
        }
    )
    result = split_notes_at_barlines(df)

    notes = result[result["type"] == "note"]
    assert len(notes) == 2

    first = notes[notes["onset"] == 0.0].iloc[0]
    assert first["release"] == 4.0
    assert first["tie_to_next"] == True  # noqa: E712

    second = notes[notes["onset"] == 4.0].iloc[0]
    assert second["release"] == 6.0
    assert second["tie_to_prev"] == True  # noqa: E712


def test_note_starting_at_bar_boundary_not_crossing_barline():
    """A note starting exactly at a bar boundary and ending within that bar
    should NOT be split or given tie flags."""
    df = pd.DataFrame(
        {
            "type": ["bar", "note", "bar", "note", "bar"],
            "onset": [0.0, 0.0, 4.0, 4.0, 8.0],
            "release": [4.0, 3.0, 8.0, 5.0, 12.0],
            "pitch": [None, 60, None, 62, None],
            "tie_to_next": [False] * 5,
            "tie_to_prev": [False] * 5,
        }
    )
    result = split_notes_at_barlines(df)

    notes = result[result["type"] == "note"]
    assert len(notes) == 2, (
        f"Expected 2 notes (no splitting needed), got {len(notes)}"
    )

    note62 = notes[notes["pitch"] == 62].iloc[0]
    assert note62["onset"] == 4.0
    assert note62["release"] == 5.0
    assert note62["tie_to_next"] == False  # noqa: E712
    assert note62["tie_to_prev"] == False  # noqa: E712
