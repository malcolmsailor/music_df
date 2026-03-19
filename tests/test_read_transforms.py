from pathlib import Path

import pandas as pd

from music_df.read import read

MIDI_PATH = str(
    Path(__file__).parent / "midi_parser_tests" / "test_files" / "misc_Palestrina.mid"
)
TPQ = 16


def test_read_without_transforms_unchanged():
    """Without transforms, read() behaves as before."""
    df = read(MIDI_PATH)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_read_with_transforms_applies_pipeline():
    df = read(MIDI_PATH, transforms=[{"quantize_df": {"tpq": TPQ}}])
    notes = df[df["type"] == "note"]

    # Onsets should be quantized to multiples of 1/TPQ
    remainders = (notes["onset"] * TPQ).round(6) % 1
    assert (remainders < 1e-9).all(), "Onsets are not quantized"

    # sort_df should have been applied (onsets non-decreasing)
    assert (notes["onset"].diff().dropna() >= -1e-9).all(), "Notes are not sorted"


def test_read_with_empty_transforms_returns_sorted():
    """An empty transforms list still triggers sort_df."""
    df = read(MIDI_PATH, transforms=[])
    notes = df[df["type"] == "note"]
    assert (notes["onset"].diff().dropna() >= -1e-9).all()
