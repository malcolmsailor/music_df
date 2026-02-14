import pandas as pd
from music_df.conversions.ms3 import _add_time_sigs, ms3_to_df


def test_ms3_to_df():
    # (Malcolm 2023-10-11) This test currently does nothing except check that
    #   the conversion runs without an exception
    # TODO: (Malcolm 2023-10-11) update the path
    TEST_PATH = "/Users/malcolm/google_drive/python/third_party/dcml_corpora/corelli/notes/op03n10d.tsv"
    TEST_MEASURES_PATH = "/Users/malcolm/google_drive/python/third_party/dcml_corpora/corelli/measures/op03n10d.tsv"
    ms3_df = pd.read_csv(TEST_PATH, sep="\t")
    measures_df = pd.read_csv(TEST_MEASURES_PATH, sep="\t")
    music_df = ms3_to_df(ms3_df, measures_df)


def _make_note_df(onsets, timesig="3/4"):
    """Helper to build a minimal note DataFrame for _add_time_sigs tests."""
    return pd.DataFrame(
        {
            "type": "note",
            "onset": onsets,
            "timesig": timesig,
        }
    )


def test_add_time_sigs_with_leading_rest():
    """First time sig should be at onset 0 even when the first note comes later."""
    df = _make_note_df([0.5, 1.0, 2.0])
    result = _add_time_sigs(df)
    ts_rows = result[result["type"] == "time_signature"]
    assert len(ts_rows) == 1
    assert ts_rows.iloc[0]["onset"] == 0


def test_add_time_sigs_at_onset_zero():
    """When the first note is already at onset 0, nothing should change."""
    df = _make_note_df([0.0, 1.0, 2.0])
    result = _add_time_sigs(df)
    ts_rows = result[result["type"] == "time_signature"]
    assert len(ts_rows) == 1
    assert ts_rows.iloc[0]["onset"] == 0
