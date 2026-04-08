import pandas as pd

from music_df.humdrum_export.df_utils.spell_df import spell_df


def test_spell_df_non_contiguous_index():
    """spell_df must spell all notes even when the DataFrame index has gaps.

    Regression test: when rows are filtered out before calling spell_df
    (e.g., percussion removal), the index becomes non-contiguous. The
    chunking loop used range(0, len(df)) which missed notes whose index
    exceeded len(df).
    """
    df = pd.DataFrame(
        {
            "type": ["bar"] + ["note"] * 5,
            "pitch": [float("nan"), 60, 64, 67, 72, 48],
            "onset": [0, 0, 0, 1, 1, 2],
            "release": [float("nan"), 1, 1, 2, 2, 3],
        }
    )
    # Simulate a gap in the index (as from filtering percussion rows)
    df.index = [0, 1, 2, 100, 101, 102]

    result = spell_df(df, chunk_len=4)

    note_spellings = result.loc[result.type == "note", "humdrum_spelling"]
    assert (note_spellings != "").all(), (
        f"Some notes were not spelled: {note_spellings.tolist()}"
    )
