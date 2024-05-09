import typing as t
from math import isnan

import pandas as pd
from mspell import GroupSpeller
from mspell.spelling_funcs import shell_spelling_to_humdrum_spelling


def humdrum_spelling_from_row(row):
    if isinstance(row.spelling, float) and isnan(row.spelling):
        return row.spelling
    assert isinstance(row.spelling, str)
    return shell_spelling_to_humdrum_spelling(row.spelling, int(row.pitch))


def spell_df(df: pd.DataFrame, speller: t.Optional[GroupSpeller] = None, chunk_len=32):
    """
    Keyword args:
        chunk_len: "chunk" length of df to spell at once. If we spell the whole
            dataframe at once, it is liable to contain all 12 pitches which
            tends to lead to nonsensical spellings. Thus we take contiguous
            "chunks" from the dataframe and spell them one by one. This
            parameter controls the length of each chunk.
    """
    df = df.copy()
    if speller is None:
        speller = GroupSpeller(pitches=True, letter_format="kern")
    df["humdrum_spelling"] = ""
    if "spelling" in df.columns:
        df["humdrum_spelling"] = df.apply(humdrum_spelling_from_row, axis=1)
    else:
        for i in range(0, len(df), chunk_len):
            # To avoid chained indexing as follows we use `row_mask`
            # df.iloc[i : i + chunk_len].loc[df.type == "note", "humdrum_spelling"] = (
            row_mask = (
                (df.index >= i) & (df.index < (i + chunk_len)) & (df.type == "note")
            )
            df.loc[row_mask, "humdrum_spelling"] = speller(
                # df.iloc[i : i + chunk_len]
                # .loc[df.type == "note", "pitch"]
                df.loc[row_mask, "pitch"].astype(int)  # type:ignore
            )
    return df
