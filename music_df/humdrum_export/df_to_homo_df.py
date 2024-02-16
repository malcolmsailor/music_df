import itertools as it
import typing as t

import pandas as pd

from music_df.sort_df import sort_df


def _merge_nonnotes(note_df: pd.DataFrame, nonnote_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([note_df, nonnote_df])
    return sort_df(df)


def df_to_homo_df(df: pd.DataFrame) -> t.Tuple[pd.DataFrame, ...]:
    """
    Returns a tuple of dataframes where all notes are "homophonic" (meaning each unique
    onset is paired with a unique release and all onsets are >= to the release paired
    with the previous onset).
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [0, 60, 64, 60, 64, 60],
    ...         "onset": [0, 0, 0, 1, 1, 1.5],
    ...         "release": [0, 1, 1, 1.5, 3.0, 3.0],
    ...         "type": ["bar"] + ["note"] * 5,
    ...     }
    ... )
    >>> df
       pitch  onset  release  type
    0      0    0.0      0.0   bar
    1     60    0.0      1.0  note
    2     64    0.0      1.0  note
    3     60    1.0      1.5  note
    4     64    1.0      3.0  note
    5     60    1.5      3.0  note
    >>> df1, df2 = df_to_homo_df(df)
    >>> df1
       pitch  onset  release  type
    0      0    0.0      0.0   bar
    1     60    0.0      1.0  note
    2     64    0.0      1.0  note
    3     60    1.0      1.5  note
    4     60    1.5      3.0  note
    >>> df2
       pitch  onset  release  type
    0      0    0.0      0.0   bar
    1     64    1.0      3.0  note
    """
    out = []
    note_mask = df.type == "note"
    i = None
    for _, note in df[note_mask].iterrows():
        for i in it.count():
            if i == len(out):
                out.append([])
                break
            prev_note = out[i][-1]
            if note.onset >= prev_note.release or (
                note.onset == prev_note.onset and note.release == prev_note.release
            ):
                break
        assert i is not None
        out[i].append(note)
    nonnote_df = df[~note_mask]
    return tuple(_merge_nonnotes(pd.DataFrame(x), nonnote_df) for x in out)
