import itertools as it
import typing as t

import pandas as pd

from music_df.sort_df import sort_df


def _merge_nonnotes(note_df: pd.DataFrame, nonnote_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([note_df, nonnote_df])
    return sort_df(df)


def split_df_by_pitch(
    df: pd.DataFrame, split_points: t.Union[t.Sequence[int], int]
) -> t.Tuple[pd.DataFrame, ...]:
    """
    Split point is included in upper df.
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [0, 32, 57, 65, 91],
    ...         "onset": [0, 0, 1, 1, 2],
    ...         "release": [0, 1, 2, 2, 3],
    ...         "type": ["bar"] + ["note"] * 4,
    ...     }
    ... )
    >>> df
       pitch  onset  release  type
    0      0      0        0   bar
    1     32      0        1  note
    2     57      1        2  note
    3     65      1        2  note
    4     91      2        3  note
    >>> df1, df2 = split_df_by_pitch(df, 65)
    >>> df1
       pitch  onset  release  type
    0      0      0        0   bar
    1     32      0        1  note
    2     57      1        2  note
    >>> df2
       pitch  onset  release  type
    0      0      0        0   bar
    1     65      1        2  note
    2     91      2        3  note
    >>> df1, df2, df3 = split_df_by_pitch(df, (60, 65))
    >>> df1
       pitch  onset  release  type
    0      0      0        0   bar
    1     32      0        1  note
    2     57      1        2  note
    >>> df2
       pitch  onset  release type
    0      0      0        0  bar
    >>> df3
       pitch  onset  release  type
    0      0      0        0   bar
    1     65      1        2  note
    2     91      2        3  note
    >>> df1, df2, df3 = split_df_by_pitch(df, (50, 65))
    >>> df1
       pitch  onset  release  type
    0      0      0        0   bar
    1     32      0        1  note
    >>> df2
       pitch  onset  release  type
    0      0      0        0   bar
    1     57      1        2  note
    >>> df3
       pitch  onset  release  type
    0      0      0        0   bar
    1     65      1        2  note
    2     91      2        3  note
    """
    if not split_points:
        return (df,)
    note_mask = df.type == "note"
    note_df = df[note_mask]
    nonnote_df = df[~note_mask]
    out = []
    if isinstance(split_points, int):
        split_points = [split_points]
    for low, hi in zip(
        [float("-inf")] + list(split_points),
        list(split_points) + [float("inf")],
    ):
        out.append(
            _merge_nonnotes(
                note_df[note_df.pitch.between(low, hi, inclusive="left")],
                nonnote_df,
            )
        )
    return tuple(out)
