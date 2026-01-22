import io  # noqa: F401
from typing import Iterable

import pandas as pd

from music_df.sort_df import sort_df


def slice_df(
    df: pd.DataFrame,
    slice_boundaries: Iterable[float],
) -> pd.DataFrame:
    """
    Slice any notes that overlap a slice boundary.

    Args:
        df: The DataFrame to slice.
        slice_boundaries: The boundaries of the slices.


    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,2.0
    ... note,64,2.0,3.0
    ... note,67,3.0,4.0
    ... '''
    ...     )
    ... )
    >>> slice_df(df, [0.0, 1.0, 3.0])
       type  pitch  onset  release  sliced
    0   bar    NaN    0.0      4.0      no
    1  note   60.0    0.0      1.0  before
    2  note   60.0    1.0      2.0   after
    3  note   64.0    2.0      3.0      no
    4  note   67.0    3.0      4.0      no

    >>> slice_df(df, [0.0])
       type  pitch  onset  release sliced
    0   bar    NaN    0.0      4.0     no
    1  note   60.0    0.0      2.0     no
    2  note   64.0    2.0      3.0     no
    3  note   67.0    3.0      4.0     no

    >>> slice_df(df, [-10.0, 10.0])
       type  pitch  onset  release sliced
    0   bar    NaN    0.0      4.0     no
    1  note   60.0    0.0      2.0     no
    2  note   64.0    2.0      3.0     no
    3  note   67.0    3.0      4.0     no

    """

    overlapping_indices = []
    new_notes = []

    for slice_boundary in slice_boundaries:
        overlapping_notes = df.loc[
            (df.onset < slice_boundary)
            & (df.release > slice_boundary)
            & (df.type == "note")
        ]
        if overlapping_notes.empty:
            continue

        overlapping_indices.extend(overlapping_notes.index)

        before_slice_notes = overlapping_notes.copy()
        before_slice_notes["release"] = slice_boundary
        before_slice_notes["sliced"] = "before"

        after_slice_notes = overlapping_notes.copy()
        after_slice_notes["onset"] = slice_boundary
        after_slice_notes["sliced"] = "after"

        new_notes.append(before_slice_notes)
        new_notes.append(after_slice_notes)

    new_df = df.loc[~df.index.isin(overlapping_indices)].copy()
    new_df["sliced"] = "no"
    new_df = pd.concat([new_df, *new_notes])
    new_df["sliced"] = new_df["sliced"].astype(
        pd.CategoricalDtype(categories=["no", "before", "after"])
    )
    new_df = sort_df(new_df)

    return new_df
