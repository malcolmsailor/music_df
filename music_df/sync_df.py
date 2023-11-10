from typing import Callable

import numpy as np
import pandas as pd


def sync_array_by_df(
    a: np.ndarray,
    music_df: pd.DataFrame,
    sync_col_name_or_names: str | list[str],
    transform: str | Callable = "mean",
) -> np.ndarray:
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62, 63],
    ...         "onset": [0.0, 0.0, 1.0, 1.0],
    ...         "release": [0.5, 1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> a = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, -1.0], [3.0, -2.0]])
    >>> sync_array_by_df(a, df, "onset")
    array([[ 0.5,  0.5],
           [ 0.5,  0.5],
           [ 2.5, -1.5],
           [ 2.5, -1.5]])
    >>> sync_array_by_df(a, df, ["onset", "release"])
    array([[ 0. ,  1. ],
           [ 1. ,  0. ],
           [ 2.5, -1.5],
           [ 2.5, -1.5]])
    """
    output = np.empty_like(a)
    grouped = music_df.groupby(sync_col_name_or_names).groups
    if isinstance(transform, str):
        transform = getattr(np, transform)
    assert not isinstance(transform, str)
    for group_indices in grouped.values():
        output[group_indices] = transform(a[group_indices], axis=0)  # type:ignore
    return output


def sync_df(
    music_df: pd.DataFrame,
    sync_col_name_or_names: str | list[str],
    val_col_name: str,
    transform: str | Callable = "mean",
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62, 63],
    ...         "onset": [0.0, 0.0, 1.0, 1.0],
    ...         "release": [0.5, 1.0, 2.0, 2.0],
    ...         "val": [1.0, 2.0, 3.0, 4.0],
    ...     }
    ... )
    >>> sync_df(df, "onset", "val")
       pitch  onset  release  val
    0     60    0.0      0.5  1.5
    1     61    0.0      1.0  1.5
    2     62    1.0      2.0  3.5
    3     63    1.0      2.0  3.5
    >>> sync_df(df, ["onset", "release"], "val")
       pitch  onset  release  val
    0     60    0.0      0.5  1.0
    1     61    0.0      1.0  2.0
    2     62    1.0      2.0  3.5
    3     63    1.0      2.0  3.5
    """
    music_df = music_df.copy()
    music_df[val_col_name] = music_df.groupby(sync_col_name_or_names)[
        val_col_name
    ].transform(transform)
    return music_df


def sync_by_onset(
    music_df: pd.DataFrame, col_name: str, transform: str | Callable = "mean"
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 61, 62, 63],
    ...         "onset": [0.0, 0.0, 1.0, 1.0],
    ...         "release": [0.5, 1.0, 2.0, 3.0],
    ...         "val": [1.0, 2.0, 3.0, 4.0],
    ...     }
    ... )
    >>> sync_by_onset(df, "val")
       pitch  onset  release  val
    0     60    0.0      0.5  1.5
    1     61    0.0      1.0  1.5
    2     62    1.0      2.0  3.5
    3     63    1.0      3.0  3.5
    """
    return sync_df(music_df, "onset", col_name, transform)
