"""
Provides a function for searching salami-slices.
"""

from typing import Any, Iterable

import pandas as pd

# def search1(vals):
#     if not vals:
#         return True
#     vals = set(vals)
#     for x in df.data:
#         if x in vals:
#             vals.remove(x)
#         if not vals:
#             break
#     return not bool(vals)


# def search2(vals):
#     unique_vals = df.data.unique()
#     return all(v in unique_vals for v in vals)


# def search3(vals):
#     return sum((df.data == i).any() for i in vals) == len(vals)


# def search4(vals):
#     for v in vals:
#         if not df.data.isin([v]).any():
#             return False
#     return True


# def search5(vals):
#     mask = df.data.isin(vals)
#     return all(v in df.data[mask].unique() for v in vals)


def all_vals_in_series(vals: Iterable[Any], series: pd.Series) -> bool:
    # I did a little bit of testing with the functions above and this option seemed
    #   to be the fastest. However my test data was not very realistic. (In particular,
    #   the series were pretty long, whereas in my use case they'll probably rarely
    #   be more than 10 items.)
    unique_vals = series.unique()
    return all(v in unique_vals for v in vals)


def find_simultaneous_feature(
    music_df: pd.DataFrame, feature_name: str, values: Iterable[Any]
) -> list[tuple[float, float]]:
    """
    Find salami-slices that contain all of the specified values.

    For example, if feature_name is "scale_degree", and values contains a list of
    degrees like ["1", "#1"], then this function will return the onset and release times
    of any salami-slices that contain both "1" and "#1" (they may contain other
    degrees as well).

    The dataframe is assumed to be salami-sliced.

    Args:
        music_df: dataframe.
        feature_name: name of the feature to search for.
        values: list of values to search for.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [60, 67, 61, 72],
    ...         "onset": [0.0, 0, 4, 4],
    ...         "release": [4.0, 4, 8, 8],
    ...         "scale_degree": ["1", "5", "#1", "1"],
    ...     }
    ... )
    >>> find_simultaneous_feature(df, "scale_degree", ["1", "#1"])
    [(4.0, 8.0)]
    >>> find_simultaneous_feature(df, "scale_degree", ["1", "7"])
    []
    """
    assert "release" in music_df.columns
    assert "onset" in music_df.columns
    out = []
    for time_bounds, simultaneity in music_df.groupby(["onset", "release"]):
        if all_vals_in_series(values, simultaneity[feature_name]):
            out.append(time_bounds)
    return out
