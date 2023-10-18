import pandas as pd

from music_df.sort_df import sort_df


def _remove_duplicates_of_type(df, row_type: str) -> pd.DataFrame:
    mask = df["type"] == row_type

    # Drop duplicates from the subset of rows with type. The '~' operator negates the
    # mask, so we concatenate the non-type rows with the de-duplicated type rows
    return pd.concat([df[~mask], df[mask].drop_duplicates()], ignore_index=True)


def merge_df(*dfs, remove_duplicates=True) -> pd.DataFrame:
    """
    >>> nan = float("nan")  # Alias to simplify below assignments
    >>> df1 = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, 60, nan, 62],
    ...         "onset": [0, 0, 4, 4],
    ...         "release": [4, 4, 8, 5],
    ...         "type": ["bar", "note", "bar", "note"],
    ...         "other": [nan, nan, nan, nan],
    ...     }
    ... )
    >>> df2 = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, 64, nan, 69],
    ...         "onset": [0, 0, 4, 4],
    ...         "release": [4, 4, 8, 8],
    ...         "type": ["bar", "note", "bar", "note"],
    ...         "other": [nan, nan, nan, nan],
    ...     }
    ... )
    >>> merge_df(df1, df2)
       pitch  onset  release  type  other
    0    NaN      0        4   bar    NaN
    1   60.0      0        4  note    NaN
    2   64.0      0        4  note    NaN
    3    NaN      4        8   bar    NaN
    4   62.0      4        5  note    NaN
    5   69.0      4        8  note    NaN
    """
    df = pd.concat(dfs, axis=0)
    if remove_duplicates:
        df = _remove_duplicates_of_type(df, "bar")
        df = _remove_duplicates_of_type(df, "time_signature")
    df = sort_df(df)
    return df
