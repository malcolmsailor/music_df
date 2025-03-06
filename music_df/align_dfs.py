import pandas as pd


def _split_events_helper(df, unique_times):
    new_rows = []
    unique_time_iter = iter(unique_times)
    next_time = next(unique_time_iter)

    for _, row in df.iterrows():
        onset = row["onset"]
        while onset >= next_time:
            next_time = next(unique_time_iter)
        while row["release"] > next_time:
            new_rows.append({**row.to_dict(), "onset": onset, "release": next_time})
            onset = next_time
            try:
                next_time = next(unique_time_iter)
            except StopIteration:
                raise NotImplementedError

        new_rows.append({**row.to_dict(), "onset": onset})

    return pd.DataFrame(new_rows)


def align_onsets_between_dataframes(*dfs) -> list[pd.DataFrame]:
    """
    >>> df1 = pd.DataFrame(
    ...     {"onset": [1, 3, 5], "release": [3, 5, 7], "attribute": ["A", "B", "C"]}
    ... )
    >>> df2 = pd.DataFrame(
    ...     {"onset": [1, 2, 4], "release": [2, 4, 7], "attribute": ["D", "E", "F"]}
    ... )
    >>> aligned_df1, aligned_df2 = align_onsets_between_dataframes(df1, df2)
    >>> aligned_df1
       onset  release attribute
    0      1        2         A
    1      2        3         A
    2      3        4         B
    3      4        5         B
    4      5        7         C
    >>> aligned_df2
       onset  release attribute
    0      1        2         D
    1      2        3         E
    2      3        4         E
    3      4        5         F
    4      5        7         F
    """
    # unique_times = sorted(set(df1['onset'].tolist() + df1['release'].tolist() + df2['onset'].tolist() + df2['release'].tolist()))
    try:
        for df in dfs:
            assert (df.onset.sort_values().values == df.onset.values).all()
            assert (df.release.sort_values().values == df.release.values).all()
    except AssertionError:
        print("Onsets/releases of all dataframes must be sorted")

    try:
        for df in dfs:
            assert (df["onset"].iloc[1:].values >= df["release"].iloc[:-1].values).all()
    except AssertionError:
        raise NotImplementedError(
            "All onsets must be greater or equal than the previous release"
        )
    unique_times = set()
    for df in dfs:
        unique_times |= set(df["onset"].tolist())
        unique_times |= set(df["release"].tolist())

    unique_times = sorted(unique_times)

    out_dfs = []
    for df in dfs:
        out_dfs.append(_split_events_helper(df, unique_times).reset_index(drop=True))

    return out_dfs
