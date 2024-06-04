import pandas as pd
import io  # For doctest

from music_df.salami_slice import appears_salami_sliced


def return_rests(
    df: pd.DataFrame, epsilon: float = 1e-7, check_salami_sliced: bool = True
):
    """
    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,0.5
    ... note,60,2.0,3.0
    ... bar,,4.0,8.0
    ... note,60,4.75,5.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release
    0   bar    NaN   0.00      4.0
    1  note   60.0   0.00      0.5
    2  note   60.0   2.00      3.0
    3   bar    NaN   4.00      8.0
    4  note   60.0   4.75      5.0
    >>> return_rests(df)
    [(0.5, 2.0), (3.0, 4.75)]
    """
    if check_salami_sliced and not appears_salami_sliced(df):
        raise NotImplementedError
    # Initialize an empty list to store the gaps
    gaps = []

    df = df[df.type == "note"]
    for (_, prev_row), (_, next_row) in zip(df.iterrows(), df.iloc[1:].iterrows()):

        # If the next onset is greater than the current release, there is a gap
        gap = next_row["onset"] - prev_row["release"]
        if gap > epsilon:
            # Add the gap to the list
            gaps.append((prev_row["release"], next_row["onset"]))

    # Return the list of gaps
    return gaps
