import numpy as np
import pandas as pd


def quantize_df(
    df, tpq: int = 4, ticks_out: bool = False, avoid_zero_dur_notes: bool = True
) -> pd.DataFrame:
    """
    >>> df = pd.DataFrame({
    ...     "pitch": [60, 61, 62, 63],
    ...     "onset": [-0.01, 1.01, 1.95, 2.9],
    ...     "release": [0.99, 2.03, 3.0, 3.97],
    ... })
    >>> df
       pitch  onset  release
    0     60  -0.01     0.99
    1     61   1.01     2.03
    2     62   1.95     3.00
    3     63   2.90     3.97

    There may be a negative zero in the output:
    >>> quantize_df(df, tpq=4)
       pitch  onset  release
    0     60   -0.0      1.0
    1     61    1.0      2.0
    2     62    2.0      3.0
    3     63    3.0      4.0

    >>> quantize_df(df, tpq=16)
       pitch   onset  release
    0     60 -0.0000      1.0
    1     61  1.0000      2.0
    2     62  1.9375      3.0
    3     63  2.8750      4.0

    >>> quantize_df(df, tpq=16, ticks_out=True)
       pitch  onset  release
    0     60      0       16
    1     61     16       32
    2     62     31       48
    3     63     46       64

    Note that by default, notes that would be rounded to have zero length
    are given the minimum length.
    >>> df = pd.DataFrame({
    ...     "pitch": [60, 61, 62],
    ...     "onset": [0.0, 0.5, 1.0],
    ...     "release": [0.4, 1.0, 2.0],
    ... })
    >>> quantize_df(df, tpq=1)
       pitch  onset  release
    0     60    0.0      1.0
    1     61    0.0      1.0
    2     62    1.0      2.0

    To avoid that, pass `avoid_zero_dur_notes=False`:
    >>> quantize_df(df, tpq=1, avoid_zero_dur_notes=False)
       pitch  onset  release
    0     60    0.0      0.0
    1     61    0.0      1.0
    2     62    1.0      2.0

    If you want to remove notes that have length < some threshold, do it to
    the dataframe before calling this function:
    >>> quantize_df(df[(df.release - df.onset) >= 0.5], tpq=1)
       pitch  onset  release
    1     61    0.0      1.0
    2     62    1.0      2.0
    """
    onsets = np.rint(df.onset.to_numpy() * tpq)
    releases = np.rint(df.release.to_numpy() * tpq)
    if avoid_zero_dur_notes:
        releases[releases == onsets] += 1
    if ticks_out:
        onsets = onsets.astype(int)
        releases = releases.astype(int)
    else:
        onsets /= tpq
        releases /= tpq
    return pd.DataFrame(
        {
            col_name: (
                df[col_name].copy()
                if col_name not in ("onset", "release")
                else {"onset": onsets, "release": releases}[col_name]
            )
            for col_name in df.columns
        }
    )
