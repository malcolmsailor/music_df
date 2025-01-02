import math
import typing as t
from types import MappingProxyType

import numpy as np
import numpy.typing as npt
import pandas as pd

from music_df.utils.search import get_item_leq

DF_TYPE_SORT_ORDER = MappingProxyType({"bar": 0, "time_signature": 1, "note": 2})


def get_eligible_onsets(
    df: pd.DataFrame,
    keep_onsets_together: bool = True,
    notes_only: bool = False,
) -> npt.NDArray[np.int_]:
    """
    This function should perhaps be renamed "get indices to eligible onsets".
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [0, 60, 64, 60, 64, 0, 60, 64, 60, 64, 0],
    ...         "onset": [0, 0, 0, 1, 1, 1.5, 1.5, 2.0, 3.0, 3.0, 5.0],
    ...         "release": [0, 1, 1, 1.5, 2.0, 0, 3.0, 3.0, 4.0, 4.5, 0],
    ...         "type": ["bar"] + ["note"] * 4 + ["bar"] + ["note"] * 4 + ["bar"],
    ...     }
    ... )
    >>> df
        pitch  onset  release  type
    0       0    0.0      0.0   bar
    1      60    0.0      1.0  note
    2      64    0.0      1.0  note
    3      60    1.0      1.5  note
    4      64    1.0      2.0  note
    5       0    1.5      0.0   bar
    6      60    1.5      3.0  note
    7      64    2.0      3.0  note
    8      60    3.0      4.0  note
    9      64    3.0      4.5  note
    10      0    5.0      0.0   bar
    >>> get_eligible_onsets(df)
    array([ 0,  3,  5,  7,  8, 10])
    >>> get_eligible_onsets(df, notes_only=True)
    array([1, 3, 6, 7, 8])
    >>> get_eligible_onsets(df, keep_onsets_together=False)
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
    >>> get_eligible_onsets(df, keep_onsets_together=False, notes_only=True)
    array([1, 2, 3, 4, 6, 7, 8, 9])
    """
    if notes_only and "type" in df.columns:
        df = df[df.type == "note"]
    if not keep_onsets_together:
        return df.index.to_numpy()
    onset_indices = np.unique(df.onset, return_index=True)[1]
    return df.index[onset_indices].to_numpy()


def get_eligible_releases(
    df: pd.DataFrame,
    keep_releases_together: bool = True,
) -> pd.Series:
    """
    Returns a series where the Index gives the indices into the dataframe
    and the values are the associated release times.
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [0, 60, 64, 60, 64, 0, 60, 64, 60, 64, 0],
    ...         "onset": [0, 0, 0, 1, 1, 1.5, 1.5, 2.0, 3.0, 3.0, 5.0],
    ...         "release": [0, 1, 1, 1.5, 2.0, 0, 3.0, 3.0, 4.0, 4.5, 0],
    ...         "type": ["bar"] + ["note"] * 4 + ["bar"] + ["note"] * 4 + ["bar"],
    ...     }
    ... )
    >>> df
        pitch  onset  release  type
    0       0    0.0      0.0   bar
    1      60    0.0      1.0  note
    2      64    0.0      1.0  note
    3      60    1.0      1.5  note
    4      64    1.0      2.0  note
    5       0    1.5      0.0   bar
    6      60    1.5      3.0  note
    7      64    2.0      3.0  note
    8      60    3.0      4.0  note
    9      64    3.0      4.5  note
    10      0    5.0      0.0   bar

    Only notes have releases so get_eligible_releases() is always `note_only`.
    (Cf get_eligible_onsets().)

    >>> get_eligible_releases(df)
    2    1.0
    3    1.5
    4    2.0
    7    3.0
    8    4.0
    9    4.5
    Name: release, dtype: float64
    >>> get_eligible_releases(df, keep_releases_together=False)
    1    1.0
    2    1.0
    3    1.5
    4    2.0
    6    3.0
    7    3.0
    8    4.0
    9    4.5
    Name: release, dtype: float64
    """
    if "type" in df.columns:
        df = df[df.type == "note"]
    if not keep_releases_together:
        return df.release
    df2 = df.sort_values(
        by="pitch", inplace=False, ignore_index=False, kind="mergesort"
    )
    df2 = df2.sort_values(
        by="release", inplace=False, ignore_index=False, kind="mergesort"
    )
    release_indices = (len(df2) - 1) - np.unique(
        np.flip(df2.release.to_numpy()), return_index=True
    )[1]
    out = df2.iloc[release_indices]["release"]
    return out


def get_df_segment_indices(
    eligible_onsets: t.Union[t.Sequence[int], npt.NDArray[np.int_]],
    eligible_releases: t.Union[t.Sequence[int], npt.NDArray[np.int_]],
    target_len: int,
):
    """
    # >>> eligible_onsets = list(range(32))
    # >>> eligible_releases = list(range(32))
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 8), (8, 16), (16, 24), (24, 32)]

    # >>> eligible_onsets = [i * 2 for i in range(16)]
    # >>> eligible_releases = [i * 2 + 1 for i in range(16)]
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 8), (8, 16), (16, 24), (24, 32)]

    # >>> eligible_onsets = [0, 3, 7, 14]
    # >>> eligible_releases = [2, 3, 6, 12, 13, 17]
    # >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    # [(0, 7), (7, 14), (14, 18)]

    We aim for target_len, but there is no firm limit on how long a segment
    might be. We depend on eligible_onsets/eligible_releases to be fairly
    evenly distributed to avoid segments that are far too long (or short).
    >>> eligible_onsets = [0, 1, 14, 15]
    >>> eligible_releases = [2, 3, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(0, 4), (1, 18)]

    >>> eligible_onsets = [0, 1, 14, 15]
    >>> eligible_releases = [16, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(0, 17), (15, 18)]

    Releases before the first eligible onset are ignored.
    >>> eligible_onsets = [14, 15]
    >>> eligible_releases = [0, 1, 16, 17]
    >>> list(get_df_segment_indices(eligible_onsets, eligible_releases, 8))
    [(14, 18)]

    There shouldn't be any other circumstance in which onsets or releases are
    skipped.
    """
    # assumes df has a range index
    start_i = None
    end_i = eligible_releases[0] - 1
    max_release_i = eligible_releases[-1]
    while end_i < max_release_i:
        if start_i is None:
            start_i = eligible_onsets[0]
        else:
            try:
                start_i = get_item_leq(eligible_onsets, end_i + 1, min_val=start_i + 1)
            except ValueError:  # pylint: disable=try-except-raise
                # We should never get here, I think this is a bug if we do
                raise
        # we calculate end_i *inclusively*, then add 1 to it to return
        #   an exclusive boundary appropriate for slicing in Python
        end_i = get_item_leq(
            eligible_releases,
            # We need to subtract 1 from target_len because we are
            #   calculating an inclusive boundary
            start_i + target_len - 1,
            min_val=max(start_i + 1, end_i + 1),
        )
        yield start_i, end_i + 1


def segment_df(df: pd.DataFrame, target_len):
    """
    This function segments dataframes such that they contain a certain target
    number of notes (or rows? not sure).
    """
    eligible_onsets = get_eligible_onsets(df)
    eligible_releases = get_eligible_releases(df).index.to_numpy()
    for start_i, end_i in get_df_segment_indices(
        eligible_onsets, eligible_releases, target_len
    ):
        yield df[start_i:end_i]


def get_notes_sounding_during(
    df: pd.DataFrame, onset: float, release: float
) -> pd.DataFrame:
    df = df[df.type == "note"]
    df = df[df.onset < release]
    df = df[df.release > onset]
    return df


def segment_df_by_onset(df: pd.DataFrame, segment_dur: float):
    """
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [0, 60, 64, 60, 64, 0, 60, 64, 60, 64, 0],
    ...         "onset": [0, 0, 0, 1, 1, 1.5, 1.5, 2.0, 3.0, 3.0, 5.0],
    ...         "release": [0, 1, 1, 1.5, 2.0, 0, 3.0, 3.0, 4.0, 4.5, 0],
    ...         "type": ["bar"] + ["note"] * 4 + ["bar"] + ["note"] * 4 + ["bar"],
    ...     }
    ... )
    >>> df
        pitch  onset  release  type
    0       0    0.0      0.0   bar
    1      60    0.0      1.0  note
    2      64    0.0      1.0  note
    3      60    1.0      1.5  note
    4      64    1.0      2.0  note
    5       0    1.5      0.0   bar
    6      60    1.5      3.0  note
    7      64    2.0      3.0  note
    8      60    3.0      4.0  note
    9      64    3.0      4.5  note
    10      0    5.0      0.0   bar

    >>> for (name, group) in segment_df_by_onset(df, 2.0):
    ...     print(group)
    ...
       pitch  onset  release  type  group
    0      0    0.0      0.0   bar    0.0
    1     60    0.0      1.0  note    0.0
    2     64    0.0      1.0  note    0.0
    3     60    1.0      1.5  note    0.0
    4     64    1.0      2.0  note    0.0
    5      0    1.5      0.0   bar    0.0
    6     60    1.5      3.0  note    0.0
       pitch  onset  release  type  group
    7     64    2.0      3.0  note    1.0
    8     60    3.0      4.0  note    1.0
    9     64    3.0      4.5  note    1.0
        pitch  onset  release type  group
    10      0    5.0      0.0  bar    2.0
    """
    df["group"] = np.floor(df["onset"] / segment_dur)
    return df.groupby("group")
