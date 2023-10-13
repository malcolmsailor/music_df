from fractions import Fraction

import pandas as pd

from music_df.add_feature import infer_barlines
from music_df.sort_df import sort_df

NOTE_NAMES = "fcgdaeb".upper()
MINOR_NOTE_NAMES = NOTE_NAMES.lower()


def tpc2name(tpc: int, minor: bool = False) -> str:
    """Turn a tonal pitch class (TPC) into a name or perform the operation on a
    collection of integers.

    >>> tpc2name(-1)
    'F'
    >>> tpc2name(-1, minor=True)
    'f'
    >>> tpc2name(-8, minor=True)
    'fb'
    >>> tpc2name(6)
    'F#'
    >>> tpc2name(13)
    'F##'

    Based on equivalent function from MS3.

    Args:
      tpc: Tonal pitch class(es) to turn into a note name.
      minor: Pass True if the string is to be returned as lowercase.

    Returns:

    """
    note_names = MINOR_NOTE_NAMES if minor else NOTE_NAMES

    acc, ix = divmod(tpc + 1, 7)
    acc_str = abs(acc) * "b" if acc < 0 else acc * "#"
    return f"{note_names[ix]}{acc_str}"


def _str_to_float(s: str) -> float:
    if "/" in s:
        numerator, denominator = map(int, s.split("/"))
        return numerator / denominator
    else:
        return float(s)


def _timesig_str_to_other(s: str) -> dict[str, int]:
    numerator, denominator = map(int, s.split("/"))
    return {"numerator": numerator, "denominator": denominator}


def _add_time_sigs(df: pd.DataFrame) -> pd.DataFrame:
    changes = (df["timesig"] != df["timesig"].shift()) | (df.index == 0)
    df_changes = pd.DataFrame(
        {
            "type": ["time_signature" for _ in df[changes].index],
            "onset": df[changes]["onset"].values,
            "other": df[changes]["timesig"].apply(_timesig_str_to_other),
        }
    )
    df_changes.index = df[changes].index - 0.5

    result = pd.concat([df, df_changes]).sort_index().reset_index(drop=True)

    return result


def _add_bars(df: pd.DataFrame) -> pd.DataFrame:
    # There should be only notes and time signatures in the dataframe
    assert set(df.type.unique()) == {"time_signature", "note"}

    df_copy = df.copy()
    # Because time signatures will have mc NAN, they will compare nonequal to
    #   before and after, meaning barlines will go before and after. This is not
    #   what we want. So instead we need to create a temporary column where we
    #   assign mc of the following bar - 1 to each time signature
    mask = df_copy["mc"].isna()
    df_copy.loc[mask, "mc"] = df_copy["mc"].bfill() - 1

    changes = df_copy["mc"] != df_copy["mc"].shift()

    # don't add a bar before an initial time signature
    if df.iloc[0]["type"] == "time_signature":
        changes.iloc[0] = False

    changes_df = df[changes]

    bar_df = pd.DataFrame(
        {
            "type": ["bar" for _ in changes_df.index],
            "onset": changes_df["onset"].values,
            "release": changes_df["onset"].shift(-1).values,
        }
    )
    bar_df.iloc[-1, -1] = df["release"].max()
    bar_df.index = changes_df.index - 0.5

    result = pd.concat([df, bar_df]).sort_index().reset_index(drop=True)

    return result


DENOM_LIMIT = 64

remap_time_column = lambda x: Fraction(x).limit_denominator(DENOM_LIMIT)


def ms3_to_df(
    ms3_df: pd.DataFrame,
    remove_zero_duration_notes: bool = True,
    drop_first_endings: bool = True,
    fractions: bool = True,
    drop_unused_cols: bool = True,
) -> pd.DataFrame:
    """ """
    out_df = ms3_df.copy()

    if drop_first_endings:
        out_df = out_df[~out_df["quarterbeats"].isna()].reset_index(drop=True)
    else:
        raise NotImplementedError

    # (Malcolm 2023-10-12) Note: the "duration" column is in whole notes,
    #   not quarter notes, which is why we need "duration_qb"
    if fractions:
        out_df["onset"] = out_df["quarterbeats"].map(remap_time_column)
        out_df["release"] = out_df["onset"] + out_df["duration_qb"].map(
            remap_time_column
        )
    else:
        out_df["onset"] = out_df["quarterbeats"].apply(_str_to_float)
        out_df["release"] = out_df["onset"] + out_df["duration_qb"].apply(_str_to_float)

    if "name" in out_df.columns:
        # remove last character of "name" column (A4 -> A, etc.)
        out_df["spelling"] = out_df["name"].str[:-1]
    else:
        # earlier versions of ms3 don't seem to have the name column but
        #   we can get the spelling from the "tpc" column
        out_df["spelling"] = out_df["tpc"].apply(tpc2name)

    # rename some columns
    out_df.rename(columns={"midi": "pitch", "staff": "part"}, inplace=True)

    # add "tie_to_next" boolean column true if "tied" is in (0, 1)
    out_df["tie_to_next"] = out_df["tied"].isin((0, 1))
    out_df["tie_to_prev"] = out_df["tied"].isin((0, -1))

    if remove_zero_duration_notes:
        # If there are any zero-duration notes (grace notes?), drop them
        out_df = out_df[out_df["onset"] != out_df["release"]]

    out_df["type"] = "note"
    out_df = _add_time_sigs(out_df)
    out_df = _add_bars(out_df)

    # out_df = infer_barlines(out_df)

    if drop_unused_cols:
        return_columns = [
            "pitch",
            "onset",
            "release",
            "tie_to_next",
            "tie_to_prev",
            "voice",
            "part",
            "spelling",
            "type",
            "other",
        ]

        out_df = out_df[return_columns]
    return sort_df(out_df)
