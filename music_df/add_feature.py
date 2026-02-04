"""
This module contains diverse functions for adding features to music dataframes.

"""

import io  # Used by doctests # noqa: F401
import re
from ast import literal_eval
from functools import partial
from itertools import chain
from math import isnan
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from music_df.constants import NAME_TO_MIDI_INSTRUMENT
from music_df.transpose import PERCUSSION_CHANNEL


def _tempo2bpm(tempo: int) -> float:
    """Convert MIDI tempo (microseconds per beat) to BPM."""
    return 60_000_000 / tempo
from music_df.sort_df import sort_df


def _to_dict_if_necessary(d):
    if isinstance(d, str):
        return literal_eval(d)
    return d


def _time_sig_dur(time_sig: dict[str, int]):
    return time_sig["numerator"] * 4 / time_sig["denominator"]


def _time_sig_dur_from_row(row):
    return row.ts_numerator * 4 / row.ts_denominator


def _time_signature_reduce(
    numerator, denominator, max_ts_denominator: int = 6, max_notes_per_bar: int = 2
):
    # from MusicBERT

    # reduction (when denominator is too large)
    while (
        denominator > 2**max_ts_denominator
        and denominator % 2 == 0
        and numerator % 2 == 0
    ):
        denominator //= 2
        numerator //= 2
    # decomposition (when length of a bar exceed max_notes_per_bar)
    while numerator > max_notes_per_bar * denominator:
        for i in range(2, numerator + 1):
            if numerator % i == 0:
                numerator //= i
                break
    return numerator, denominator


def simplify_time_sigs(
    music_df: pd.DataFrame,
    simplify_func: Callable[[int, int], tuple[int, int]] = _time_signature_reduce,
) -> pd.DataFrame:
    time_sig_mask = music_df.type == "time_signature"

    def f(other):
        dict_ = _to_dict_if_necessary(other)
        dict_["numerator"], dict_["denominator"] = simplify_func(
            dict_["numerator"], dict_["denominator"]
        )
        return dict_

    music_df.loc[time_sig_mask, "other"] = music_df[time_sig_mask].other.apply(f)
    return music_df


def infer_barlines(
    music_df: pd.DataFrame, keep_old_index: bool = False
) -> pd.DataFrame:
    time_sig_mask = music_df.type == "time_signature"
    time_sigs = [series for (_, series) in music_df[time_sig_mask].iterrows()]

    notes = music_df[music_df.type == "note"]
    if notes.empty:
        return music_df

    assert time_sigs and (
        time_sigs[0].onset <= notes.iloc[0].onset
    ), (
        "There is no time signature before the first note; default time signature not yet implemented"
    )

    assert len(time_sigs)

    barline_onset_accumulator = []

    for time_sig1, time_sig2 in zip(time_sigs, chain(time_sigs[1:], [None])):
        time_sig_dur = _time_sig_dur(_to_dict_if_necessary(time_sig1.other))
        if time_sig2 is not None:
            end = time_sig2.onset
        else:
            end = music_df.release.max()
        barline_onset_accumulator.append(
            np.arange(time_sig1.onset, end, step=time_sig_dur)
        )
    barline_onsets = np.concatenate(barline_onset_accumulator)
    barline_releases = np.concatenate([barline_onsets[1:], [end]])  # type:ignore
    barlines = pd.DataFrame({"onset": barline_onsets, "release": barline_releases})
    barlines["type"] = "bar"

    # Ensure that index values will be unique
    barlines.index += max(music_df.index) + 1

    out_df = pd.concat([music_df, barlines])
    out_df = sort_df(out_df, ignore_index=False)

    out_df = out_df.reset_index(drop=not keep_old_index)

    return out_df


def make_time_signatures_explicit(
    music_df: pd.DataFrame, default_time_signature: dict[str, int] | None = None
) -> pd.DataFrame:
    """
    Add "ts_numerator" and "ts_denominator" columns to the dataframe.

    Thus every row in the dataframe will have an explicit time signature.

    If the dataframe already has "ts_numerator" and "ts_denominator" columns, it is
    returned unchanged.

    Args:
        music_df: The dataframe to add the time signature columns to.
        default_time_signature: The time signature to use if no time signature is
            present in the dataframe. If not provided, 4/4 is assumed.
    """
    if "ts_numerator" in music_df.columns and "ts_denominator" in music_df.columns:
        # There appears to be nothing to be done
        return music_df

    music_df = music_df.copy()

    if default_time_signature is None:
        default_time_signature = {"numerator": 4, "denominator": 4}

    time_sig_mask = music_df.type == "time_signature"
    time_sigs = music_df[music_df.type == "time_signature"]
    music_df["ts_numerator"] = float("nan")
    music_df["ts_denominator"] = float("nan")

    # krn music_df function seems to return time sigs as string representations of
    # dicts, whereas midi function returns them as dicts. Probably the latter behavior
    #   should be enforced everywhere.
    music_df.loc[time_sig_mask, "ts_numerator"] = [
        _to_dict_if_necessary(d)["numerator"] for _, d in time_sigs.other.items()
    ]
    music_df.loc[time_sig_mask, "ts_denominator"] = [
        _to_dict_if_necessary(d)["denominator"] for _, d in time_sigs.other.items()
    ]

    music_df["ts_numerator"] = music_df.ts_numerator.ffill()
    music_df["ts_denominator"] = music_df.ts_denominator.ffill()

    music_df["ts_numerator"] = music_df.ts_numerator.fillna(
        value=default_time_signature["numerator"]
    )
    music_df["ts_denominator"] = music_df.ts_denominator.fillna(
        value=default_time_signature["denominator"]
    )
    music_df["ts_numerator"] = music_df.ts_numerator.astype(int)
    music_df["ts_denominator"] = music_df.ts_denominator.astype(int)
    return music_df


def add_default_time_sig(
    music_df: pd.DataFrame,
    default_time_signature: dict[str, int] | None = None,
    keep_old_index: bool = False,
) -> pd.DataFrame:
    """
    Add default time signature to dataframes that lack them (or lack an initial one).

    >>> nan = float("nan")  # Alias to simplify below assignments

    No time signature at all:
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, 60, nan, 62],
    ...         "onset": [0, 0, 4, 4],
    ...         "release": [4, 4, 8, 5],
    ...         "type": ["bar", "note", "bar", "note"],
    ...         "other": [nan, nan, nan, nan],
    ...     }
    ... )
    >>> df
       pitch  onset  release  type  other
    0    NaN      0        4   bar    NaN
    1   60.0      0        4  note    NaN
    2    NaN      4        8   bar    NaN
    3   62.0      4        5  note    NaN
    >>> add_default_time_sig(df)
       pitch  onset  release            type                               other
    0    NaN      0      NaN  time_signature  {'numerator': 4, 'denominator': 4}
    1    NaN      0      4.0             bar                                 NaN
    2   60.0      0      4.0            note                                 NaN
    3    NaN      4      8.0             bar                                 NaN
    4   62.0      4      5.0            note                                 NaN

    Missing initial time signature:
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, 60, nan, nan, 62],
    ...         "onset": [0, 0, 4, 4, 4],
    ...         "release": [4, 4, nan, 7, 5],
    ...         "type": ["bar", "note", "time_signature", "bar", "note"],
    ...         "other": [nan, nan, {"numerator": 3, "denominator": 4}, nan, nan],
    ...     }
    ... )
    >>> df
       pitch  onset  release            type                               other
    0    NaN      0      4.0             bar                                 NaN
    1   60.0      0      4.0            note                                 NaN
    2    NaN      4      NaN  time_signature  {'numerator': 3, 'denominator': 4}
    3    NaN      4      7.0             bar                                 NaN
    4   62.0      4      5.0            note                                 NaN
    >>> add_default_time_sig(df)
       pitch  onset  release            type                               other
    0    NaN      0      NaN  time_signature  {'numerator': 4, 'denominator': 4}
    1    NaN      0      4.0             bar                                 NaN
    2   60.0      0      4.0            note                                 NaN
    3    NaN      4      NaN  time_signature  {'numerator': 3, 'denominator': 4}
    4    NaN      4      7.0             bar                                 NaN
    5   62.0      4      5.0            note                                 NaN

    No missing time signature:
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, nan, 60, nan, nan, 62],
    ...         "onset": [0, 0, 0, 4, 4, 4],
    ...         "release": [nan, 4, 4, nan, 7, 5],
    ...         "type": [
    ...             "time_signature",
    ...             "bar",
    ...             "note",
    ...             "time_signature",
    ...             "bar",
    ...             "note",
    ...         ],
    ...         "other": [
    ...             {"numerator": 4, "denominator": 4},
    ...             nan,
    ...             nan,
    ...             {"numerator": 3, "denominator": 4},
    ...             nan,
    ...             nan,
    ...         ],
    ...     }
    ... )
    >>> df
       pitch  onset  release            type                               other
    0    NaN      0      NaN  time_signature  {'numerator': 4, 'denominator': 4}
    1    NaN      0      4.0             bar                                 NaN
    2   60.0      0      4.0            note                                 NaN
    3    NaN      4      NaN  time_signature  {'numerator': 3, 'denominator': 4}
    4    NaN      4      7.0             bar                                 NaN
    5   62.0      4      5.0            note                                 NaN
    >>> df.equals(add_default_time_sig(df))
    True
    """

    time_sig_mask = music_df.type == "time_signature"
    if time_sig_mask.any() and (
        music_df[time_sig_mask].index[0]
        <= music_df[music_df.type.isin({"note", "bar"})].index[0]
    ):
        return music_df
    column_order = music_df.columns
    if default_time_signature is None:
        default_time_signature = {"numerator": 4, "denominator": 4}

    time_sig_df = pd.DataFrame(
        {
            "onset": [0],
            "type": ["time_signature"],
            "other": [default_time_signature],
        }
    )

    if "ts_numerator" in music_df.columns:
        time_sig_df["ts_numerator"] = [default_time_signature["numerator"]]
    if "ts_denominator" in music_df.columns:
        time_sig_df["ts_denominator"] = [default_time_signature["denominator"]]

    # ensure indices are unique
    time_sig_df.index += max(music_df.index) + 1
    out_df = pd.concat([time_sig_df, music_df], axis=0)
    out_df = out_df[column_order]

    out_df = out_df.reset_index(drop=not keep_old_index)
    return out_df


def make_tempos_explicit(music_df: pd.DataFrame, default_tempo: float) -> pd.DataFrame:
    """
    Add "tempo" column to the dataframe.

    Thus every row in the dataframe will have an explicit tempo.

    If there are no tempo events in the dataframe, the tempo is set to the default
    tempo. The default tempo is also used for any rows that precede the first tempo.

    Args:
        music_df: The dataframe to add the tempo column to.
        default_tempo: The tempo to use if no tempo events are present.
    """
    # If there already *is* a tempo column, we just want to make sure it doesn't
    #   have any nans in it
    if "tempo" in music_df.columns:
        music_df["tempo"] = music_df.tempo.ffill()
        music_df["tempo"] = music_df.tempo.fillna(value=default_tempo)
        return music_df

    # Otherwise, we check for tempo events

    # First handle midi tempi
    tempo_mask = music_df.type == "set_tempo"
    music_df["tempo"] = float("nan")
    music_df.loc[tempo_mask, "tempo"] = [
        _tempo2bpm(_to_dict_if_necessary(d)["tempo"])
        for _, d in music_df[tempo_mask].other.items()
    ]
    # Next handle BPM tempi from musicxml etc
    tempo_mask = music_df.type == "tempo"
    music_df.loc[tempo_mask, "tempo"] = [
        _to_dict_if_necessary(d)["tempo"] for _, d in music_df[tempo_mask].other.items()
    ]
    music_df["tempo"] = music_df.tempo.ffill()
    music_df["tempo"] = music_df.tempo.fillna(value=default_tempo)

    return music_df


def add_time_sig_dur(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "time_sig_dur" column specifying the quarter duration of each time signature.
    """
    music_df["time_sig_dur"] = float("nan")
    music_df.loc[music_df.type == "time_signature", "time_sig_dur"] = music_df[
        music_df.type == "time_signature"
    ].apply(_time_sig_dur_from_row, axis=1)
    music_df["time_sig_dur"] = music_df.time_sig_dur.ffill()
    return music_df


def add_bar_durs(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "bar_dur" column specifying the duration of each bar.

    The "bar_dur" column will be NaN for non-bar rows. We also set the "release" column
    to the sum of the "onset" and "bar_dur" columns for bar rows.

    Args:
        music_df: The dataframe to add the bar duration column to. Must have at least
            one bar (i.e., one row with type == "bar").
    """
    bar_mask = music_df.type == "bar"
    if not bar_mask.any():
        raise ValueError("Score must have at least one bar")
    bars = music_df[bar_mask]
    bar_durs = bars.iloc[1:].onset.reset_index(drop=True) - bars.iloc[
        :-1
    ].onset.reset_index(drop=True)
    last_bar = bars.iloc[-1]
    last_bar_dur = music_df.release.max() - last_bar.onset
    bar_durs = pd.concat([bar_durs, pd.Series([last_bar_dur])]).reset_index(drop=True)
    music_df["bar_dur"] = float("nan")
    music_df.loc[bar_mask, "bar_dur"] = bar_durs.astype(float).to_numpy()
    music_df.loc[bar_mask, "release"] = (
        music_df.loc[bar_mask, "onset"] + music_df.loc[bar_mask, "bar_dur"]
    )
    return music_df


def split_long_bars(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Split "long" bars (bars whose actual duration exceeds the time signature duration).

    Note: sorts result before returning it.
    """
    assert (
        "ts_numerator" in music_df.columns and "ts_denominator" in music_df.columns
    ), "call make_time_signatures_explicit(music_df) first"

    orig_cols = music_df.columns

    music_df = add_time_sig_dur(music_df)

    music_df = add_bar_durs(music_df)

    long_bars = music_df["bar_dur"] > music_df["time_sig_dur"]
    if long_bars.any():
        added_bars = []
        for i, long_bar in music_df[long_bars].iterrows():
            last_release = long_bar.release
            assert not isnan(last_release)
            remaining_dur = long_bar.bar_dur - long_bar.time_sig_dur
            onset = long_bar.onset
            prev_bar = long_bar

            # We need to modify the release of the long measure in place
            music_df.loc[i, "release"] = onset + long_bar.time_sig_dur  # type:ignore

            new_bar = None
            while remaining_dur > 0:
                onset += long_bar.time_sig_dur
                new_bar = long_bar.copy()
                new_bar.onset = onset
                added_bars.append(new_bar)
                prev_bar.release = onset
                prev_bar = new_bar
                remaining_dur -= long_bar.time_sig_dur

            assert new_bar is not None
            new_bar.release = last_release
        music_df = pd.concat([music_df, pd.DataFrame(added_bars)])
        music_df = sort_df(music_df)
    return music_df[orig_cols]


def number_bars(music_df: pd.DataFrame, initial_bar_number: int = 1) -> pd.DataFrame:
    """
    Add "bar_number" column specifying the number of each bar.

    Args:
        music_df: The dataframe to add the bar number column to. The dataframe must have
            at least one bar (i.e., one row with type == "bar").
        initial_bar_number: The number of the first bar. The convention in music is that
            the first full bar should be numbered 1. Note that this function isn't smart
            enough to distinguish pickup measures (normally numbered 0 in music
            notation).
    """
    bar_mask = music_df.type == "bar"
    if not bar_mask.sum():
        raise ValueError("No bars found")
    bar_numbers = np.arange(
        start=initial_bar_number, stop=bar_mask.sum() + initial_bar_number
    )

    music_df["bar_number"] = float("nan")
    music_df.loc[bar_mask, "bar_number"] = bar_numbers

    return music_df


def make_bar_explicit(
    music_df: pd.DataFrame, default_bar_number: int = -1, initial_bar_number: int = 1
) -> pd.DataFrame:
    """
    Add "bar_number" column specifying the bar number of each row.

    Thus every row in the dataframe will have an explicit bar number.

    The actual bar numbering is performed by the number_bars function.

    Args:
        music_df: The dataframe to add the bar number column to. The dataframe must have
            at least one bar (i.e., one row with type == "bar").
        default_bar_number: The number to use for rows that precede the first bar.
        initial_bar_number: The number of the first bar. The convention in music is that
            the first full bar should be numbered 1. Note that this function isn't smart
            enough to distinguish pickup measures (normally numbered 0 in music
            notation).
    """
    bar_mask = music_df.type == "bar"
    # TODO: (Malcolm 2023-12-25) maybe I should use appears_to_have_pickup_measure to
    #   determine initial_bar_number?
    if not len(bar_mask):
        raise ValueError("No bars found")

    music_df = number_bars(music_df, initial_bar_number)
    music_df.loc[:, "bar_number"] = music_df["bar_number"].ffill()
    music_df.loc[:, "bar_number"] = music_df["bar_number"].fillna(
        value=default_bar_number
    )
    music_df.loc[:, "bar_number"] = music_df.bar_number.astype(int)
    return music_df


def get_bar_relative_onset(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "bar_relative_onset" column specifying the offset of each row from the onset
    of the bar.

    For example, if a note has onset 4.5 and the preceding bar has onset 3, then the
    bar relative onset is 1.5.

    Args:
        music_df: The dataframe to add the bar relative onset column to.
    """
    bar_mask = music_df.type == "bar"
    if not len(bar_mask):
        raise ValueError("No bars found")
    music_df["bar_onset"] = float("nan")
    music_df.loc[bar_mask, "bar_onset"] = music_df.onset[bar_mask]
    music_df["bar_onset"] = music_df.bar_onset.ffill()

    null_mask = music_df["bar_onset"].isnull()

    # No notes should have null values
    assert not (music_df[null_mask].type == "note").sum()
    music_df["bar_onset"] = music_df.bar_onset.fillna(value=0)

    # assert not music_df["bar_onset"].isnull().values.any()  # type:ignore
    music_df["bar_relative_onset"] = music_df.onset - music_df.bar_onset
    music_df = music_df.drop("bar_onset", axis=1)
    return music_df


def add_default_velocity(
    music_df: pd.DataFrame, default_velocity: int = 96
) -> pd.DataFrame:
    """
    Add default velocity where it is missing.
    """
    if "velocity" not in music_df.columns:
        music_df["velocity"] = default_velocity
    else:
        music_df["velocity"] = music_df.velocity.fillna(value=default_velocity)
    return music_df


def add_default_midi_instrument(
    music_df: pd.DataFrame,
    default_instrument: int = 0,
) -> pd.DataFrame:
    """
    Add default MIDI instrument where it is missing.
    """
    if "midi_instrument" not in music_df.columns:
        music_df["midi_instrument"] = default_instrument
    else:
        music_df["midi_instrument"] = music_df.midi_instrument.fillna(
            value=default_instrument
        )
    return music_df


def make_instruments_explicit(
    music_df: pd.DataFrame, default_instrument: int = 0
) -> pd.DataFrame:
    """
    Add "midi_instrument" column specifying the MIDI instrument of each row.

    Args:
        music_df: The dataframe to add the MIDI instrument column to.
        default_instrument: The MIDI instrument to use if no program changes are
            present.
    """
    if "track" not in music_df.columns:
        return add_default_midi_instrument(music_df, default_instrument)
    program_change_mask = music_df.type == "program_change"
    music_df["midi_instrument"] = float("nan")
    music_df.loc[program_change_mask, "midi_instrument"] = [
        _to_dict_if_necessary(d)["program"]
        for _, d in music_df.other[program_change_mask].items()
    ]

    grouped_by_track = music_df.groupby("track", dropna=False)
    accumulator = []
    for track, group_df in grouped_by_track:
        group_df["midi_instrument"] = group_df.midi_instrument.ffill()
        accumulator.append(group_df)

    out_df = pd.concat(accumulator)
    out_df = out_df.sort_index(axis=0)

    out_df = add_default_midi_instrument(out_df, default_instrument=default_instrument)
    out_df["midi_instrument"] = out_df.midi_instrument.astype(int)
    return out_df


def explicit_instruments_to_program_changes(music_df: pd.DataFrame) -> pd.DataFrame:
    """This is a sort of inverse of make_instruments_explicit."""
    if "midi_instrument" not in music_df.columns:
        raise ValueError("midi_instrument column not found")
    program_changes = []
    for (track, channel), contents in music_df.groupby(["track", "channel"]):
        # We rely on the order of the dataframe being preserved by groupby, see
        #  https://stackoverflow.com/a/26465555/10155119
        instrument_changes = contents.midi_instrument != contents.midi_instrument.shift(
            1
        )
        reference_rows = contents[instrument_changes]
        for i, row in reference_rows.iterrows():
            program_change = pd.Series(
                {
                    "type": "program_change",
                    "track": track,
                    "channel": channel,
                    "onset": row.onset,
                    "other": {"program": row.midi_instrument},
                }
            )
            program_changes.append(program_change)

    out_df = pd.concat([music_df, pd.DataFrame(program_changes)])
    out_df = sort_df(out_df)
    return out_df


def instruments_to_midi_instruments(
    music_df: pd.DataFrame,
    default_instrument: int = 0,
    translation: dict[str, int] = NAME_TO_MIDI_INSTRUMENT,
    raise_error_on_missing: bool = False,
) -> pd.DataFrame:
    """This function sends an "instrument" column with string values to a
    "midi_instrument" column with ints specifying General MIDI programs.
    I am not actually using it anywhere, however.
    """
    music_df = music_df.copy()
    if raise_error_on_missing:
        missing = []
        for instr in music_df.instrument.unique():
            if instr not in translation:
                missing.append(instr)
        if missing:
            raise ValueError(f"Missing instrument translations for {missing}")
    music_df["midi_instrument"] = music_df.instrument.apply(
        lambda x: translation.get(x, default_instrument)
    )
    return music_df


def add_scale_degrees(music_df: pd.DataFrame):
    """
    Add "scale_degree" column specifying the scale degree of each note.

    The scale degree is inferred from the note's spelling and key. See examples
    below.

    >>> df = pd.DataFrame(
    ...     {
    ...         # we omit all other columns
    ...         # (Malcolm 2023-12-22) Note that "bb" is not supported by music21 so
    ...         #   we handle it separately.
    ...         "type": ["bar"] + ["note"] * 9,
    ...         "spelling": [
    ...             float("nan"),
    ...             "Db",
    ...             "F",
    ...             "Gb",
    ...             "C",
    ...             "C#",
    ...             "C##",
    ...             "Fb",
    ...             "F--",
    ...             "Fbb",
    ...         ],
    ...         "key": ["na"] + ["Gb"] * 9,
    ...     }
    ... )
    >>> add_scale_degrees(df)
       type spelling key scale_degree
    0   bar      NaN  na           na
    1  note       Db  Gb            5
    2  note        F  Gb            7
    3  note       Gb  Gb            1
    4  note        C  Gb           #4
    5  note       C#  Gb          ##4
    6  note      C##  Gb         ###4
    7  note       Fb  Gb           b7
    8  note      F--  Gb          bb7
    9  note      Fbb  Gb          bb7

    There is an issue with some scale degrees turning up as floats when reading saved
    CSVs later. To avoid this, we replace NaN with "na" string.
    >>> df = pd.DataFrame(
    ...     {
    ...         # we omit all other columns
    ...         "type": ["bar"] + ["note"] * 2,
    ...         "spelling": [float("nan"), "C", "F"],
    ...         "key": ["na"] + ["C"] * 2,
    ...     }
    ... )
    >>> scale_degrees_df = add_scale_degrees(df)
    >>> scale_degrees_df
       type spelling key scale_degree
    0   bar      NaN  na           na
    1  note        C   C            1
    2  note        F   C            4
    >>> from io import StringIO
    >>> output = StringIO()
    >>> df.to_csv(output)
    >>> csv_str = output.getvalue()
    >>> input_ = StringIO(csv_str)
    >>> df2 = pd.read_csv(input_)
    >>> df2
       Unnamed: 0  type spelling key scale_degree
    0           0   bar      NaN  na           na
    1           1  note        C   C            1
    2           2  note        F   C            4

    Checking minor key behavior
    >>> df = pd.DataFrame(
    ...     {
    ...         # we omit all other columns
    ...         "type": ["bar"] + ["note"] * 10,
    ...         "spelling": [
    ...             float("nan"),
    ...             "E",
    ...             "Fb",
    ...             "F",
    ...             "F#",
    ...             "Gb",
    ...             "G",
    ...             "G#",
    ...             "Ab",
    ...             "G##",
    ...             "A",
    ...         ],
    ...         "key": ["na"] + ["a"] * 10,
    ...     }
    ... )
    >>> add_scale_degrees(df)
        type spelling key scale_degree
    0    bar      NaN  na           na
    1   note        E   a            5
    2   note       Fb   a           b6
    3   note        F   a            6
    4   note       F#   a           #6
    5   note       Gb   a           b7
    6   note        G   a            7
    7   note       G#   a           #7
    8   note       Ab   a           b1
    9   note      G##   a          ##7
    10  note        A   a            1
    """
    from music21.key import Key
    from music21.pitch import Pitch

    assert "spelling" in music_df.columns
    assert "key" in music_df.columns

    mapping = {}

    # (Malcolm 2023-12-22) we could save a little time caching keys globally
    keys = {}

    for (spelling, key), _ in music_df.groupby(["spelling", "key"]):
        if key not in keys:
            key_obj = Key(key)
            keys[key] = key_obj
        else:
            key_obj = keys[key]

        scale_degree_int, accidental = key_obj.getScaleDegreeAndAccidentalFromPitch(
            Pitch(spelling[0] + spelling[1:].replace("b", "-"))
        )

        if accidental is None:
            scale_degree = str(scale_degree_int)
        else:
            scale_degree = f"{accidental.modifier.replace('-', 'b')}{scale_degree_int}"

        mapping[(spelling, key)] = scale_degree

    note_mask = music_df.type == "note"
    music_df["scale_degree"] = "na"
    music_df.loc[note_mask, "scale_degree"] = music_df.loc[note_mask].apply(
        lambda row: mapping[(row.spelling, row.key)], axis=1, result_type=None
    )

    return music_df


def decompose_scale_degrees(music_df: pd.DataFrame, max_alteration: int = 2):
    """
    Decompose "scale_degree" into "scale_degree_step" and "scale_degree_alteration".

    For example,
       - the scale degree 5 has step 5 and alteration "_"
       - the scale degree #4 has step 4 and alteration "#"
       - the scale degree bb7 has step 7 and alteration "bb"

    Args:
        music_df: The dataframe to decompose the scale degree column of.
        max_alteration: The maximum number of accidentals to allow. If the alteration
            is greater than this, it is set to "x".

    >>> df = pd.DataFrame(
    ...     {
    ...         # we omit all other columns
    ...         # (Malcolm 2023-12-22) Note that "bb" is not supported by music21 so
    ...         #   we handle it separately.
    ...         "type": ["bar"] + ["note"] * 9,
    ...         "spelling": [
    ...             float("nan"),
    ...             "Db",
    ...             "F",
    ...             "Gb",
    ...             "C",
    ...             "C#",
    ...             "C##",
    ...             "Fb",
    ...             "F--",
    ...             "Fbb",
    ...         ],
    ...         "key": ["na"] + ["Gb"] * 9,
    ...     }
    ... )
    >>> df = add_scale_degrees(df)
    >>> decompose_scale_degrees(df)
       type spelling key scale_degree scale_degree_step scale_degree_alteration
    0   bar      NaN  na           na                na                      na
    1  note       Db  Gb            5                 5                       _
    2  note        F  Gb            7                 7                       _
    3  note       Gb  Gb            1                 1                       _
    4  note        C  Gb           #4                 4                       #
    5  note       C#  Gb          ##4                 4                      ##
    6  note      C##  Gb         ###4                 4                       x
    7  note       Fb  Gb           b7                 7                       b
    8  note      F--  Gb          bb7                 7                      bb
    9  note      Fbb  Gb          bb7                 7                      bb
    """
    music_df["scale_degree_step"] = "na"
    music_df["scale_degree_alteration"] = "na"

    note_mask = music_df.type == "note"

    music_df.loc[note_mask, "scale_degree_step"] = music_df.loc[
        note_mask, "scale_degree"
    ].apply(
        lambda s: re.search(r"\d+", s).group()  # type:ignore
    )
    music_df.loc[note_mask, "scale_degree_alteration"] = music_df.loc[
        note_mask, "scale_degree"
    ].apply(
        lambda s: re.search(r"\D*", s).group()  # type:ignore
    )
    music_df.loc[note_mask, "scale_degree_alteration"] = music_df.loc[
        note_mask, "scale_degree_alteration"
    ].apply(lambda s: s if len(s) <= max_alteration else "x")
    music_df.loc[music_df.scale_degree_alteration == "", "scale_degree_alteration"] = (
        "_"
    )
    return music_df


def concatenate_features(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    """
    Create a new feature by concatenating the values of the given features.
    >>> csv_table = '''
    ... type,pitch,onset,release,foo,bar
    ... bar,,0.0,4.0,,
    ... note,60,0.0,0.5,a,1.0
    ... note,60,0.0,1.5,b,2.0
    ... note,60,1.0,2.0,c,3.0
    ... note,60,2.0,3.0,d,4.0
    ... bar,,4.0,8.0,,
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release  foo  bar
    0   bar    NaN    0.0      4.0  NaN  NaN
    1  note   60.0    0.0      0.5    a  1.0
    2  note   60.0    0.0      1.5    b  2.0
    3  note   60.0    1.0      2.0    c  3.0
    4  note   60.0    2.0      3.0    d  4.0
    5   bar    NaN    4.0      8.0  NaN  NaN
    >>> concatenate_features(df, ["foo", "bar"])
       type  pitch  onset  release  foo  bar foo_bar
    0   bar    NaN    0.0      4.0  NaN  NaN      na
    1  note   60.0    0.0      0.5    a  1.0    a1.0
    2  note   60.0    0.0      1.5    b  2.0    b2.0
    3  note   60.0    1.0      2.0    c  3.0    c3.0
    4  note   60.0    2.0      3.0    d  4.0    d4.0
    5   bar    NaN    4.0      8.0  NaN  NaN      na
    """
    concat_feature_name = "_".join(features)
    assert concat_feature_name not in df.columns
    df[concat_feature_name] = df[features].astype(str).sum(axis=1)
    df.loc[
        ((df[features].isna()) | (df[features] == "na")).any(axis=1),
        concat_feature_name,
    ] = "na"
    return df


def _key_signature_from_key(key: str) -> int:
    """
    >>> _key_signature_from_key("C")
    0
    >>> _key_signature_from_key("F")
    -1
    >>> _key_signature_from_key("F#")
    6
    >>> _key_signature_from_key("f#")
    3
    >>> _key_signature_from_key("f##")
    10
    >>> _key_signature_from_key("Cb")
    -7
    >>> _key_signature_from_key("cb")
    -10
    """
    major_mode = key[0].isupper()
    pitch = key[0].upper()
    alteration_str = key[1:]
    alteration = 0
    if alteration_str:
        flats = alteration_str.count("b")
        sharps = alteration_str.count("#")
        assert not (flats and sharps)
        alteration = (sharps - flats) * 7
    raw_key_sig = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F": -1}[pitch]
    if not major_mode:
        raw_key_sig -= 3
    return raw_key_sig + alteration


def add_key_signature_from_key(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "key_signature" column specifying the key signature of each row.

    Sharps are positive, flats are negative.
    >>> csv_table = '''
    ... type,pitch,key
    ... bar,,
    ... note,60,C
    ... note,69,a
    ... note,68,ab
    ... note,68,Ab
    ... note,70,Cbb
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> add_key_signature_from_key(df)
       type  pitch  key  key_signature
    0   bar    NaN  NaN            NaN
    1  note   60.0    C            0.0
    2  note   69.0    a            0.0
    3  note   68.0   ab           -7.0
    4  note   68.0   Ab           -4.0
    5  note   70.0  Cbb          -14.0
    """
    key_row_mask = ~music_df["key"].isna()
    music_df["key_signature"] = float("nan")
    music_df.loc[key_row_mask, "key_signature"] = music_df.loc[
        key_row_mask, "key"
    ].apply(_key_signature_from_key)
    return music_df


def add_enharmonic_key_signature_from_key(music_df: pd.DataFrame) -> pd.DataFrame:
    """An "enharmonic" key signature is between -5 (5 flats) and 6 (6 sharps).

    For example, F# and Gb have different key signatures (6 and -6 respectively), but
    the same enharmonic key signature (6).

    >>> csv_table = '''
    ... type,pitch,key
    ... bar,,
    ... note,60,B#
    ... note,69,b-
    ... note,68,F#
    ... note,68,Gb
    ... note,70,Cbb
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> add_enharmonic_key_signature_from_key(df)
       type  pitch  key  key_signature  enh_key_signature
    0   bar    NaN  NaN            NaN                NaN
    1  note   60.0   B#           12.0                0.0
    2  note   69.0   b-            2.0                2.0
    3  note   68.0   F#            6.0                6.0
    4  note   68.0   Gb           -6.0                6.0
    5  note   70.0  Cbb          -14.0               -2.0
    """
    if "key_signature" not in music_df.columns:
        music_df = add_key_signature_from_key(music_df)
    music_df["enh_key_signature"] = music_df["key_signature"] % 12
    music_df.loc[(music_df["enh_key_signature"] > 6), "enh_key_signature"] -= 12
    return music_df


PC_TO_KEY_SIG = (0, -5, 2, -3, 4, -1, 6, 1, -4, 3, -2, 5)


def _key_signature_from_pc_and_mode(
    row, pc_col_name: str = "key_pc", mode_col_name: str = "mode"
) -> float:
    if isnan(row[pc_col_name]) or (
        isinstance(row[mode_col_name], float) and isnan(row[mode_col_name])
    ):
        return float("nan")
    out = (
        PC_TO_KEY_SIG[int(row[pc_col_name])]
        + (-3 if row[mode_col_name] == "m" else 0) % 12
    )
    if out > 6:
        out -= 12
    return out


def add_key_signature_from_pc_and_mode(
    music_df: pd.DataFrame,
    added_col_name: str = "key_signature",
    pc_col_name: str = "key_pc",
    mode_col_name: str = "mode",
) -> pd.DataFrame:
    """
    Add a key signature column inferred from the key pitch-class and mode.

    Because pitch-classes are enharmonic, key signatures are enharmonic as well
    (between -5 and 6). In other words, both F# and Gb will be indicated by key_pc=6
    and mode="M", so they can't be distinguished from each other.

    >>> csv_table = '''
    ... type,pitch,key_pc,mode
    ... bar,,
    ... note,60,0,M
    ... note,69,9,m
    ... note,68,9,M
    ... note,68,8,m
    ... note,70,3,m
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> add_key_signature_from_pc_and_mode(df)
       type  pitch  key_pc mode  key_signature
    0   bar    NaN     NaN  NaN            NaN
    1  note   60.0     0.0    M            0.0
    2  note   69.0     9.0    m            0.0
    3  note   68.0     9.0    M            3.0
    4  note   68.0     8.0    m            5.0
    5  note   70.0     3.0    m            6.0
    """
    music_df[added_col_name] = music_df.apply(
        partial(
            _key_signature_from_pc_and_mode,
            pc_col_name=pc_col_name,
            mode_col_name=mode_col_name,
        ),
        axis=1,
    )
    return music_df


def add_key_signature(music_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add "key_signature" column specifying the key signature of each row.

    The dataframe must have either:
        - a "key_pc" column specifying the pitch-class of the key, and a "mode" column
          specifying the mode (M or m), or
        - a "key" column specifying the key.

    >>> csv_table = '''
    ... type,pitch,key_pc,mode
    ... bar,,
    ... note,60,0,M
    ... note,69,9,m
    ... note,68,9,M
    ... note,68,8,m
    ... note,70,3,m
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> add_key_signature(df)
       type  pitch  key_pc mode  key_signature
    0   bar    NaN     NaN  NaN            NaN
    1  note   60.0     0.0    M            0.0
    2  note   69.0     9.0    m            0.0
    3  note   68.0     9.0    M            3.0
    4  note   68.0     8.0    m            5.0
    5  note   70.0     3.0    m            6.0

    >>> csv_table = '''
    ... type,pitch,key
    ... bar,,
    ... note,60,C
    ... note,69,a
    ... note,68,ab
    ... note,68,Ab
    ... note,70,Cbb
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> add_key_signature(df)
       type  pitch  key  key_signature
    0   bar    NaN  NaN            NaN
    1  note   60.0    C            0.0
    2  note   69.0    a            0.0
    3  note   68.0   ab           -7.0
    4  note   68.0   Ab           -4.0
    5  note   70.0  Cbb          -14.0
    """
    if "mode" in music_df.columns and "key_pc" in music_df.columns:
        return add_key_signature_from_pc_and_mode(music_df)
    elif "key" in music_df.columns:
        return add_key_signature_from_key(music_df)
    else:
        raise ValueError


def add_sounding_bass(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'sounding_bass_idx' column: index of lowest-pitched note sounding at each note's onset.

    A note is sounding at time t if: onset <= t < release
    Non-note rows get NaN.

    >>> df = pd.DataFrame({
    ...     "type": ["note", "note", "note"],
    ...     "pitch": [60.0, 48.0, 55.0],
    ...     "onset": [0.0, 0.0, 1.0],
    ...     "release": [2.0, 1.5, 2.5],
    ... })
    >>> result = add_sounding_bass(df)
    >>> result["sounding_bass_idx"].tolist()
    [1.0, 1.0, 1.0]

    If multiple notes share the lowest pitch, the first by index wins:
    >>> df = pd.DataFrame({
    ...     "type": ["note", "note"],
    ...     "pitch": [60.0, 60.0],
    ...     "onset": [0.0, 0.0],
    ...     "release": [1.0, 1.0],
    ... })
    >>> result = add_sounding_bass(df)
    >>> result["sounding_bass_idx"].tolist()
    [0.0, 0.0]

    Non-note rows get NaN:
    >>> df = pd.DataFrame({
    ...     "type": ["bar", "note", "note"],
    ...     "pitch": [float("nan"), 60.0, 48.0],
    ...     "onset": [0.0, 0.0, 0.0],
    ...     "release": [4.0, 1.0, 1.0],
    ... })
    >>> result = add_sounding_bass(df)
    >>> import math
    >>> math.isnan(result["sounding_bass_idx"].iloc[0])
    True
    >>> result["sounding_bass_idx"].iloc[1:].tolist()
    [2.0, 2.0]
    """
    df = df.copy()
    df["sounding_bass_idx"] = float("nan")

    notes = df[df.type == "note"]
    if len(notes) == 0:
        return df

    # Exclude percussion from bass candidates (pitches represent drums, not notes)
    if "channel" in notes.columns:
        bass_candidates = notes[notes["channel"] != PERCUSSION_CHANNEL]
    else:
        bass_candidates = notes

    onsets = notes["onset"].values
    releases = notes["release"].values

    cand_onsets = bass_candidates["onset"].values
    cand_releases = bass_candidates["release"].values
    cand_pitches = bass_candidates["pitch"].values
    cand_indices = bass_candidates.index.values

    unique_onsets = np.unique(onsets)
    bass_idx_map = {}

    for t in unique_onsets:
        sounding = (cand_onsets <= t) & (cand_releases > t)
        if sounding.any():
            sounding_pitches = cand_pitches[sounding]
            sounding_indices = cand_indices[sounding]
            min_pos = np.argmin(sounding_pitches)
            bass_idx_map[t] = sounding_indices[min_pos]

    df.loc[notes.index, "sounding_bass_idx"] = notes["onset"].map(bass_idx_map)

    return df
