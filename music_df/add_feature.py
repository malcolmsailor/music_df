from functools import partial
import io  # Used by doctests
import re
from ast import literal_eval
from itertools import chain
from math import isnan
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from mido import tempo2bpm

from music_df.constants import NAME_TO_MIDI_INSTRUMENT
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

    assert (
        time_sigs[0].onset <= music_df[music_df.type == "note"].iloc[0].onset
    ), "There is no time signature before the first note; default time signature not yet implemented"

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
    # (Malcolm 2023-10-16) why are we sorting by onset and type here? Can we
    #       remove this?
    # out_df = out_df.sort_values(
    #     by="onset",
    #     axis=0,
    #     kind="mergesort",  # default sort is not stable
    #     ignore_index=False,
    # )
    # out_df = out_df.sort_values(
    #     by="type",
    #     axis=0,
    #     key=lambda col: col.where(col != "bar", "zzz"),
    #     kind="mergesort",  # default sort is not stable
    #     ignore_index=False,
    # )
    out_df = sort_df(out_df, ignore_index=False)

    out_df = out_df.reset_index(drop=not keep_old_index)

    return out_df


def make_time_signatures_explicit(
    music_df: pd.DataFrame, default_time_signature: dict[str, int] | None = None
) -> pd.DataFrame:
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
        tempo2bpm(_to_dict_if_necessary(d)["tempo"])
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
    music_df["time_sig_dur"] = float("nan")
    music_df.loc[music_df.type == "time_signature", "time_sig_dur"] = music_df[
        music_df.type == "time_signature"
    ].apply(_time_sig_dur_from_row, axis=1)
    music_df["time_sig_dur"] = music_df.time_sig_dur.ffill()
    return music_df


def add_bar_durs(music_df: pd.DataFrame) -> pd.DataFrame:
    bar_mask = music_df.type == "bar"
    if not bar_mask.any():
        raise ValueError(f"Score must have at least one bar")
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
    """Adds a "bar_number" column to the dataframe.

    There must be at least one row with type == "bar".
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
    if "velocity" not in music_df.columns:
        music_df["velocity"] = default_velocity
    else:
        music_df["velocity"] = music_df.velocity.fillna(value=default_velocity)
    return music_df


def add_default_midi_instrument(
    music_df: pd.DataFrame,
    default_instrument: int = 0,
) -> pd.DataFrame:
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
    if "track" not in music_df.columns:
        return add_default_midi_instrument(music_df, default_instrument)
    program_change_mask = music_df.type == "program_change"
    music_df["midi_instrument"] = float("nan")
    music_df.loc[program_change_mask, "midi_instrument"] = [
        _to_dict_if_necessary(d)["program"]
        for _, d in music_df.other[program_change_mask].items()
    ]

    grouped_by_track = music_df.groupby("track")
    accumulator = []
    for track, group_df in grouped_by_track:
        group_df["midi_instrument"] = group_df.midi_instrument.ffill()
        accumulator.append(group_df)

    out_df = pd.concat(accumulator)
    out_df = out_df.sort_index(axis=0)

    out_df = add_default_midi_instrument(out_df, default_instrument=default_instrument)
    out_df["midi_instrument"] = out_df.midi_instrument.astype(int)
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
    Key signatures are between -5 and 6.
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
