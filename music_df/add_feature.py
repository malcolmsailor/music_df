from ast import literal_eval
from itertools import chain
from typing import Callable

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


def infer_barlines(music_df: pd.DataFrame) -> pd.DataFrame:
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

    out_df = pd.concat([music_df, barlines])
    out_df = out_df.sort_values(
        by="onset", axis=0, kind="mergesort"  # default sort is not stable
    )
    out_df = out_df.sort_values(
        by="type",
        axis=0,
        key=lambda col: col.where(col != "bar", "zzz"),
        kind="mergesort",  # default sort is not stable
    )
    out_df = sort_df(out_df)
    out_df = out_df.reset_index(drop=True)
    return out_df


def make_time_signatures_explicit(
    music_df: pd.DataFrame, default_time_signature: dict[str, int] | None = None
) -> pd.DataFrame:
    if default_time_signature is None:
        default_time_signature = {"numerator": 4, "denominator": 4}
    # music_df["time_signature_numerator"]
    # music_df["time_signature_denominator"]
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

    music_df["ts_numerator"] = music_df.ts_numerator.fillna(method="ffill")
    music_df["ts_denominator"] = music_df.ts_denominator.fillna(method="ffill")

    music_df["ts_numerator"] = music_df.ts_numerator.fillna(
        value=default_time_signature["numerator"]
    )
    music_df["ts_denominator"] = music_df.ts_denominator.fillna(
        value=default_time_signature["denominator"]
    )
    music_df["ts_numerator"] = music_df.ts_numerator.astype(int)
    music_df["ts_denominator"] = music_df.ts_denominator.astype(int)
    return music_df


def make_tempos_explicit(music_df: pd.DataFrame, default_tempo: float) -> pd.DataFrame:
    tempo_mask = music_df.type == "set_tempo"
    music_df["tempo"] = float("nan")
    music_df.loc[tempo_mask, "tempo"] = [
        tempo2bpm(_to_dict_if_necessary(d)["tempo"])
        for _, d in music_df[tempo_mask].other.items()
    ]
    music_df["tempo"] = music_df.tempo.fillna(method="ffill")
    music_df["tempo"] = music_df.tempo.fillna(value=default_tempo)
    return music_df


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
    bar_mask = music_df.type == "bar"
    if not len(bar_mask):
        raise ValueError("No bars found")

    music_df = number_bars(music_df, initial_bar_number)
    music_df["bar_number"] = music_df["bar_number"].fillna(method="ffill")
    music_df["bar_number"] = music_df["bar_number"].fillna(value=default_bar_number)
    music_df["bar_number"] = music_df.bar_number.astype(int)
    return music_df


def get_bar_relative_onset(music_df: pd.DataFrame) -> pd.DataFrame:
    bar_mask = music_df.type == "bar"
    if not len(bar_mask):
        raise ValueError("No bars found")
    music_df["bar_onset"] = float("nan")
    music_df.loc[bar_mask, "bar_onset"] = music_df.onset[bar_mask]
    music_df["bar_onset"] = music_df.bar_onset.fillna(method="ffill")

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
    default_instrument: int = 0,  # TODO: (Malcolm 2023-08-22) verify
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
    program_change_mask = music_df.type == "program_change"
    music_df["midi_instrument"] = float("nan")
    music_df.loc[program_change_mask, "midi_instrument"] = [
        _to_dict_if_necessary(d)["program"]
        for _, d in music_df.other[program_change_mask].items()
    ]

    grouped_by_track = music_df.groupby("track")
    accumulator = []
    for track, group_df in grouped_by_track:
        group_df["midi_instrument"] = group_df.midi_instrument.fillna(method="ffill")
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
