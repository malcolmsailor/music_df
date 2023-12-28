import contextlib
import fractions
import inspect
import math
import os
import tempfile
import warnings

import mido
import pandas as pd
import pytest

from music_df import sort_df
from music_df.midi_parser import df_to_midi, midi_to_csv, midi_to_table

SCRIPT_DIR = os.path.dirname((os.path.realpath(__file__)))

TRACK = 2
PITCH = 4
ONSET = 5
RELEASE = 6

PALMID = os.path.join(SCRIPT_DIR, "test_files", "misc_Palestrina.mid")

warnings.simplefilter("error")

# TODO test microtones
# Test the sort of file that would have invoked this warning:
# Can I deprecate or at least rewrite this warning?
# if num_tracks == 1:
#     warnings.warn(
#         "Midi files of just one track exported from Logic "
#         "don't put meta messages on a separate track. Support "
#         "for these is not yet implemented and there is likely to "
#         "be a crash very soon..."
#     )


@contextlib.contextmanager
def temp_midi(mido_file):
    # inspect.currentframe call from
    # https://stackoverflow.com/a/5067654/10155119
    output_path = os.path.join(
        SCRIPT_DIR,
        inspect.currentframe().f_back.f_back.f_code.co_name + ".mid",  # type:ignore
    )
    mido_file.save(output_path)
    try:
        yield output_path
    finally:
        os.remove(output_path)


def init_midi_file(n_tracks=1, type_=1):
    mid = mido.MidiFile(type=type_)
    if n_tracks != 1 and type_ == 0:
        raise ValueError
    for _ in range(n_tracks):
        track = mido.MidiTrack()
        mid.tracks.append(track)
    return mid


def test_orphan_note_on_at_end():
    mid = init_midi_file(type_=0)
    mid.tracks[0].extend(
        [
            mido.Message("note_on", note=60, time=0),
        ]
    )
    with temp_midi(mid) as output_path:
        with pytest.warns(UserWarning) as record:
            midi_to_table(output_path, time_type=float)
        assert len(record) == 1
        assert "Pitch 60 is still on" in record[0].message.args[0]  # type:ignore


def test_orphan_note_off():
    mid = init_midi_file(type_=0)
    mid.tracks[0].extend(
        [
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_off", note=60, time=64),
            mido.Message("note_off", note=60, time=64),
        ]
    )
    with temp_midi(mid) as output_path:
        with pytest.warns(UserWarning) as record:
            midi_to_table(output_path, time_type=float)
        assert len(record) == 1
        assert (
            "no note_on event still sounding"
            in record[0].message.args[0]  # type:ignore
        )


def test_simultaneous_note_on_and_off():
    mid = init_midi_file(type_=0)
    mid.tracks[0].extend(
        [
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_on", note=60, time=64),
            mido.Message("note_off", note=60, time=0),
            mido.Message("note_off", note=60, time=64),
        ]
    )
    with temp_midi(mid) as output_path:
        midi_to_table(output_path, time_type=float)


def test_consecutive_notes():
    # This test should run without any warnings or errors
    mid = init_midi_file(type_=0)
    mid.tracks[0].extend(
        [
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_on", note=60, time=64),
            mido.Message("note_off", note=60, time=0),
            mido.Message("note_off", note=60, time=64),
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_off", note=60, time=64),
        ]
    )
    with temp_midi(mid) as output_path:
        midi_to_table(output_path, time_type=float)


def test_overlapping_notes():
    # possible approaches to overlapping notes:
    #   1. a note_on event followed by a note_on event without an interceding
    #       note_off is understood implicitly as terminating the preceding
    #       note
    #   2. a note_off event that follows two note_on events terminates *both*
    #       note_on events
    #   3. a note_on that follows another note_on (without an interceding
    #       note_off) is ignored (don't do this, but it's at least a possible
    #       approach)
    #   4. a note_off that follows two note_ons terminates only one of them
    #       (the first one? the second one?)

    mid = init_midi_file(type_=0)
    mid.tracks[0].extend(
        [
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_on", note=60, time=64),
            mido.Message("note_off", note=60, time=64),
            mido.Message("note_off", note=60, time=64),
        ]
    )
    with temp_midi(mid) as output_path:
        with pytest.warns(UserWarning) as record:
            midi_to_table(output_path, time_type=float)
        assert len(record) == 1
        assert (
            "no note_on event still sounding"
            in record[0].message.args[0]  # type:ignore
        )
        result = midi_to_table(
            output_path,
            overlapping_notes="end_first",
            time_type=fractions.Fraction,
        )
        dur1 = result.iloc[0]["release"] - result.iloc[0]["onset"]
        dur2 = result.iloc[1]["release"] - result.iloc[1]["onset"]
        assert dur1 == dur2, "dur1 != dur2"
        result = midi_to_table(
            output_path,
            overlapping_notes="end_last",
            time_type=fractions.Fraction,
        )
        dur1 = result.iloc[0]["release"] - result.iloc[0]["onset"]
        dur2 = result.iloc[1]["release"] - result.iloc[1]["onset"]
        assert dur1 == dur2 * 3, "dur1 != dur2 * 3"


def test_midi_to_table():
    result = midi_to_table(PALMID)


def test_sort_order():
    def _get_i(iterable, lda):
        for i, item in enumerate(iterable):
            if lda(item):
                return i

    mid = init_midi_file(n_tracks=5, type_=1)
    mid.tracks[1].extend(
        [
            mido.Message("note_on", note=60, time=0),
            mido.Message("note_off", note=60, time=240),
        ]
    )
    mid.tracks[2].extend(
        [
            mido.Message("note_on", note=64, time=0),
            mido.Message("note_off", note=64, time=480),
        ]
    )
    mid.tracks[3].extend(
        [
            mido.Message("note_on", note=72, time=0),
            mido.Message("note_on", note=48, time=0),
            mido.Message("note_off", note=48, time=480),
            mido.Message("note_off", note=72, time=480),
        ]
    )
    mid.tracks[4].extend(
        [
            mido.Message("note_on", note=67, time=0),
            mido.Message("note_on", note=67, time=0),
            mido.Message("note_off", note=67, time=480),
            mido.Message("note_off", note=67, time=480),
        ]
    )
    with temp_midi(mid) as output_path:
        out = midi_to_table(
            output_path,
            time_type=float,
            overlapping_notes="end_first",
        )
    # shorter note in lower-indexed track goes after longer note in
    # higher-indexed track
    greater_i: int = out[(out.track == 1) & (out.onset == 0) & (out.pitch == 60)].index[
        0
    ]  # type:ignore
    lower_i: int = out[(out.track == 2) & (out.onset == 0) & (out.pitch == 64)].index[
        0
    ]  # type:ignore
    assert greater_i > lower_i
    # assert _get_i(
    #     out, lambda x: x[TRACK] == 1 and x[ONSET] == 0 and x[PITCH] == 60
    # ) > _get_i(out, lambda x: x[TRACK] == 2 and x[ONSET] == 0 and x[PITCH] == 64)
    # higher pitch goes before lower pitch when simultaneously attacked on same
    #   track
    lower_i: int = out[
        (out.track == 3) & (out.onset == 0) & (out.pitch == 72) & (out.release == 2)
    ].index[
        0
    ]  # type:ignore
    greater_i: int = out[
        (out.track == 3) & (out.onset == 0) & (out.pitch == 48) & (out.release == 1)
    ].index[
        0
    ]  # type:ignore
    assert greater_i > lower_i
    # assert _get_i(
    #     out,
    #     lambda x: x[TRACK] == 3
    #     and x[ONSET] == 0
    #     and x[PITCH] == 72
    #     and x[RELEASE] == 2,
    # ) < _get_i(
    #     out,
    #     lambda x: x[TRACK] == 3
    #     and x[ONSET] == 0
    #     and x[PITCH] == 48
    #     and x[RELEASE] == 1,
    # )
    # shorter note in track goes before longer simultaneous note in same track
    lower_i: int = out[(out.track == 4) & (out.onset == 0) & (out.release == 1)].index[
        0
    ]  # type:ignore
    greater_i: int = out[
        (out.track == 4) & (out.onset == 0) & (out.release == 2)
    ].index[
        0
    ]  # type:ignore
    assert greater_i > lower_i
    # assert _get_i(
    #     out, lambda x: x[TRACK] == 4 and x[ONSET] == 0 and x[RELEASE] == 1
    # ) < _get_i(out, lambda x: x[TRACK] == 4 and x[ONSET] == 0 and x[RELEASE] == 2)

    #
    # def _get_panda_i(df, lda):
    #     for i, row in df.iterrows():
    #         if lda(row):
    #             return i

    # with temp_midi(mid) as output_path:
    #     out_df = midi_to_table(
    #         output_path,
    #         time_type=float,
    #         overlapping_notes="end_first",
    #     )
    # # shorter note in higher-indexed track goes after longer note in
    # # lower-indexed track
    # assert _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 1 and x["onset"] == 0 and x["pitch"] == 60,
    # ) > _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 2 and x["onset"] == 0 and x["pitch"] == 64,
    # )
    # # higher pitch goes before lower pitch when simultaneously attacked on same
    # #   track
    # assert _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 3
    #     and x["onset"] == 0
    #     and x["pitch"] == 72
    #     and x["release"] == 2,
    # ) < _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 3
    #     and x["onset"] == 0
    #     and x["pitch"] == 48
    #     and x["release"] == 1,
    # )
    # # shorter note in track goes before longer simultaneous note in same track
    # assert _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 4 and x["onset"] == 0 and x["release"] == 1,
    # ) < _get_panda_i(
    #     out_df,
    #     lambda x: x["track"] == 4 and x["onset"] == 0 and x["release"] == 2,
    # )
    # # # A file that was giving me grief:
    # # in_path = os.path.join(SCRIPT_DIR, "test_files", "misc_Bach.mxl")
    # # _, mid_path = tempfile.mkstemp(suffix=".mid")
    # # score = music21.converter.parse(in_path, format="musicxml")
    # # score = score.stripTies()
    # # score.write("midi", mid_path)
    # # df = midi_to_table(
    # #     mid_path,
    # #     time_type=float,
    # #     overlapping_notes="end_first",
    # #     output_type="pandas",
    # #     pitch_sort_asc=True,
    # #     track_sort_asc=None,
    # # )
    # # os.remove(mid_path)
    # # df = df[df["type"] == "note"].reset_index(drop=True)
    # # sorted_df = df.sort_values(["onset", "pitch", "release"])
    # # assert (df.index == sorted_df.index).all()


def test_midi_to_csv():
    _, csv_path = tempfile.mkstemp(suffix=".csv")
    midi_to_csv(PALMID, csv_path)
    os.remove(csv_path)


def test_df_to_midi():
    orig_df = midi_to_table(PALMID)
    _, csv_path = tempfile.mkstemp(suffix=".csv")
    orig_df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)

    # "track" is a float if any items are nan; that can cause issues so we
    #   explicitly set it to nan here
    df["track"] = df.track.astype(float)

    _, mid_path = tempfile.mkstemp(suffix=".mid")
    df_to_midi(df, mid_path)
    df2 = midi_to_table(mid_path)
    os.remove(mid_path)
    df = df[df.type == "note"].reset_index(drop=True)
    df2 = df2[df2.type == "note"].reset_index(drop=True)
    df.drop(
        columns=["filename", "other", "type", "instrument", "label"],
        inplace=True,
        errors="ignore",
    )
    df2.drop(columns=["filename", "other", "type"], inplace=True, errors="ignore")
    df = sort_df(df)
    df2 = sort_df(df2)
    assert len(df) == len(df2)
    for (_, note1), (_, note2) in zip(df.iterrows(), df2.iterrows()):
        for name, val in note1.items():
            if isinstance(val, float):
                assert math.isclose(val, note2[name])  # type:ignore
            else:
                assert val == note2[name]  # type:ignore


if __name__ == "__main__":
    test_sort_order()
    test_orphan_note_on_at_end()
    test_orphan_note_off()
    test_simultaneous_note_on_and_off()
    test_consecutive_notes()
    test_overlapping_notes()
    test_midi_to_csv()
    test_df_to_midi()
