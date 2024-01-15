import collections
import csv
import fractions
import os
import re
import warnings
from ast import literal_eval
from math import isnan
from numbers import Number
from typing import List, Optional, Tuple, Type, Union

import mido
import pandas as pd

from music_df.sort_df import sort_df

NUM_CHANNELS = 16


class MidiError(Exception):
    pass


def _convert_to_abs_time_and_sort(in_mid):
    def _sorter_string(msg):
        # We want to put pitchwheel events before any note events that occur
        # simultaneously
        if msg.type == "pitchwheel":
            return "aaaa"
        return msg.type

    for track in in_mid.tracks:
        tick_time = 0
        for msg in track:
            tick_time += msg.time
            # NB Mido docs say it is preferred to treat messages as immutable
            #   but I don't see the advantage in creating a bunch of copies
            #   here
            msg.time = tick_time
        # pitchwheel before note_off before note_on
        track.sort(key=_sorter_string)
        track.sort(key=lambda msg: msg.time)


def midi_to_table(
    in_midi_fname,
    time_type: Type = float,
    max_denominator: int = 8192,
    overlapping_notes: str = "end_all",
    pb_tup_dict: Optional[dict] = None,
    display_name: str | None = None,
    notes_only: bool = False,
    warn_for_orphan_note_offs: bool = True,
) -> pd.DataFrame:
    """Read a midi file and return a pandas DataFrame.

    Note-on and note-off events will be compiled into a single event with
    attack and release.

    Args:
        in_midi_fname: path to input midi file.

    Keyword args:
        time_type: the numeric type that should be used to express time
            attributes. The default is `float`, but if preserving exact relative
            timings is important it may be better to use `fractions.Fraction`
            to avoid float rounding issues. If `int`, then returned in ticks per beat.
        max_denominator: integer. Only has an effect if `time_type` is
            `fractions.Fraction`, in which case this argument sets the
            maximum denominator.
            Default: 8192
        overlapping_notes: string. Defines what happens when more than one
            note-on event on the same pitch occurs on the same track and
            channel, with no intervening note-off event. Ideally there should
            not be any such events, but that cannot be guaranteed. To
            illustrate the effect of the possible values, imagine there is a
            midi file with note-on events at time 0 and time 1, and note-off
            events at time 2 and time 3, with all events having the same pitch
            (e.g., midi number 60).
            Possible values:
                "end_all" (default): a note-off event ends *all* sounding
                    note-on events with the same track and channel and pitch.
                    Thus in the example there would be two notes:
                        - onset 0, release 2
                        - onset 1, release 2
                    There would be an "orphan" note-off event at time 3, which
                    will emit a warning.
                    This setting corresponds most closely to how midi playback
                    ordinarily behaves.
                "end_first": a note-off event ends the *first* sounding note-on
                    event with the same track and channel and pitch. Thus the
                    example would produce the following two notes:
                        - onset 0, release 2
                        - onset 1, release 3
                "end_last": a note-off event ends the *last* sounding note-on
                    event with the same track and channel and pitch. Thus the
                    example would produce the following two notes:
                        - onset 0, release 3
                        - onset 1, release 2
        pb_tup_dict: dictionary. TODO document.
        display_name: the value of the "filename" column in the returned dataframe. If
            not passed, uses in_midi_fname.
    Returns: a dataframe "events".
        - note events are the combination of a note-on with the following
            note-off message.
        - all other midi messages map to a single event.

        All events have onset, release, channel, pitch, velocity,
        and 'other' fields. pitch, velocity, and duration are null
        for non-note events; channel is null for events that do not have a
        channel attribute; all other fields go into "other" as a string
        representation.

        Tracks and channels are zero-indexed.

        Simultaneous note-on and note-off events on the same pitch will be
        interpreted as a note off *followed* by a note on event. (I.e., we
        understand repeated notes on the same pitch (and expect a preceding
        note-on and a following note-off), rather than a note with zero length)

        Output will be sorted with `sort_df()` function. I believe (haven't double
        checked recently) this sorts notes as follows:
            - by onset
            - by type (alphabetical, except for note events, which go after all
                other message types)
            - then by track number, from highest to lowest (the logic of this
                order being that we expect bass voice to be last)
            - then by pitch, from highest to lowest
            - then by time of release

        (Except for the last criterion (which is unlikely to apply and indeed
        can only apply if overlapping_notes has a value other than "end_all")
        and "type", this sort order is taken from Andie's script.)

    Raises:
        exceptions
    """

    # TODO flag to control behavior when sustain pedal is depressed
    if display_name is None:
        display_name = os.path.basename(in_midi_fname)

    def _event(
        type_,
        track=None,
        channel=None,
        pitch=None,
        onset=None,
        release=None,
        velocity=None,
        other=None,
    ):
        return {
            "filename": display_name,
            "type": type_,
            "track": track,
            "channel": channel,
            "pitch": pitch,
            "onset": onset,
            "release": release,
            "velocity": velocity,
            "other": other,
        }

    def _get_time(tick_time):
        if time_type == int:
            return tick_time
        if time_type == fractions.Fraction:
            return fractions.Fraction(tick_time, ticks_per_beat).limit_denominator(
                max_denominator=max_denominator
            )
        return time_type(tick_time / ticks_per_beat)

    def _pitch_bend_handler(track_pb_dict, msg):
        track_pb_dict[msg.channel] = msg.pitch

    def _note_on_handler(msg, track_note_on_dict, track_pb_dict):
        if track_pb_dict is None:
            # msg.note as key is a midinumber (0-127)
            # msg.note as second item of tuple is a pitch number (which depends
            #   in principle on the temperament)
            track_note_on_dict[msg.channel][msg.note].append((msg, msg.note))
            return
        pitch_bend = 0 if channel not in track_pb_dict else track_pb_dict[channel]
        # TODO document that an error will be raised if this is not found
        pitch = inverse_pb_tup_dict[(msg.note, pitch_bend)]
        track_note_on_dict[msg.channel][msg.note].append((msg, pitch))

    def _note_off_handler(msg, track_i, track_note_on_dict):
        midinum = msg.note
        out = []
        while True:
            try:
                note_on_msg, pitch = track_note_on_dict[msg.channel][midinum].pop(
                    0 if overlapping_notes == "end_first" else -1
                )
            except IndexError:
                if out:
                    return out
                if msg.note != 0 and warn_for_orphan_note_offs:
                    # Some midi writers seem to place note_off w/ pitch 0
                    #   at end of track (e.g., hum2mid) so we ignore those
                    warnings.warn(
                        f"note_off event with pitch {msg.note} at "
                        f"time {_get_time(msg.time)} "
                        f"on track {track_i}, "
                        f"channel {msg.channel}, but no note_on event still sounding"
                    )
                return None

            onset = _get_time(note_on_msg.time)
            release = _get_time(msg.time)
            out.append(
                _event(
                    type_="note",
                    track=track_i,
                    channel=msg.channel,
                    pitch=pitch,
                    onset=onset,
                    release=release,
                    velocity=note_on_msg.velocity,
                )
            )
            if overlapping_notes != "end_all":
                return out

    def _other_msg_handler(msg, track_i):
        other = vars(msg)
        onset = _get_time(other.pop("time"))
        type_ = other.pop("type")
        try:
            channel = other.pop("channel")
        except KeyError:
            channel = None
        return _event(
            type_=type_,
            track=track_i,
            channel=channel,
            onset=onset,
            other=other,
        )

    # Sorting the tracks avoids orphan note or pitchwheel events.
    try:
        in_mid = mido.MidiFile(in_midi_fname)
    except Exception as exc:
        raise MidiError(f"unable to read file {in_midi_fname}") from exc
    _convert_to_abs_time_and_sort(in_mid)
    num_tracks = len(in_mid.tracks)
    ticks_per_beat = in_mid.ticks_per_beat
    out = []

    note_on_dict = {
        i: {j: collections.defaultdict(list) for j in range(NUM_CHANNELS)}
        for i in range(num_tracks)
    }
    if pb_tup_dict is not None:
        pitch_bend_dict = {
            i: {j: {} for j in range(NUM_CHANNELS)} for i in range(num_tracks)
        }
        inverse_pb_tup_dict = {v: k for k, v in pb_tup_dict.items()}

    # messages don't have a track attribute, so (as far as I can tell) the only
    # way to assign each message to a track is to iterate over the tracks
    # and then combine and sort afterwards. This seems inefficient (but maybe
    # it's what mido is doing under the hood anyway when we iterate over the
    # whole file?)
    for track_i, track in enumerate(in_mid.tracks):
        if pb_tup_dict is None:
            track_pb_dict = None
        else:
            track_pb_dict = pitch_bend_dict[track_i]  # type:ignore
        for msg in track:
            if msg.type == "pitchwheel" and track_pb_dict is not None:
                _pitch_bend_handler(track_pb_dict, msg)
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                notes = _note_off_handler(msg, track_i, note_on_dict[track_i])
                if notes:
                    out.extend(notes)
            elif msg.type == "note_on":
                _note_on_handler(msg, note_on_dict[track_i], track_pb_dict)
            elif not notes_only:
                out.append(_other_msg_handler(msg, track_i))

    for track_i, track in note_on_dict.items():
        for channel_i, channel in track.items():
            try:
                assert not channel
            except AssertionError:
                for pitch, note_ons in channel.items():
                    try:
                        assert not note_ons
                    except AssertionError:
                        warnings.warn(
                            f"Pitch {pitch} is still on (no note-off event on "
                            f"track {track_i}, channel {channel_i} before end)"
                        )

    df = pd.DataFrame(out)
    return sort_df(df)


def midi_to_csv(in_midi_fname, out_csv_fname, *args, **kwargs):
    df = midi_to_table(in_midi_fname, *args, **kwargs)
    df.to_csv(out_csv_fname, index=False)


def midi_dirs_to_csv(list_of_dirs, out_csv_fname):
    raise NotImplementedError("I need to update to pandas version of midi_to_table")
    # we sort so that order of items will be in deterministic order
    list_of_dirs = sorted(list_of_dirs)
    with open(out_csv_fname, "w") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            (
                "file",
                "type",
                "track",
                "channel",
                "pitch",
                "onset",
                "release",
                "velocity",
                "other",
            )
        )
        for dir in list_of_dirs:
            fnames = [f for f in os.listdir(dir) if f.endswith(".mid")]
            fnames.sort()
            for fname in fnames:
                print(fname)
                rows = midi_to_table(os.path.join(dir, fname), header=False)
                csvwriter.writerows(rows)


def midi_files_to_csv(list_of_files, out_csv_fname, append=False):
    raise NotImplementedError("I need to update to pandas version of midi_to_table")
    # we sort so that order of items will be in deterministic order
    list_of_files = sorted(list_of_files)
    mode = "a" if append else "w"
    with open(out_csv_fname, mode) as csvfile:
        csvwriter = csv.writer(csvfile)
        if not append:
            csvwriter.writerow(
                (
                    "file",
                    "type",
                    "track",
                    "channel",
                    "pitch",
                    "onset",
                    "release",
                    "velocity",
                    "other",
                )
            )
        for fname in list_of_files:
            try:
                rows = midi_to_table(fname, header=False)
            except mido.KeySignatureError:
                print(f"Skipping {fname} due to KeySignatureError")
            except EOFError:
                print(f"Skipping {fname} due to EOFError")
            csvwriter.writerows(rows)


def _to_dict_if_necessary(d):
    if isinstance(d, str):
        return literal_eval(d)
    return d


def df_to_midi(
    df: pd.DataFrame,
    midi_path: str,
    ts: Optional[Union[str, Tuple[int, int]]] = None,
) -> None:
    """Inverse function of midi_to_table.

    For now, at least, ignores everything except notes, time_signatures,
    pitchwheel, and text events (for annotations).

    If the dataframe has a "track" column, assumes that track 0 will be for
    global events and notes will be on tracks 1--n where n is the number of
    tracks.
    """
    mid = mido.MidiFile()

    if "track" in df.columns:
        n_tracks = int(df.track.max()) + 1
    else:
        n_tracks = 1
    for _ in range(n_tracks):
        mid.tracks.append(mido.MidiTrack())
    if ts is not None:
        if isinstance(ts, str):
            m = re.match(r"(?P<numer>\d+)/(?P<denom>\d+)$", ts)
            assert m is not None
            numer = int(m.group("numer"))
            denom = int(m.group("denom"))
        else:
            numer, denom = ts
        mid.tracks[0].append(
            mido.MetaMessage("time_signature", numerator=numer, denominator=denom)
        )
    if "type" not in df.columns:
        # if the df does not have a "type" column, assume all rows are
        # notes
        df["type"] = "note"
    for _, event in df.iterrows():
        track_i = (
            0
            if (
                n_tracks == 1 or (isinstance(event.track, float) and isnan(event.track))
            )
            else int(event.track)
        )
        event_type = event.type
        if event_type == "text":
            mid.tracks[0].append(
                mido.MetaMessage("text", text=event.text, time=event.onset)
            )
        elif event_type == "time_signature":
            attrs = _to_dict_if_necessary(event.other)
            mid.tracks[0].append(
                mido.MetaMessage(
                    "time_signature",
                    numerator=attrs["numerator"],
                    denominator=attrs["denominator"],
                    time=event.onset,
                )
            )
        elif event_type == "set_tempo":
            # Midi tempo
            mid.tracks[0].append(
                mido.MetaMessage(
                    "set_tempo",
                    tempo=event.tempo
                    if hasattr(event, "tempo")
                    else _to_dict_if_necessary(event.other)["tempo"],
                )
            )
        elif event_type == "tempo":
            # BPM tempo
            if hasattr(event, "tempo"):
                midi_tempo = event.tempo
            else:
                midi_tempo = _to_dict_if_necessary(event.other)["tempo"]
            mid.tracks[0].append(
                mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(midi_tempo))
            )
        elif event_type == "pitchwheel":
            try:
                channel = int(event.channel)
            except AttributeError:
                channel = 0
            mid.tracks[track_i].append(
                mido.Message("pitchwheel", channel=channel, pitch=event.other["pitch"])
            )
        elif event_type == "note":
            try:
                channel = int(event.channel)
            except AttributeError:
                channel = 0
            try:
                velocity = int(event.velocity)
            except AttributeError:
                velocity = 64
            mid.tracks[track_i].append(
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=int(event.pitch),
                    velocity=velocity,
                    time=event.onset,
                )
            )
            mid.tracks[track_i].append(
                mido.Message(
                    "note_off",
                    channel=channel,
                    note=int(event.pitch),
                    velocity=velocity,
                    time=event.release,
                )
            )
    _abs_to_delta_times(mid)
    mid.save(filename=midi_path)


def _abs_to_delta_times(mf, skip=(), precision=12):
    for track_i, track in enumerate(mf.tracks):
        if track_i in skip:
            continue
        # it's important that note-offs don't go after note-ons that should be
        #   at the same instant. Numerical error with floats sometimes leads to
        #   that happening, so we round to precision when sorting here.
        track.sort(key=lambda x: round(x.time, precision))
        current_tick_time = 0  # unrounded
        for msg in track:
            abs_tick_time = mf.ticks_per_beat * msg.time
            delta_tick_time = abs_tick_time - current_tick_time
            msg.time = round(delta_tick_time)
            current_tick_time = abs_tick_time
