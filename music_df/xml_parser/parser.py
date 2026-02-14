from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import typing as t
import warnings
import xml.sax
from collections import defaultdict
from dataclasses import asdict
from enum import Enum, auto
from fractions import Fraction
from functools import reduce
from types import MappingProxyType
from zipfile import ZipFile

import pandas as pd

from music_df.sort_df import sort_df

try:
    from cache_lib import cacher  # type:ignore
except ImportError:

    def cacher():
        def identity_wrap(f):
            return f

        return identity_wrap


from music_df.xml_parser.objects import (
    LIMIT_DENOMINATOR,
    Measure,
    Note,
    Tempo,
    Text,
    TimeSignature,
)
from music_df.xml_parser.repeats import get_repeat_segments
from music_df.xml_parser.ties import merge_ties

# TODO I believe this parser only handles "partwise" and not "timewise"
#   scores.

LOGGER = logging.getLogger(__name__)

RepeatOptions = t.Literal["yes", "no", "drop", "max2"]


class XMLParseException(Exception):
    pass


class State(Enum):
    NULL = auto()
    PART_LIST = auto()
    SCORE_PART = auto()
    PART_NAME = auto()
    MIDI_INSTRUMENT = auto()
    MIDI_PROGRAM = auto()
    PART = auto()
    DIVISIONS = auto()
    MEASURE = auto()
    NOTE = auto()
    DUR = auto()
    PITCH = auto()
    STEP = auto()
    OCTAVE = auto()
    ALTER = auto()
    FORWARD = auto()
    BACKUP = auto()
    VOICE = auto()
    BARLINE = auto()
    REPEAT = auto()
    SOUND = auto()
    TIME = auto()
    BEATS = auto()
    BEAT_TYPE = auto()
    FIGURED_BASS = auto()
    DIRECTION = auto()
    WORDS = auto()
    MOVEMENT_TITLE = auto()
    TRANSPOSE = auto()
    DIATONIC = auto()
    CHROMATIC = auto()
    UNPITCHED = auto()


WHITE_KEYS = MappingProxyType({"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11})

WARNED_RE_GRACE_NOTES = False


class MusicXmlHandler(xml.sax.ContentHandler):
    _init_and_end = {
        "part_list",
        "part_name",
        "score_part",
        "midi_instrument",
        "midi_program",
        "part",
        "measure",
        "note",
        "duration",
        "pitch",
        "step",
        "octave",
        "alter",
        "divisions",
        "forward",
        "backup",
        "voice",
        "barline",
        "time",
        "beats",
        "beat_type",
        "figured_bass",
        "direction",
        "words",
        "movement_title",
        "transpose",
        "chromatic",
        "unpitched",
        "diatonic",
    }
    _process = {
        "rest",
        "tie",
        "chord",
        "grace",
        "repeat",
        "ending",
        "cue",
        "sound",
    }

    def __init__(
        self,
        expand_repeats: RepeatOptions = "yes",
        max_measure_overhang_to_trim: float | Fraction = Fraction(1, 4),
    ):
        super().__init__()
        self._expand_repeats_flag = expand_repeats
        self._parts = []
        self._part_info = defaultdict(dict)
        self._current_part: t.Optional[t.List[Note]] = None
        self._current_part_number: int = 0
        self._current_part_id = None
        self._now: t.Optional[Fraction] = None
        self._current_note: t.Optional[Note] = None
        self._current_dur: t.Optional[Fraction] = None
        self._current_pitch = None
        self._divisions: t.Optional[int] = None
        self._state_stack: t.List[State] = [State.NULL]
        self._parsed = False
        self._chord: t.Optional[bool] = None
        self._time_shift: t.Optional[Fraction] = None
        self._measure_num: t.Optional[int] = None
        self._measures: t.Optional[t.List[Measure]] = None
        self._measure_i: t.Optional[int] = None
        self._measure_ends: defaultdict[int, defaultdict[int, Fraction]] = defaultdict(
            lambda: defaultdict(lambda: Fraction(-1024))
        )
        self._char_accumulator: t.List[str] = []
        self._repeats: t.List[t.DefaultDict[int, t.Dict[str, t.Dict[str, t.Any]]]] = []
        self._special_repeat_symbols: t.DefaultDict[int, t.Dict[str, t.Any]] = (
            defaultdict(dict)
        )
        self._current_part_repeats: t.Optional[
            t.DefaultDict[int, t.Dict[str, t.Dict[str, t.Any]]]
        ] = None
        self._repeats_have_been_expanded = False
        self._measure_rests = defaultdict(set)
        self._cue = False
        self._ts_numer: t.Optional[int] = None
        self._ts_denom: t.Optional[int] = None
        self._time_sigs: t.Optional[t.List[TimeSignature]] = None
        self._time_sig_i: t.Optional[int] = None
        self._current_time_sig_dur: t.Optional[Fraction] = None
        self._tempi: t.Optional[t.List[Tempo]] = None
        self._tempo_i: t.Optional[int] = None
        self._words: t.List[Text] = []
        self._max_measure_overhang_to_trim: Fraction = Fraction(
            max_measure_overhang_to_trim
        ).limit_denominator(LIMIT_DENOMINATOR)

    def _advance(self):
        assert self._now is not None and self._time_shift is not None
        self._now = (self._now + self._time_shift).limit_denominator(LIMIT_DENOMINATOR)
        self._time_shift = None

    def _init_part(self, attrs):
        assert self._state_stack[-1] is State.NULL
        assert self._chord is None
        assert self._measure_num is None
        self._state_stack.append(State.PART)
        self._now = Fraction(0)
        self._time_shift = Fraction(0)
        self._current_part = []
        if not self._current_part_number:
            self._measures = []
            self._time_sigs = []
            self._tempi = []
        self._measure_i = 0
        self._time_sig_i = 0
        self._current_time_sig_dur = None
        self._tempo_i = 0
        self._measure_num = 0
        self._part_measure_ends = self._measure_ends[self._current_part_number]
        self._at_measure_start = False
        self._current_part_repeats = defaultdict(dict)
        self._current_part_id = attrs["id"]

    def _end_part(self):
        assert self._state_stack[-1] is State.PART
        assert self._current_part_repeats is not None
        self._state_stack.pop()
        self._parts.append(self._current_part)
        self._current_part = None
        self._now = None
        self._current_part_number += 1
        self._measure_num = None
        self._repeats.append(self._current_part_repeats)
        self._current_part_repeats = None
        self._current_part_id = None

    def _init_measure(self, attrs):
        assert (
            self._measure_num is not None
            and self._measures is not None
            and self._measure_i is not None
        )
        assert self._state_stack[-1] is State.PART
        self._state_stack.append(State.MEASURE)
        self._measure_num += 1
        self._at_measure_start = True
        if not self._current_part_number:
            self._measures.append(Measure(onset=self._now))
        else:
            assert self._measures[self._measure_i].onset == self._now

    def _end_measure(self):
        assert self._state_stack[-1] is State.MEASURE
        assert (
            self._measure_num is not None
            and self._measures is not None
            and self._current_part_repeats is not None
        )
        self._state_stack.pop()

        # MusicXML defaults to 4/4 when no time signature is specified
        if self._current_time_sig_dur is None:
            default_ts = TimeSignature(
                onset=Fraction(0), numer=4, denom=4
            )
            self._current_time_sig_dur = default_ts.quarter_duration
            if not self._current_part_number:
                self._time_sigs.append(default_ts)

        # It can occur that, due to rounding errors with complex tuplets, a measure
        #   might be somewhat too long, causing misalignment with the other parts and
        #   leading to a parsing failure. We simply check if the measure is within a
        #   certain distance of the expected length (according to the time signature)
        #   and, if so, set it to the expected length.

        expected_release = (
            self._measures[self._measure_i].onset + self._current_time_sig_dur
        )

        actual_release = self._measure_ends[self._current_part_number][
            self._measure_num
        ]

        if (
            self._max_measure_overhang_to_trim
            >= abs(actual_release - expected_release)
            > 0
        ):
            most_recent_note = self._current_part[-1]

            # In case the final note or notes lie entirely outside of the measure
            while most_recent_note.onset > expected_release:
                self._current_part.pop()
                most_recent_note = self._current_part[-1]

            # If the note is too long, trim it
            if most_recent_note.release > expected_release:
                most_recent_note.release = expected_release

            # Update the measure end
            self._measure_ends[self._current_part_number][
                self._measure_num
            ] = expected_release

        try:
            # ensure that the measure has the same length in all parts
            assert (
                len(
                    {
                        measure_ends[self._measure_num]
                        for measure_ends in self._measure_ends.values()
                    }
                )
                == 1
            )
        except AssertionError:
            # if the disagreement is because of a measure rest in one or more
            # parts, adjust those parts to have the same length
            if not any(
                (self._measure_num in measure_rests)
                for measure_rests in self._measure_rests.values()
            ):
                raise
            # TODO actually, this is going to be complicated: we need to
            #   handle the case where the measure rest was in a previous
            #   part and then we continued (because we set self._now to
            #   the end of the measure below). In that case, we need
            #   to increment *all subsequent notes and other items in
            #   that part*!
            # For now, we can just pass if we are in the very last measure
            if not self._measure_num == max(self._measure_ends[0]):
                raise
            # TODO actually I think musescore is smart enough to adjust
            #   the length of measure rests... I believe this is not the
            #   problem I thought it was.
        # Some musicxml files seem not to bother to "fill in" the rest of the
        #   measure after using "backup" to add notes (which may or may not
        #   extend to the end of the measure). We could take the measure length
        #   from the notated time signature but I don't think there's any
        #   guarantee that those will correspond to the duration of the notes in
        #   the measure either, so I think it's best to just take the furthest
        #   we've gotten in the measure as the end of the measure.
        self._now = self._part_measure_ends[self._measure_num]
        if not self._current_part_number:
            self._measures[-1].release = self._now
        # ensure that any repeats are the same across all parts
        if self._measure_num in self._current_part_repeats:
            assert all(
                self._current_part_repeats[self._measure_num]
                == repeats[self._measure_num]
                for repeats in self._repeats
            )
        self._measure_i += 1

    def _init_barline(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.BARLINE)

    def _end_barline(self):
        assert self._state_stack[-1] is State.BARLINE
        self._state_stack.pop()

    def _init_divisions(self, attrs):
        self._state_stack.append(State.DIVISIONS)

    def _end_divisions(self):
        assert self._state_stack[-1] is State.DIVISIONS
        self._state_stack.pop()

    def _init_note(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.NOTE)
        self._current_note = Note(
            part=self._current_part_id,
            instrument=self._part_info[self._current_part_id].get("instrument", None),
            midi_instrument=self._part_info[self._current_part_id].get(
                "midi_instrument", None
            ),
        )
        self._chord = False
        self._cue = False

    def _end_note(self):
        assert self._state_stack[-1] is State.NOTE
        self._state_stack.pop()
        if not self._current_note.grace:
            self._current_note.dur = self._current_dur
            self._time_shift = self._current_dur
            self._current_dur = None
        assert self._current_note.is_valid()
        if not self._current_note.grace and (
            self._current_note.release > self._part_measure_ends[self._measure_num]
        ):
            self._part_measure_ends[self._measure_num] = self._current_note.release
        if not self._cue:
            self._current_part.append(self._current_note)
        self._current_note = None
        self._chord = None

    def _init_duration(self, attrs):
        assert self._state_stack[-1] in (
            State.NOTE,
            State.FORWARD,
            State.BACKUP,
            State.FIGURED_BASS,
        )
        assert self._current_dur is None
        self._state_stack.append(State.DUR)

    def _end_duration(self):
        assert self._state_stack[-1] is State.DUR
        self._state_stack.pop()
        if self._state_stack[-1] is State.FORWARD:
            self._now = (self._now + self._current_dur).limit_denominator(
                LIMIT_DENOMINATOR
            )
            if self._now > self._part_measure_ends[self._measure_num]:
                self._part_measure_ends[self._measure_num] = self._now
            self._current_dur = None
        elif self._state_stack[-1] is State.BACKUP:
            self._now = (self._now - self._current_dur).limit_denominator(
                LIMIT_DENOMINATOR
            )
            self._current_dur = None
        elif self._state_stack[-1] is State.FIGURED_BASS:
            LOGGER.debug(f"ignoring duration of figured bass")
            self._current_dur = None

        # Otherwise, do nothing
        else:
            assert self._state_stack[-1] in {State.NOTE}

    def _init_pitch_onset_handler(self):
        if not self._chord and not self._current_note.grace:
            if self._at_measure_start:
                self._at_measure_start = False
            else:
                self._advance()
        self._current_note.onset = self._now

    def _init_pitch(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        self._state_stack.append(State.PITCH)
        self._current_pitch = {}
        # <chord/> elements (indicating that we should not advance)
        #   come before <pitch> elements; thus if self._chord is False
        #   we advance. Either way, we can now initialize the note onset.

        self._init_pitch_onset_handler()

    def _end_pitch(self):
        assert self._state_stack[-1] is State.PITCH
        self._state_stack.pop()
        self._handle_pitch()
        self._current_pitch = None

    def _init_unpitched(self, attrs):
        # Because we initialize note onsets in `_init_pitch`, we need
        #   to do likewise for unpitched notes here. However, for my purposes
        #   we probably want to omit unpitched notes.
        assert self._state_stack[-1] is State.NOTE
        self._state_stack.append(State.UNPITCHED)
        self._current_note.unpitched = True
        self._init_pitch_onset_handler()

    def _end_unpitched(self):
        assert self._state_stack[-1] is State.UNPITCHED
        self._state_stack.pop()

    def _init_step(self, attrs):
        assert self._state_stack[-1] is State.PITCH
        self._state_stack.append(State.STEP)

    def _end_step(self):
        assert self._state_stack[-1] is State.STEP
        self._state_stack.pop()

    def _init_octave(self, attrs):
        assert self._state_stack[-1] is State.PITCH
        self._state_stack.append(State.OCTAVE)

    def _end_octave(self):
        assert self._state_stack[-1] is State.OCTAVE
        self._state_stack.pop()

    def _init_alter(self, attrs):
        assert self._state_stack[-1] is State.PITCH
        self._state_stack.append(State.ALTER)

    def _end_alter(self):
        assert self._state_stack[-1] is State.ALTER
        self._state_stack.pop()

    def _init_voice(self, attrs):
        assert self._state_stack[-1] in {State.NOTE, State.DIRECTION}
        self._state_stack.append(State.VOICE)

    def _end_voice(self):
        assert self._state_stack[-1] is State.VOICE
        self._state_stack.pop()

    def _init_forward(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.FORWARD)

    def _end_forward(self):
        assert self._state_stack[-1] is State.FORWARD
        self._state_stack.pop()

    def _init_backup(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.BACKUP)

    def _end_backup(self):
        assert self._state_stack[-1] is State.BACKUP
        self._state_stack.pop()

    def _init_time(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.TIME)

    def _end_time(self):
        assert self._state_stack[-1] is State.TIME
        self._state_stack.pop()
        if not self._current_part_number:
            time_sig = TimeSignature(
                onset=self._now, numer=self._ts_numer, denom=self._ts_denom
            )
            self._time_sigs.append(time_sig)
        else:
            time_sig = self._time_sigs[self._time_sig_i]
            assert time_sig.onset == self._now
            assert time_sig.numer == self._ts_numer
            assert time_sig.denom == self._ts_denom
            self._time_sig_i += 1
        self._current_time_sig_dur = time_sig.quarter_duration

    def _init_beats(self, attrs):
        assert self._state_stack[-1] is State.TIME
        self._state_stack.append(State.BEATS)

    def _end_beats(self):
        assert self._state_stack[-1] is State.BEATS
        self._state_stack.pop()

    def _init_beat_type(self, attrs):
        assert self._state_stack[-1] is State.TIME
        self._state_stack.append(State.BEAT_TYPE)

    def _end_beat_type(self):
        assert self._state_stack[-1] is State.BEAT_TYPE
        self._state_stack.pop()

    def _init_part_list(self, attrs):
        assert self._state_stack[-1] is State.NULL
        self._state_stack.append(State.PART_LIST)

    def _end_part_list(self):
        assert self._state_stack[-1] is State.PART_LIST
        self._state_stack.pop()

    def _init_score_part(self, attrs):
        assert self._state_stack[-1] is State.PART_LIST
        self._state_stack.append(State.SCORE_PART)
        self._current_part_id = attrs["id"]

    def _end_score_part(self):
        assert self._state_stack[-1] is State.SCORE_PART
        self._state_stack.pop()
        self._current_part_id = None

    def _init_part_name(self, attrs):
        assert self._state_stack[-1] is State.SCORE_PART
        self._state_stack.append(State.PART_NAME)

    def _end_part_name(self):
        assert self._state_stack[-1] is State.PART_NAME
        self._state_stack.pop()

    def _init_midi_instrument(self, attrs):
        assert self._state_stack[-1] is State.SCORE_PART
        self._state_stack.append(State.MIDI_INSTRUMENT)

    def _end_midi_instrument(self):
        assert self._state_stack[-1] is State.MIDI_INSTRUMENT
        self._state_stack.pop()

    def _init_midi_program(self, attrs):
        assert self._state_stack[-1] is State.MIDI_INSTRUMENT
        self._state_stack.append(State.MIDI_PROGRAM)

    def _end_midi_program(self):
        assert self._state_stack[-1] is State.MIDI_PROGRAM
        self._state_stack.pop()

    def _init_transpose(self, attrs):
        # (Malcolm 2023-12-30) <transpose> occurs within an <attributes> tag
        #   but we don't parse that. I'm omitting an assertion here.
        self._state_stack.append(State.TRANSPOSE)

    def _end_transpose(self):
        assert self._state_stack[-1] is State.TRANSPOSE
        self._state_stack.pop()

    def _init_diatonic(self, attrs):
        assert self._state_stack[-1] is State.TRANSPOSE
        self._state_stack.append(State.DIATONIC)

    def _end_diatonic(self):
        assert self._state_stack[-1] is State.DIATONIC
        self._state_stack.pop()

    def _init_chromatic(self, attrs):
        assert self._state_stack[-1] is State.TRANSPOSE
        self._state_stack.append(State.CHROMATIC)

    def _end_chromatic(self):
        assert self._state_stack[-1] is State.CHROMATIC
        self._state_stack.pop()

    def _init_figured_bass(self, attrs):
        assert self._state_stack[-1] is State.MEASURE
        self._state_stack.append(State.FIGURED_BASS)

    def _end_figured_bass(self):
        assert self._state_stack[-1] is State.FIGURED_BASS
        self._state_stack.pop()

    def _init_direction(self, attrs):
        self._state_stack.append(State.DIRECTION)

    def _end_direction(self):
        assert self._state_stack[-1] is State.DIRECTION
        self._state_stack.pop()

    def _init_words(self, attrs):
        self._state_stack.append(State.WORDS)

    def _end_words(self):
        assert self._state_stack[-1] is State.WORDS
        self._state_stack.pop()

    def _init_movement_title(self, attrs):
        self._state_stack.append(State.MOVEMENT_TITLE)

    def _end_movement_title(self):
        assert self._state_stack[-1] is State.MOVEMENT_TITLE
        self._state_stack.pop()

    def startElement(self, name, attrs):
        name = name.replace("-", "_")
        if name in self._init_and_end:
            getattr(self, "_init_" + name)(attrs)
        elif name in self._process:
            getattr(self, "_process_" + name)(attrs)

    def endElement(self, name):
        name = name.replace("-", "_")
        self._handle_chars()
        if name in self._init_and_end:
            getattr(self, "_end_" + name)()

    def characters(self, content):
        if self._state_stack[-1] in {
            State.PART_NAME,
            State.MIDI_PROGRAM,
            State.DUR,
            State.STEP,
            State.OCTAVE,
            State.ALTER,
            State.DIVISIONS,
            State.VOICE,
            State.BEATS,
            State.BEAT_TYPE,
            State.WORDS,
            State.MOVEMENT_TITLE,
            State.CHROMATIC,
            State.DIATONIC,
        }:
            self._char_accumulator.append(content)

    def _handle_chars(self):
        content = "".join(self._char_accumulator)
        self._char_accumulator.clear()
        if self._state_stack[-1] is State.DUR:
            self._current_dur = self._handle_duration(int(content))
        elif self._state_stack[-1] is State.STEP:
            self._current_pitch["step"] = content
        elif self._state_stack[-1] is State.OCTAVE:
            self._current_pitch["octave"] = int(content)
        elif self._state_stack[-1] is State.ALTER:
            self._current_pitch["alter"] = int(content)
        elif self._state_stack[-1] is State.DIVISIONS:
            self._divisions = int(content)
        elif self._state_stack[-1] is State.VOICE:
            if self._state_stack[-2] is State.NOTE:
                self._current_note.voice = content
        elif self._state_stack[-1] is State.BEATS:
            self._ts_numer = int(content)
        elif self._state_stack[-1] is State.BEAT_TYPE:
            self._ts_denom = int(content)
        elif self._state_stack[-1] is State.PART_NAME:
            assert self._current_part_id is not None
            self._part_info[self._current_part_id]["instrument"] = content
        elif self._state_stack[-1] is State.MIDI_PROGRAM:
            assert self._current_part_id is not None
            self._part_info[self._current_part_id]["midi_instrument"] = int(content)
        elif self._state_stack[-1] is State.WORDS:
            self._words.append(Text(self._now, content))
        elif self._state_stack[-1] is State.MOVEMENT_TITLE:
            self._words.append(Text(Fraction(0), f"Movement title: {content}"))
        elif self._state_stack[-1] is State.CHROMATIC:
            self._part_info[self._current_part_id]["chromatic_transpose"] = int(content)
        elif self._state_stack[-1] is State.DIATONIC:
            self._part_info[self._current_part_id]["diatonic_transpose"] = int(content)

    def endDocument(self):
        self._parsed = True

    def _handle_duration(self, ticks: int):
        return Fraction(ticks, self._divisions)

    def _handle_pitch(self):
        assert self._current_note is not None
        self._current_note.set_spelling(
            step=self._current_pitch["step"],
            alter=self._current_pitch.get("alter", 0),
            chromatic_transpose=self._part_info[self._current_part_id].get(
                "chromatic_transpose", None
            ),
            diatonic_transpose=self._part_info[self._current_part_id].get(
                "diatonic_transpose", None
            ),
        )
        self._current_note.pitch = (
            WHITE_KEYS[self._current_pitch["step"]]
            + self._current_pitch.get("alter", 0)
            + (self._current_pitch["octave"] + 1) * 12
        ) + self._part_info[self._current_part_id].get("chromatic_transpose", 0)

    def _process_rest(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        self._current_note.pitch = 0
        # as far as I know, rests can never be in chords
        assert self._chord is False
        if self._at_measure_start:
            self._at_measure_start = False
        else:
            self._advance()
        self._current_note.onset = self._now
        if attrs.get("measure") == "yes":
            self._measure_rests[self._current_part_number].add(self._measure_num)

    def _process_cue(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        self._cue = True

    def _process_chord(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        self._chord = True

    def _process_tie(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        if attrs["type"] == "start":
            self._current_note.tie_to_next = True
        elif attrs["type"] == "stop":
            self._current_note.tie_to_prev = True

    def _process_grace(self, attrs):
        assert self._state_stack[-1] is State.NOTE
        self._current_note.grace = True

    def _process_repeat(self, attrs):
        assert self._state_stack[-1] is State.BARLINE
        repeat_times = int(attrs.get("times", 2))
        if self._expand_repeats_flag == "max2":
            repeat_times = min(2, repeat_times)
        self._current_part_repeats[self._measure_num][attrs["direction"]] = {
            # The default is 2 for "play 2 times" (rather than "1" for "repeat
            # once")
            "times": repeat_times
        }

        if "after-jump" in attrs:
            # TODO?
            raise NotImplementedError

    def _process_ending(self, attrs):
        assert self._state_stack[-1] is State.BARLINE
        if attrs["type"] == "start":
            # We need to know where endings start but can otherwise ignore them.
            self._current_part_repeats[self._measure_num]["start-ending"] = {
                "number": attrs["number"]
            }

    def _add_tempo(self, bpm: float):
        if not self._current_part_number:
            self._tempi.append(Tempo(onset=self._now, bpm=bpm))
        else:
            tempo = self._tempi[self._tempo_i]
            assert tempo.onset == self._now
            assert tempo.bpm == bpm
        self._tempo_i += 1

    def _process_sound(self, attrs):
        assert self._measure_num is not None and self._current_part_repeats is not None
        if "forward-repeat" in attrs:
            if "forward" not in self._current_part_repeats[self._measure_num]:
                self._current_part_repeats[self._measure_num]["forward"] = {"times": 2}
        for attr in ("fine", "coda", "dacapo", "salsegno", "segno", "tocoda"):
            if attr in attrs:
                self._special_repeat_symbols[self._measure_num][attr] = attrs.get(
                    "time-only"
                )
        if "tempo" in attrs:
            assert self._now is not None
            self._add_tempo(float(attrs.get("tempo")))
        # if "fine" in attrs:
        #     pass
        # if "coda" in attrs:
        #     pass
        # if "dacapo" in attrs:
        #     pass
        # if "dalsegno" in attrs:
        #     pass
        # if "segno" in attrs:
        #     pass
        # if "time-only" in attrs:
        #     pass
        # if "tocoda" in attrs:
        #     pass

    def _expand_repeats(self, warn: bool = False):
        def _expand_sub(list_):
            expanded = []
            for (repeat_start, _), (orig_start, orig) in zip(
                repeated_segments, orig_segments
            ):
                offset = repeat_start - orig_start
                for item in list_:
                    if item.onset >= orig:
                        break
                    if item.onset >= orig_start:
                        new_item = item.copy()
                        new_item.onset += offset
                        if new_item.release is not None:
                            new_item.release += offset
                        expanded.append(new_item)
            return expanded

        assert not self._repeats_have_been_expanded
        repeats = self._repeats[0]
        measure_ends = self._measure_ends[0]
        orig_segments, repeated_segments, _ = get_repeat_segments(repeats, measure_ends, warn=warn)
        out = []
        for part in self._parts:
            out.append(_expand_sub(part))

        self._measures = _expand_sub(self._measures)
        self._time_sigs = _expand_sub(self._time_sigs)

        self._repeats_have_been_expanded = True
        self._parts = out

    def _drop_endings(self, warn: bool = False):
        assert not self._repeats_have_been_expanded

        repeats = self._repeats[0]
        measure_ends = self._measure_ends[0]
        orig_segments, _, segment_types = get_repeat_segments(repeats, measure_ends, warn=warn)
        prev_ending_num = None
        prev_ending_i = None
        to_remove = []
        for i, segment_type in enumerate(segment_types):
            m = re.match(r"^ending_(?P<num>\d+)$", segment_type)
            if not m:
                continue
            ending_num = int(m.group("num"))
            if prev_ending_num is not None and ending_num > prev_ending_num:
                to_remove.append(prev_ending_i)
            prev_ending_num = ending_num
            prev_ending_i = i

        out = []

        def _remove_endings_from_list(list_):
            # This is a quick-and-dirty implementation with pretty bad
            #   complexity
            offset = 0
            items_to_remove = set()
            offsets = []
            for segment_i in to_remove:
                start, stop = orig_segments[segment_i]
                offset += stop - start
                offsets.append((stop, offset))
                for i, item in enumerate(list_):
                    if item.onset >= stop:
                        break
                    elif item.onset >= start:
                        items_to_remove.add(i)
            offset = 0
            offset_i = 0
            out = []
            for i, item in enumerate(list_):
                if offset_i < len(offsets) and item.onset >= offsets[offset_i][0]:
                    offset = offsets[offset_i][1]
                    offset_i += 1
                if i in items_to_remove:
                    continue
                item.onset -= offset
                if item.release is not None:
                    item.release -= offset
                out.append(item)
            return out

        for part in self._parts:
            out.append(_remove_endings_from_list(part))

        self._measures = _remove_endings_from_list(self._measures)
        self._time_sigs = _remove_endings_from_list(self._time_sigs)

        self._parts = out
        self._repeats_have_been_expanded = True

    def get_df(self, sort=True, warn: bool = False) -> pd.DataFrame:
        expand_repeats = self._expand_repeats_flag
        assert self._parsed
        if (
            (expand_repeats != "no")
            and (not self._repeats_have_been_expanded)
            and any(repeats for repeats in self._repeats)
        ):
            if expand_repeats in ("yes", "max2"):
                self._expand_repeats(warn=warn)
            elif expand_repeats == "drop":
                self._drop_endings(warn=warn)
            else:
                raise ValueError(
                    f"expand_repeats must be in ('yes', 'max2', 'no', 'drop')"
                )

        no_rests = [[note for note in part if note.pitch] for part in self._parts]
        no_graces = [[note for note in part if not note.grace] for part in no_rests]
        if warn and (not WARNED_RE_GRACE_NOTES) and [len(l) for l in no_rests] != [
            len(l) for l in no_graces
        ]:
            warnings.warn("removing grace notes")
        merged_ties = []
        for i, part in enumerate(no_graces):
            merged_ties.append(merge_ties(part, warn=warn))
        # merged_ties = [merge_ties(part) for part in no_rests]
        all_parts = reduce(
            list.__add__,
            merged_ties + [self._measures, self._time_sigs, self._tempi, self._words],
        )
        df = pd.DataFrame([item.asdict() for item in all_parts])
        if not len(df):
            return df
        if "unpitched" in df.columns:
            # this note attribute is used only for internal validation of
            #   notes (because otherwise we expect the note's pitch attr to
            #   be non-null). For external use, the user can just check
            #   if the pitch attribute of a note is nan.
            df = df.drop("unpitched", axis=1)
        if "pitch" not in df.columns:
            # this occurs if there are no notes in the score
            df["pitch"] = float("nan")
        if sort:
            df = sort_df(df)
        return df


class MxlMetaHandler(xml.sax.ContentHandler):
    def __init__(self):
        super().__init__()
        self.musicxml_path: t.Optional[str] = None

    def startElement(self, name, attrs):
        if name == "rootfile" and self.musicxml_path is None:
            self.musicxml_path = attrs["full-path"]

    def endDocument(self):
        assert self.musicxml_path is not None


def parse_mxl_metadata(archive: ZipFile) -> str:
    handler = MxlMetaHandler()
    with archive.open("META-INF/container.xml") as inf:
        xml.sax.parse(inf, handler)
    return handler.musicxml_path


def parse_xml(fp, sort=True, expand_repeats: RepeatOptions = "yes", warn: bool = False):
    handler = MusicXmlHandler(expand_repeats=expand_repeats)
    xml.sax.parse(fp, handler)
    df = handler.get_df(sort=sort, warn=warn)
    return df


def parse_mxl(path, sort=True, expand_repeats="yes", warn: bool = False):
    archive = ZipFile(path)
    musicxml_path = parse_mxl_metadata(archive)
    with archive.open(musicxml_path) as inf:
        return parse_xml(inf, sort=sort, expand_repeats=expand_repeats, warn=warn)


@cacher()
def read_mscore(path) -> bytes:
    if not shutil.which("mscore"):
        raise ValueError("Can't find 'mscore' in path")
    _, musicxml_path = tempfile.mkstemp(suffix=".xml")
    try:
        subprocess.run(
            ["mscore", path, "-o", musicxml_path],
            check=True,
            capture_output=True,
        )
        with open(musicxml_path, "rb") as inf:
            data = inf.read()
    finally:
        os.remove(musicxml_path)
    return data


def parse_musescore(path, sort=True, expand_repeats: RepeatOptions = "yes", warn: bool = False):
    _, musicxml_path = tempfile.mkstemp(suffix=".xml")

    try:
        xml_bytes = read_mscore(path)
        with open(musicxml_path, "wb") as outf:
            outf.write(xml_bytes)
        return parse_xml(musicxml_path, sort=sort, expand_repeats=expand_repeats, warn=warn)
    finally:
        os.remove(musicxml_path)


def parse(
    path, sort=True, expand_repeats: RepeatOptions = "yes", warn: bool = False
) -> pd.DataFrame:
    """Parse a musicxml file, return a Pandas DataFrame.

    Args:
        path: path to musicxml file. Extension can be any of 'mscx', 'mxl', or
            'xml'.

    Keyword args:
        sort: whether to sort the DataFrame before returning.
        expand_repeats: string. One of
            "yes": repeats are expanded
            "no": no processing of repeats takes place
            "drop": non-final endings are dropped (e.g., 1st endings if there
                are 1st and 2nd endings).
            "max2": repeats take place but the `times` attribute is set to at most
                2.
            Default: "yes".
        warn: whether to emit warnings (default: False).

    Returns:
        None

    Raises:
        exceptions
    """
    try:
        if path.endswith(".mscx") or path.endswith(".mscz"):
            return parse_musescore(path, sort=sort, expand_repeats=expand_repeats, warn=warn)
        if path.endswith(".mxl"):
            return parse_mxl(path, sort=sort, expand_repeats=expand_repeats, warn=warn)
        else:
            return parse_xml(path, sort=sort, expand_repeats=expand_repeats, warn=warn)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise XMLParseException(f"Parsing {path} failed") from e
