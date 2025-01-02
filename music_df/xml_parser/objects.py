from __future__ import annotations

import typing as t
from copy import copy
from dataclasses import asdict, dataclass
from fractions import Fraction

from mspell.transpose import transpose_spelling

LIMIT_DENOMINATOR = 64


@dataclass
class DFItem:
    def asdict(self):
        out = asdict(self)
        out["type"] = self._type
        return out

    def copy(self):
        return copy(self)


@dataclass
class Note(DFItem):
    pitch: t.Optional[int] = None
    onset: t.Optional[Fraction] = None
    release: t.Optional[Fraction] = None
    tie_to_next: bool = False
    tie_to_prev: bool = False
    grace: bool = False
    # We can use voice, when available, to differentiate ties
    voice: t.Optional[str] = None
    part: t.Optional[str] = None
    spelling: t.Optional[str] = None
    instrument: t.Optional[str] = None
    midi_instrument: t.Optional[str] = None
    unpitched: bool = False

    _type = "note"

    def is_valid(self) -> bool:
        if self.onset is None:
            return False
        if (self.pitch is None) and not (self.unpitched):
            return False
        return self.release is not None or self.grace

    @property
    def dur(self) -> Fraction:
        if self.release is None:
            return None
        return (self.release - self.onset).limit_denominator(LIMIT_DENOMINATOR)

    @dur.setter
    def dur(self, val: Fraction):
        self.release = (self.onset + val).limit_denominator(LIMIT_DENOMINATOR)

    def copy(self, remove_ties: bool = False) -> Note:
        out = copy(self)
        if remove_ties:
            out.tie_to_next = False
            out.tie_to_prev = False
        return out

    def set_spelling(
        self,
        step: str,
        alter: int,
        chromatic_transpose: None | int,
        diatonic_transpose: None | int,
    ):
        if alter < 0:
            acc = -alter * "-"
        else:
            acc = alter * "#"
        spelling = step + acc
        if chromatic_transpose is not None:
            assert diatonic_transpose is not None
            spelling = transpose_spelling(
                [spelling],
                chromatic_steps=chromatic_transpose,
                diatonic_steps=diatonic_transpose,
                flat_char="-",
            )[0]
        self.spelling = spelling

    def __str__(self):
        init_tie = "⌒" if self.tie_to_prev else ""
        end_tie = "⌒" if self.tie_to_next else ""
        return f"{init_tie}{self.pitch}:{self.onset}-{self.release}{end_tie}"


@dataclass
class Measure(DFItem):
    _type = "bar"
    onset: t.Optional[Fraction] = None
    release: t.Optional[Fraction] = None
    # TODO: (Malcolm 2024-08-10) remove
    # expected_duration: t.Optional[Fraction] = None

    # @property
    # def expected_release(self):
    #     assert self.onset is not None and self.expected_duration is not None
    #     return self.onset + self.expected_duration


@dataclass
class Tempo(DFItem):
    _type = "tempo"
    onset: Fraction
    bpm: float

    def asdict(self):
        return {
            "type": self._type,
            "onset": self.onset,
            "other": {"tempo": self.bpm},
            "release": self.release,
        }

    @property
    def release(self):
        return None


@dataclass
class TimeSignature(DFItem):
    _type = "time_signature"
    onset: Fraction
    numer: int
    denom: int

    def asdict(self):
        # to be consistent with how time-sigs are indicated in midi_parser
        return {
            "type": self._type,
            "onset": self.onset,
            "other": {"numerator": self.numer, "denominator": self.denom},
            "release": self.release,
        }

    @property
    def release(self):
        return None

    @property
    def quarter_duration(self):
        # 4/4: 4 * 4 / 4 = 4
        # 2/4: 2 * 4 / 4 = 2
        # 6/8: 6 * 4 / 8 = 3
        return Fraction(self.numer * 4, self.denom)


@dataclass
class Text(DFItem):
    _type = "text"
    onset: Fraction
    content: str

    def asdict(self):
        return {
            "type": self._type,
            "onset": self.onset,
            "other": {"text": self.content},
            "release": self.release,
        }

    @property
    def release(self):
        return None
