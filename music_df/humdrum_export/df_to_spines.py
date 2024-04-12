import itertools
import logging
import math
import re
import typing as t
from collections import defaultdict
from fractions import Fraction
from numbers import Number

import numpy as np
import pandas as pd

try:
    from metricker import Meter
except ImportError:
    pass
else:

    from mspell import Speller

    from music_df.humdrum_export.df_to_homo_df import df_to_homo_df
    from music_df.humdrum_export.dur_to_kern import KernDurError, dur_to_kern

    TOKEN_ORDER = {
        "bar": 0,
        "time_signature": 1,
        "note": 2,
    }

    LOGGER = logging.getLogger(__name__)

    def kern_barline():
        return "="

    def kern_ts(numer, denom):
        return f"*M{numer}/{denom}"

    def _get_kern_notes_sub(
        symbol: str,
        dur: int | float | Fraction,
        measure_offset: int | float | Fraction,
        meter: Meter,
    ):
        """ """
        float_durs, kern_durs = zip(
            *dur_to_kern(
                dur,
                offset=measure_offset,
                meter=meter,
                raise_exception_on_unrecognized_duration=True,
            )
        )
        offsets = [0] + list(itertools.accumulate(float_durs))[:-1]
        return offsets, [d + symbol for d in kern_durs]  # type:ignore

    def get_kern_rest(rest_dur: float | int, measure_offset: float | int, meter: Meter):
        """
        >>> fourfour, threefour = Meter("4/4"), Meter("3/4")
        >>> get_kern_rest(6, 3.5, fourfour)
        ([0, 0.5, 4.5], ['8r', '1r', '4.r'])
        """
        return _get_kern_notes_sub("r", rest_dur, measure_offset, meter)

    def kern_to_float_dur(kern: str) -> float:
        """
        >>> kern_to_float_dur("4")
        1.0
        >>> kern_to_float_dur("4.")
        1.5
        >>> kern_to_float_dur("4..")
        1.75
        >>> kern_to_float_dur("96")  # doctest: +ELLIPSIS
        0.0416666666666...
        >>> kern_to_float_dur("24...")
        0.3125

        For grace notes, we return 0
        >>> kern_to_float_dur("q0.104166")
        0.0

        """
        if kern.startswith("q"):
            return 0.0
        # Some special cases:
        if kern.startswith("3%2"):
            return 4 * 2 / 3  # Triplet whole note
        if kern.startswith("3%4"):
            return 4 * 4 / 3  # Triplet breve
        if kern.startswith("8%9"):
            return 4 * 9 / 8  # 9/8 full rest
        if kern.startswith("2%9"):
            return 18.0  # 9/2 full rest

        m = re.match(r"^\[?(?P<recip>\d+)(?P<dots>\.*).*", kern)
        assert m is not None
        out = 4 / int(m.group("recip"))
        dot_factor = sum(2**-i for i in range(len(m.group("dots")) + 1))
        out *= dot_factor
        return out

    def get_kern_notes(
        note: pd.Series,
        measure_offset: Number,
        meter: Meter,
        speller: t.Optional[Speller],
        color: t.Optional[str] = None,
        label_name: t.Optional[str] = None,
        label_mask_col: t.Optional[str] = None,
        nth_note_label_col: t.Optional[str] = None,
    ):
        """
        >>> fourfour, threefour = Meter("4/4"), Meter("3/4")
        >>> speller = Speller(pitches=True, letter_format="kern")
        >>> note = pd.Series({"pitch": 58, "onset": 3.5, "release": 9.5})
        >>> get_kern_notes(note, 3.5, fourfour, speller)
        ([0, 0.5, 4.5], ['[8B-', '1B-_', '4.B-]'], None)
        >>> note = pd.Series({"pitch": 58, "onset": 2.0, "release": 4.0})
        >>> get_kern_notes(note, 2.0, fourfour, speller)
        ([0], ['2B-'], None)
        >>> get_kern_notes(note, 2.0, threefour, speller)
        ([0, 1.0], ['[4B-', '4B-]'], None)
        >>> note = pd.Series({"pitch": 58, "onset": 2.0, "release": 4.0, "label": "hi"})
        >>> get_kern_notes(note, 2.0, threefour, speller, label_name="label")
        ([0, 1.0], ['[4B-', '4B-]'], ['hi', 'hi'])
        """
        if "humdrum_spelling" in note.index:
            spelled = note.humdrum_spelling
        else:
            spelled = speller(note.pitch)  # type:ignore
        labels = None
        dur = note.release - note.onset
        offsets, notes = _get_kern_notes_sub(
            spelled, dur, measure_offset, meter  # type:ignore
        )
        if len(notes) > 1:
            notes[0] = "[" + notes[0]
            notes[-1] = notes[-1] + "]"
        if len(notes) > 2:
            notes[1:-1] = [n + "_" for n in notes[1:-1]]
        if color is not None:
            notes = [n + color for n in notes]
        if label_name is not None:
            label = note[label_name]
            if label_mask_col is not None and not note[label_mask_col]:
                # (Malcolm 2023-09-29) We need to return empty strings because
                #   the code below expects to find a label for every note.
                labels = ["" for _ in notes]
            else:
                labels = [label for _ in notes]

        if nth_note_label_col:
            nth_note_label = note[nth_note_label_col]
            if not nth_note_label:
                nth_note_label = ""
            if label_name is not None:
                labels = [
                    nth_note_label if not l else f"{nth_note_label} {l}"
                    for l in labels  # type:ignore
                ]
            else:
                labels = [nth_note_label for _ in notes]
        return offsets, notes, labels

    # def get_ts_dur(numer: int, denom: int) -> float:
    #     """
    #     >>> get_ts_dur(4, 4)
    #     4.0
    #     >>> get_ts_dur(4, 2)
    #     8.0
    #     >>> get_ts_dur(4, 8)
    #     2.0
    #     >>> get_ts_dur(3, 4)
    #     3.0
    #     >>> get_ts_dur(6, 8)
    #     3.0
    #     >>> get_ts_dur(9, 8)
    #     4.5
    #     """
    #     beat = denom**-1 * 4
    #     return numer * beat

    def handle_rest(
        start_time,
        end_time,
        measure_start_time,
        meter,
        tokens_list,
        labels_list=None,
    ):
        offsets, rests = get_kern_rest(
            end_time - start_time, start_time - measure_start_time, meter
        )
        for offset, rest in zip(offsets, rests):
            tokens_list.append((start_time + offset, TOKEN_ORDER["note"], rest))
            if labels_list is not None:
                # We need to append a dummy value so we can zip tokens and labels
                #   together later
                labels_list.append("REMOVE")
                assert len(tokens_list) == len(labels_list)

    def get_kern_spine(
        voice_part: pd.DataFrame,
        speller: t.Optional[Speller],
        label_col: t.Optional[str] = None,
        label_mask_col: t.Optional[str] = None,
        label_color_col: t.Optional[str] = None,
        nth_note_label_col: t.Optional[str] = None,
    ) -> t.List[str]:
        """
        >>> speller = Speller(pitches=True, letter_format="kern")
        >>> df = pd.DataFrame(
        ...     {
        ...         "pitch": [60, 64, 67],
        ...         "onset": [0, 0, 1],
        ...         "release": [1, 1, 2],
        ...         "type": ["note"] * 3,
        ...     }
        ... )
        >>> df
        pitch  onset  release  type
        0     60      0        1  note
        1     64      0        1  note
        2     67      1        2  note
        >>> get_kern_spine(df, speller)
        ['4c 4e', '4g']
        >>> df = pd.DataFrame(
        ...     {
        ...         "pitch": [0, 60, 64, 67, 72, 60],
        ...         "onset": [0, 0, 1.5, 2.0, 3.5, 8.0],
        ...         "release": [float("nan"), 1, 2.0, 3.0, 8.0, 9.0],
        ...         "type": ["bar"] + ["note"] * 5,
        ...     }
        ... )
        >>> df
        pitch  onset  release  type
        0      0    0.0      NaN   bar
        1     60    0.0      1.0  note
        2     64    1.5      2.0  note
        3     67    2.0      3.0  note
        4     72    3.5      8.0  note
        5     60    8.0      9.0  note
        >>> get_kern_spine(df, speller)
        ['=', '4c', '8r', '8e', '4g', '8r', '[8cc', '1cc]', '4c']

        Note that the duration of a final empty measure is ignored.
        >>> df = pd.DataFrame(
        ...     {
        ...         "pitch": [float("nan")] * 4,
        ...         "onset": [0, 4, 8, 12],
        ...         "release": [4, 8, 12, 16],
        ...         "type": ["bar"] * 4,
        ...     }
        ... )
        >>> df
        pitch  onset  release type
        0    NaN      0        4  bar
        1    NaN      4        8  bar
        2    NaN      8       12  bar
        3    NaN     12       16  bar
        >>> get_kern_spine(df, speller)  #
        ['=', '1r', '=', '1r', '=', '1r', '=']

        >>> df = pd.DataFrame(
        ...     {
        ...         "pitch": [0, 60, 0, 61],
        ...         "onset": [0, 0, 4, 5],
        ...         "release": [4, 5, 8, 8],
        ...         "type": ["bar", "note", "bar", "note"],
        ...     }
        ... )
        >>> df
        pitch  onset  release  type
        0      0      0        4   bar
        1     60      0        5  note
        2      0      4        8   bar
        3     61      5        8  note
        >>> get_kern_spine(df, speller)
        ['=', '[1c', '=', '4c]', '[4c#', '2c#]']

        If `label_col` argument is provided, the notes will be labeled with the contents
        of this column (which should be a string) as system label. In the event that there
        are two notes their labels are joined by a newline.
        >>> df = pd.DataFrame(
        ...     {
        ...         "pitch": [60, 64, 67],
        ...         "onset": [0, 0, 1],
        ...         "release": [1, 1, 2],
        ...         "type": ["note"] * 3,
        ...         "label": ["hi", "bye", "hello"],
        ...     }
        ... )
        >>> df
        pitch  onset  release  type  label
        0     60      0        1  note     hi
        1     64      0        1  note    bye
        2     67      1        2  note  hello
        >>> get_kern_spine(df, speller, label_col="label")
        ['!LO:TX:b:t=hi\\\\nbye', '4c 4e', '!LO:TX:b:t=hello', '4g']
        """
        tokens = []
        labels = []

        prev_onset = -1
        prev_release: float = 0.0
        measure_start: float = 0.0

        meter = Meter("4/4")  # Set default meter

        this_measure_tokens = []
        this_measure_labels = []

        skip_measure = False

        # TODO: (Malcolm 2024-02-29) remove obsolete code?
        # actual_bar_dur = 0
        # expected_bar_dur = None

        for i, row in voice_part.iterrows():
            if skip_measure:
                if row.type != "bar":
                    continue
                this_measure_tokens = []
                this_measure_labels = []
                handle_rest(
                    measure_start,
                    row.onset,
                    0.0,
                    meter,
                    this_measure_tokens,
                    (
                        this_measure_labels
                        if (label_col is not None or nth_note_label_col is not None)
                        else None
                    ),
                )
                prev_release: float = row.onset

            if row.onset > prev_release:
                try:
                    handle_rest(
                        prev_release,
                        row.onset,
                        measure_start,
                        meter,
                        this_measure_tokens,
                        (
                            this_measure_labels
                            if (label_col is not None or nth_note_label_col is not None)
                            else None
                        ),
                    )
                except KernDurError as exc:
                    skip_measure = True
                    LOGGER.warning(f"Replacing measure contents with rest due to {exc}")
                    continue

                # TODO: (Malcolm 2024-03-01) remove dead code
                # offsets, rests = get_kern_rest(
                #     row.onset - prev_release,
                #     prev_release - measure_start,
                #     meter,
                #     # row.onset - prev_release, row.onset - measure_start, meter
                # )
                # for offset, rest in zip(offsets, rests):
                #     this_measure_tokens.append(
                #         (prev_release + offset, TOKEN_ORDER["note"], rest)  # type:ignore
                #     )
                #     if label_col is not None or nth_note_label_col is not None:
                #         # We need to append a dummy value so we can zip tokens and labels
                #         #   together below
                #         this_measure_labels.append("REMOVE")
                #         assert len(this_measure_labels) == len(this_measure_tokens)
                # actual_bar_dur += kern_to_float_dur(rest)
                prev_release: float = row.onset
            if row.type == "bar":
                measure_start = row.onset
                this_measure_tokens.append(
                    (row.onset, TOKEN_ORDER["bar"], kern_barline())
                )
                if label_col is not None or nth_note_label_col is not None:
                    # We need to append a dummy value so we can zip tokens and labels
                    #   together below
                    this_measure_labels.append("REMOVE")
                    assert len(this_measure_labels) == len(this_measure_tokens)
                tokens.extend(this_measure_tokens)
                labels.extend(this_measure_labels)
                this_measure_tokens = []
                this_measure_labels = []
                skip_measure = False

            elif row.type == "time_signature":
                this_measure_tokens.append(
                    (
                        row.onset,
                        TOKEN_ORDER["time_signature"],
                        kern_ts(
                            numer=row.other["numerator"],
                            denom=row.other["denominator"],
                        ),
                    )
                )
                meter = Meter(f"{row.other['numerator']}/{row.other['denominator']}")
                if label_col is not None or nth_note_label_col is not None:
                    # We need to append a dummy value so we can zip tokens and labels
                    #   together below
                    this_measure_labels.append("REMOVE")
                    assert len(this_measure_labels) == len(this_measure_tokens)
            elif row.type in {"tempo", "text"}:
                # (Malcolm 2023-12-18) For now, we do nothing about tempi or text
                continue
            else:
                assert row.type == "note"
                try:
                    offsets, kern_notes, note_labels = get_kern_notes(
                        row,
                        row.onset - measure_start,
                        meter,
                        speller,
                        row.color_char if "color_char" in row.index else None,
                        label_name=label_col,
                        label_mask_col=label_mask_col,
                        nth_note_label_col=nth_note_label_col,
                    )
                except KernDurError as exc:
                    skip_measure = True
                    LOGGER.warning(f"Replacing measure contents with rest due to {exc}")
                    continue

                if note_labels is None:
                    note_labels = [None] * len(kern_notes)

                for i, (offset, kern_note, label) in enumerate(
                    zip(offsets, kern_notes, note_labels)
                ):
                    if prev_onset == row.onset and prev_release == row.release:
                        j = -(len(offsets) - i)
                        this_measure_tokens[j][2] += f" {kern_note}"
                        if label_col is not None or nth_note_label_col is not None:
                            if label == "":
                                pass
                            elif not this_measure_labels[j]:
                                if label_color_col is not None:
                                    label_color = row[label_color_col]
                                    text_token = (
                                        f"!LO:TX:b:color={label_color}:t={label}"
                                    )
                                else:
                                    text_token = f"!LO:TX:b:t={label}"
                                this_measure_labels[j] = text_token
                            elif label is not None:
                                if label_color_col is not None:
                                    # assert row[label_color_col] == "#FF0000"
                                    m = re.search(
                                        r"color=(?P<color>[^:]+):",
                                        this_measure_labels[j],
                                    )
                                    assert m is not None
                                    existing_color = m.group("color")
                                    if not row[label_color_col] == existing_color:
                                        LOGGER.warning(
                                            f"{row[label_color_col]=} does not match existing label color {existing_color}, ignoring"
                                        )
                                this_measure_labels[j] += rf"\n{label}"
                    else:
                        this_measure_tokens.append(
                            [row.onset + offset, TOKEN_ORDER["note"], kern_note]
                        )
                        # actual_bar_dur += kern_to_float_dur(kern_note)
                        if label_col is not None or nth_note_label_col is not None:
                            if label == "":
                                this_measure_labels.append("")
                            elif label is not None:
                                if label_color_col is not None:
                                    label_color = row[label_color_col]
                                    text_token = (
                                        f"!LO:TX:b:color={label_color}:t={label}"
                                    )
                                else:
                                    text_token = f"!LO:TX:b:t={label}"
                                this_measure_labels.append(text_token)
                            assert len(this_measure_labels) == len(this_measure_tokens)

            prev_onset = row.onset
            if row.type == "note":
                prev_release: float = row.release

        if this_measure_tokens:
            # In case the score doesn't have a final barline we need to
            #   collect the last measure contents
            tokens.extend(this_measure_tokens)
            labels.extend(this_measure_labels)

        if labels:
            assert len(labels) == len(tokens)
            tokens = [
                tuple(token[:2]) + (label,) + (token[2],)
                for (token, label) in zip(tokens, labels)
            ]

        tokens.sort(key=lambda x: x[1])
        tokens.sort(key=lambda x: x[0])

        tokens = [x for token in tokens for x in token[2:]]

        if labels:
            tokens = [token for token in tokens if token not in {"REMOVE", ""}]

        return tokens

    def df_to_spines(
        df,
        label_col: t.Optional[str] = None,
        label_mask_col: t.Optional[str] = None,
        label_color_col: t.Optional[str] = None,
        nth_note_label_col: t.Optional[str] = None,
    ) -> t.List[t.List[str]]:
        if label_mask_col is not None:
            assert len(df.loc[df[label_mask_col], label_color_col].unique()) == 1
        if df.iloc[-1].type != "bar":
            last_bar = pd.DataFrame({"type": ["bar"], "onset": [df.release.max()]})
            df = pd.concat([df, last_bar])
        # df needs to have "part" and "voice" columns
        if "part" not in df.columns:
            if "track" in df.columns:
                df["part"] = df.track
            else:
                df["part"] = float("nan")
                df.loc[df.type == "note", "part"] = 1.0
        if "voice" not in df.columns:
            df["voice"] = float("nan")
            df.loc[df.type == "note", "voice"] = 1.0
        if "humdrum_spelling" in df.columns:
            speller = None
        else:
            speller = Speller(pitches=True, letter_format="kern")
        voices = df.voice[~df.voice.isna()].unique()
        parts = df.part.unique()
        parts = df.part[~df.part.isna()].unique()
        pairs = itertools.product(voices, parts)
        spines = []
        for voice, part in pairs:
            voice_part = df[
                (df.voice.isna() | (df.voice == voice))
                & (df.part.isna() | (df.part == part))
            ]
            if not (voice_part.type == "note").any():
                continue
            homo_parts = df_to_homo_df(voice_part)
            spines.extend(
                [
                    get_kern_spine(
                        homo_part,
                        speller,
                        label_col=label_col,
                        label_mask_col=label_mask_col,
                        label_color_col=label_color_col,
                        nth_note_label_col=nth_note_label_col,
                    )
                    for homo_part in homo_parts
                ]
            )
        return spines
