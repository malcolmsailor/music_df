import math
import typing as t
from fractions import Fraction
from numbers import Number

try:
    from metricker import Meter
except ImportError:
    pass
else:

    NOTE_VALUES = [Fraction(4, 2**i) for i in range(10)] + [
        Fraction(2, 3 * 2**i) for i in range(10)
    ]
    NOTE_VALUES.sort(reverse=True)

    MAX_DEPTH = 3
    INPUT_MULTIPLICANDS = [1] + [
        Fraction(1 * 2**i, (2 * 2**i - 1)) for i in range(1, MAX_DEPTH)
    ]
    ALLOWED_ERROR = 0.002
    INV_ALLOWED_ERROR = 1 - ALLOWED_ERROR

    class KernDurError(Exception):
        pass

    class _Dur:
        def __init__(self, onset: float | Fraction, release: float | Fraction):
            self.onset = onset
            self.release = release

        def __repr__(self):
            return (
                f"{self.__class__.__name__}(onset={self.onset}, release={self.release})"
            )

        def __eq__(self, other):
            return self.onset == other.onset and self.release == other.release

        @property
        def dur(self) -> float | Fraction:
            return self.release - self.onset

    def duration_float_to_recip(input: float, threshold=0.01) -> str:
        """
        A Python implementation of humlib's Convert::durationFloatToRecip function.

        Differences: omits timebase parameter (for now, at least).

        >>> duration_float_to_recip(4.0)
        '1'
        >>> duration_float_to_recip(6.0)
        '1.'
        >>> duration_float_to_recip(1 / 3)
        '12'
        >>> duration_float_to_recip(3 / 4)
        '8.'
        >>> duration_float_to_recip(9 / 2)
        '8%9'
        >>> duration_float_to_recip(0.3333)
        '12'
        >>> duration_float_to_recip(0.333)
        '12'
        >>> duration_float_to_recip(0.334)
        '12'

        If threshold is set too high the next tests fail.
        >>> duration_float_to_recip(0.1667)
        '24'
        >>> duration_float_to_recip(0.1666)
        '24'
        >>> duration_float_to_recip(0.1670)
        '24'
        >>> duration_float_to_recip(0.0833)
        '48'
        >>> duration_float_to_recip(0.04165)
        '96'
        """
        if input == 0.0625:
            output = "64"
            return output
        if input == 0.125:
            output = "32"
            return output
        if input == 0.25:
            output = "16"
            return output
        if input == 0.5:
            output = "8"
            return output
        if input == 1.0:
            output = "4"
            return output
        if input == 2.0:
            output = "2"
            return output
        if input == 4.0:
            output = "1"
            return output
        if input == 8.0:
            output = "0"
            return output
        if input == 12.0:
            output = "0."
            return output
        if input == 16.0:
            output = "00"
            return output
        if input == 24.0:
            output = "00."
            return output
        if input == 32.0:
            output = "000"
            return output
        if input == 48.0:
            output = "000."
            return output

        # special case for triplet whole notes:
        if abs(input - (4.0 * 2.0 / 3.0)) < 0.0001:
            return "3%2"

        # special case for triplet breve notes:
        if abs(input - (4.0 * 4.0 / 3.0)) < 0.0001:
            return "3%4"

        # special case for 9/8 full rests
        if abs(input - (4.0 * 9.0 / 8.0)) < 0.0001:
            return "8%9"

        # special case for 9/2 full-measure rest
        if abs(input - 18.0) < 0.0001:
            return "2%9"

        # handle special rounding cases primarily for SCORE which
        # only stores 4 digits for a duration
        if math.isclose(input, 1 / 3, abs_tol=0.01):
            return "12"
        if math.isclose(input, 1 / 6, abs_tol=0.01):
            return "24"
        if math.isclose(input, 1 / 12, abs_tol=0.01):
            # triplet 32nd note, which has a real duration of 0.0833333 etc.
            return "48"
        if math.isclose(input, 1 / 24, abs_tol=0.01):
            # triplet 64th note, which has a real duration of 0.0833333 etc.
            return "96"

        basic = 4.0 / input
        diff = basic - int(basic)
        if diff > (1 - threshold):
            diff = 1.0 - diff
            basic += diff

        output = []
        if diff < threshold:
            output.append(str(int(basic)))
        else:
            testinput = input / 3.0 * 2.0
            basic = 4.0 / testinput
            diff = basic - int(basic)
            if diff < threshold:
                output.append(str(int(basic)))
                output.append(".")
            else:
                testinput = input / 7.0 * 4.0
                basic = 4.0 / testinput
                diff = basic - int(basic)
                if diff < threshold:
                    output.append(str(int(basic)))
                    output.append("..")
                else:
                    testinput = input / 15.0 * 4.0
                    basic = 2.0 / testinput
                    diff = basic - int(basic)
                    if diff < threshold:
                        output.append(str(int(basic)))
                        output.append("...")
                    else:
                        # Don't know what it could be so echo as a grace note.
                        output.append("q")
                        output.append(str(input))

        return "".join(output)

    def attempt_to_represent_as_tuple(inp: float):
        """
        >>> attempt_to_represent_as_tuple(0.14406779661015978)
        """
        # TODO: (Malcolm 2024-02-29)
        pass

    def split_fives_hack(dur: _Dur) -> list[_Dur]:
        """
        There is an issue where durations of 5, (or of 5/16, etc.), which cannot be
        represented in notation without a tie, make it through when they start on a
        downbeat in certain meters,
        e.g.:

        >>> six_four = Meter("6/4")
        >>> six_four.split_at_metric_strong_points([_Dur(0.0, 5.0)])
        [_Dur(onset=0.0, release=5.0)]

        To avoid this circumstance, we just hackily divide them into 3 + 2.
        >>> split_fives_hack(_Dur(0.0, 5.0))
        [_Dur(onset=0.0, release=3.0), _Dur(onset=3.0, release=5.0)]
        >>> split_fives_hack(_Dur(0.0, 10.0))
        [_Dur(onset=0.0, release=6.0), _Dur(onset=6.0, release=10.0)]
        >>> split_fives_hack(_Dur(0.0, 1.25))
        [_Dur(onset=0.0, release=0.75), _Dur(onset=0.75, release=1.25)]
        """
        # TODO: (Malcolm 2023-12-18) better solution than this hack
        for divisor in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96]:
            if math.isclose(math.log(divisor * dur.dur / 5, 2) % 1, 0, abs_tol=1e-9):
                dur1 = _Dur(dur.onset, dur.onset + 3 / 5 * dur.dur)
                dur2 = _Dur(dur1.release, dur.release)
                return [dur1, dur2]
        return [dur]

    def dur_to_kern(
        inp: float | int | Fraction,
        offset: float | int | Fraction,
        meter: t.Union[str, Meter],
        raise_exception_on_unrecognized_duration: bool = False,
    ) -> list[tuple[float, str]]:
        """

        >>> dur_to_kern(10.25, 0.0, "4/4")
        [(4.0, '1'), (4.0, '1'), (2.0, '2'), (0.25, '16')]
        >>> dur_to_kern(10.0, 0.25, "4/4")
        [(0.25, '16'), (0.5, '8'), (1.0, '4'), (2.0, '2'), (4.0, '1'), (2.0, '2'), (0.25, '16')]
        >>> dur_to_kern(5.5, 2.0, "3/4")
        [(1.0, '4'), (3.0, '2.'), (1.5, '4.')]
        >>> dur_to_kern(9.25, 0.5, "9/8")
        [(0.5, '8'), (0.5, '8'), (1.5, '4.'), (1.5, '4.'), (4.5, '8%9'), (0.75, '8.')]
        >>> dur_to_kern(0.3333, 0.0, "4/4")
        [(0.3333, '12')]
        >>> dur_to_kern(0.1666, 0.0, "4/4")
        [(0.1666, '24')]
        >>> dur_to_kern(0.1667, 0.3333, "2/4")
        [(0.16670000000000001, '24')]
        >>> dur_to_kern(0.1667, 1.6667, "2/4")
        [(0.16670000000000007, '24')]
        >>> dur_to_kern(0.1667, 1.75, "2/4")
        [(0.16670000000000007, '24')]
        >>> dur_to_kern(0.1670000000000016, 1.5, "2/4")
        [(0.1670000000000016, '24')]

        Following used to give a recursion error, fixed by `split_fives_hack()`
        >>> dur_to_kern(5.0, 6.0, "6/4")
        [(3.0, '2.'), (2.0, '2')]
        """
        if isinstance(meter, str):
            meter = Meter(meter)
        split_durs = meter.split_at_metric_strong_points(
            [_Dur(offset, offset + inp)], min_split_dur=0.5  # type:ignore
        )
        # (Malcolm 2023-10-10) I'm not precisely sure what the rationale for applying
        #   split_odd_duration is
        split_durs = split_durs[:-1] + meter.split_odd_duration(
            split_durs[-1], min_split_dur=1.0  # type:ignore
        )
        # Hack
        split_durs = split_durs[:-1] + split_fives_hack(split_durs[-1])

        durs = [float(dur.release - dur.onset) for dur in split_durs]
        # It's possible these will be different
        output_durs = []
        output_kern_durs = []
        for d in durs:
            kern_dur = duration_float_to_recip(d)
            if kern_dur.startswith("q"):
                remainder = d
                temp_offset = offset
                for divisor in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]:
                    whole, remainder = divmod(remainder, 1 / divisor)
                    whole /= divisor
                    if whole <= 0.0:
                        continue
                    result1 = dur_to_kern(
                        whole,
                        temp_offset,
                        meter,
                        raise_exception_on_unrecognized_duration,
                    )
                    output_durs.extend([r[0] for r in result1])
                    output_kern_durs.extend([r[1] for r in result1])
                    if remainder < 1e-6:
                        break
                    temp_offset += whole
                if remainder > 1e-6 and raise_exception_on_unrecognized_duration:
                    raise KernDurError(
                        f"Unrecognized duration {inp} with {offset=} in {meter=}"
                    )
                # if d > 1:
                #     whole, frac = divmod(d, 1.0)
                #     if whole <= 0:
                #         continue
                #     result1 = dur_to_kern(
                #         whole,
                #         offset,
                #         meter,
                #         raise_exception_on_unrecognized_duration,
                #     )
                #     result2 = dur_to_kern(
                #         frac,
                #         offset + whole,
                #         meter,
                #         raise_exception_on_unrecognized_duration,
                #     )
                #     output_durs.extend([r[0] for r in result1])
                #     output_durs.extend([r[0] for r in result2])
                #     output_kern_durs.extend([r[1] for r in result1])
                #     output_kern_durs.extend([r[1] for r in result2])

                # TODO: (Malcolm 2024-02-29) restore
                # elif raise_exception_on_unrecognized_duration:
                #     raise KernDurError(
                #         f"Unrecognized duration {inp} with {offset=} in {meter=}"
                #     )
                # else:
                #     # TODO: (Malcolm 2024-02-28) remove?
                #     output_durs.append(d)
                #     output_kern_durs.append(kern_dur)
            else:
                output_durs.append(d)
                output_kern_durs.append(kern_dur)

        # kern_durs = [duration_float_to_recip(d) for d in durs]

        # if raise_exception_on_unrecognized_duration and any(
        #     d.startswith("q") for d in kern_durs
        # ):
        #     raise KernDurError(f"Unrecognized duration {inp} with {offset=} in {meter=}")
        return list(zip(output_durs, output_kern_durs))

    # def dur_to_kern_old(
    #     inp: Number,
    #     offset: Number = 0,
    #     # unbreakable_value: Number = Fraction(1, 1),
    #     rest: bool = False,
    #     dotted_rests: bool = False,
    #     time_sig_dur: Number = 4,
    #     dur_return_type: t.Type = float,
    # ) -> t.List[t.Tuple[Number, str]]:
    #     """Converts a duration to kern format.

    #     Works with duple rhythms including ties and dotted notes. Probably doesn't
    #     work yet with triplets and other non-duple subdivisions.

    #     Was originally based on Craig Sapp's durationToKernRhythm function in
    #     Convert.cpp from humextra. I'm unsure what the "timebase" parameter of
    #     Sapp's function does, so I omitted it.

    #     Keyword arguments:
    #         # unbreakable value: integer/float. 1 = quarter, 0.5 eighth, etc. offset:
    #         # TODO implement?
    #         used to specify the division of the unbreakable value upon
    #             which the note begins.
    #         time_sig_dur: must be specified for rests, so that rests won't be
    #             longer than the measure that (should) contain them.

    #     Returns:
    #         a list of 2-tuples:
    #             - first item is dur_return_type corresponding to the duration of the
    #                 item in quarter-notes
    #             - second item is a string giving the kern representation of the
    #                 rhythm

    #     >>> dur_to_kern(1.0)
    #     [(1.0, '4')]
    #     >>> dur_to_kern(5.0)
    #     [(4.0, '1'), (1.0, '4')]

    #     >>> dur_to_kern(5.0, time_sig_dur=3.0)
    #     [(3.0, '2.'), (2.0, '2')]
    #     >>> dur_to_kern(3.5, offset=0.5) # TODO revise
    #     [(3.5, '2..')]
    #     >>> dur_to_kern(12.0)
    #     [(4.0, '1'), (4.0, '1'), (4.0, '1')]
    #     >>> dur_to_kern(10.0)
    #     [(4.0, '1'), (4.0, '1'), (2.0, '2')]
    #     >>> dur_to_kern(2.25)
    #     [(2.0, '2'), (0.25, '16')]
    #     """
    #     # The maximum number of dots a note can receive will be one fewer
    #     # than max_depth. (So max_depth = 3 means double-dotted notes are
    #     # allowed, but no greater.)

    #     if rest and not dotted_rests:
    #         max_depth = 1
    #     else:
    #         max_depth = MAX_DEPTH

    #     # We can potentially do this with other subdivisions by replacing
    #     # "2" with "3", "5", etc.

    #     def subfunc(inp: Number, depth: float):
    #         if depth == max_depth:
    #             return None

    #         testinput = inp * INPUT_MULTIPLICANDS[depth]
    #         basic = 4 / testinput
    #         diff = basic - int(basic)
    #         # In Sapp's code, which does not employ recursion, the next
    #         # condition is only tested when depth == 0.
    #         if diff > INV_ALLOWED_ERROR:
    #             diff = 1 - diff
    #             basic += ALLOWED_ERROR

    #         if diff < ALLOWED_ERROR:
    #             output = str(int(basic)) + depth * "."
    #             return output

    #         return subfunc(inp, depth + 1)

    #     adjusted_input = inp
    #     output = []

    #     offset %= time_sig_dur
    #     first_m = True
    #     while offset + adjusted_input >= time_sig_dur:
    #         within_measure_input = time_sig_dur - offset
    #         adjusted_input -= within_measure_input
    #         while within_measure_input:
    #             temp_output = subfunc(within_measure_input, 0)
    #             if temp_output:
    #                 output.append(
    #                     (dur_return_type(within_measure_input), temp_output)
    #                 )
    #                 break

    #             for note_value in NOTE_VALUES:
    #                 if within_measure_input > note_value:
    #                     within_measure_input = within_measure_input - note_value
    #                     temp_temp_output = subfunc(note_value, 0)
    #                     if temp_temp_output:
    #                         output.append(
    #                             (dur_return_type(note_value), temp_temp_output)
    #                         )
    #                         break
    #         if first_m:
    #             output.reverse()
    #             first_m = False
    #         offset = 0

    #     # offset = offset % unbreakable_value

    #     # if offset != 0 and adjusted_input + offset > unbreakable_value:
    #     # if adjusted_input + offset > unbreakable_value:
    #     #     unbroken_input = unbreakable_value - offset
    #     #     sub_output = []
    #     #     while True:
    #     #         temp_output = subfunc(unbroken_input, 0)
    #     #         if temp_output:
    #     #             sub_output.append(
    #     #                 (dur_return_type(unbroken_input), temp_output)
    #     #             )
    #     #             break

    #     #         for note_value in NOTE_VALUES:
    #     #             if unbroken_input > note_value:
    #     #                 unbroken_input = unbroken_input - note_value
    #     #                 temp_temp_output = subfunc(note_value, 0)
    #     #                 if temp_temp_output:
    #     #                     sub_output.append(
    #     #                         (dur_return_type(note_value), temp_temp_output)
    #     #                     )
    #     #                     break
    #     #     adjusted_input -= unbreakable_value - offset
    #     #     if rest:
    #     #         sub_output.reverse()
    #     #     output += sub_output

    #     if not adjusted_input:
    #         return output

    #     counter = 0
    #     while True:
    #         counter += 1
    #         temp_output = subfunc(adjusted_input, 0)
    #         if temp_output:
    #             output.append((dur_return_type(adjusted_input), temp_output))
    #             break

    #         break_out = False
    #         for note_value in NOTE_VALUES:
    #             if note_value < ALLOWED_ERROR:
    #                 break_out = True
    #                 break
    #             if adjusted_input > note_value:
    #                 adjusted_input = adjusted_input - note_value
    #                 temp_temp_output = subfunc(note_value, 0)
    #                 if temp_temp_output:
    #                     output.append(
    #                         (dur_return_type(note_value), temp_temp_output)
    #                     )
    #                     break_out = True
    #                     break
    #         if break_out:
    #             break

    #     return output
