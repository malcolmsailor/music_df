from numbers import Number
import re
import warnings
import itertools
import typing as t


def _get_onset_from_measure_num(measure_num, measure_ends):
    if measure_num == 0:
        return 0
    return measure_ends[measure_num]


def _update_ending_number(ending_number: str):
    try:
        ending_int = int(re.match(r".*(\d+).*", ending_number).group(1))
    except IndexError:
        warnings.warn(f"no ending number found in {ending_number}")
        return "unknown"
    else:
        return str(ending_int + 1)


def get_repeat_segments(
    repeats: t.Dict[int, t.Dict[str, t.Dict[str, t.Any]]],
    measure_ends: t.Dict[int, Number],
) -> t.Tuple[
    t.List[t.Tuple[Number, Number]],
    t.List[t.Tuple[Number, Number]],
    t.List[str],
]:
    """
    Args:
        repeats: a dictionary:
            - keys are integers indicating bar numbers. (Expected to be
                1-indexed.)
            - values are dictionaries storing repeat attributes where:
                - keys indicate the repeat type ("backward", "forward",
                    or "start-ending")
                - values are a dict that stores the other attributes of the
                    repeat (e.g., "times")
        measure_ends: a dictionary:
            - keys are integers indicating bar numbers. (Expected to be
                1-indexed.)
            - values are Fractions indicating the onset time of the closing
                barline of each measure

    Returns:
        a 3-tuple of "orig_segments", "repeated_segments", and "segment_types".
            The first two items are lists of "start" and "end" times.
            The third item is a list of strings.



    >>> measure_ends = {i + 1: (i + 1) * 4 for i in range(16)}

    >>> get_repeat_segments({}, measure_ends)
    ([(0, inf)], [(0, inf)], ['no_repeat'])

    >>> get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"backward": {"times": 2}}},
    ...     measure_ends)
    ([(0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, inf)], ['simple_repeat', 'simple_repeat', 'no_repeat'])

    Note that "times" attribute of forward repeats is ignored.

    >>> get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"backward": {"times": 3}}},
    ...     measure_ends)
    ([(0, 8), (0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, 24), (24, inf)], ['simple_repeat', 'simple_repeat', 'simple_repeat', 'no_repeat'])

    Repeat of a single measure:

    >>> get_repeat_segments(
    ...     {1: {"forward": {"times": 2}, "backward": {"times": 2}}},
    ...     measure_ends)
    ([(0, 4), (0, 4), (4, inf)], [(0, 4), (4, 8), (8, inf)], ['simple_repeat', 'simple_repeat', 'no_repeat'])

    First forward repeat not at start.

    >>> get_repeat_segments(
    ...     {2: {"forward": {"times": 2}},
    ...      3: {"backward": {"times": 2}}},
    ...     measure_ends)
    ([(0, 4), (4, 12), (4, 12), (12, inf)], [(0, 4), (4, 12), (12, 20), (20, inf)], ['no_repeat', 'simple_repeat', 'simple_repeat', 'no_repeat'])

    Multiple repeats.

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"backward": {"times": 2}},
    ...      3: {"forward": {"times": 2}},
    ...      6: {"backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 8), (0, 8), (8, 24), (8, 24), (24, inf)]
    >>> segment_types
    ['simple_repeat', 'simple_repeat', 'simple_repeat', 'simple_repeat', 'no_repeat']

    Repeats with a gap.

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"backward": {"times": 2}},
    ...      4: {"forward": {"times": 2}},
    ...      6: {"backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 8), (0, 8), (8, 12), (12, 24), (12, 24), (24, inf)]
    >>> segment_types
    ['simple_repeat', 'simple_repeat', 'no_repeat', 'simple_repeat', 'simple_repeat', 'no_repeat']

    Endings.

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
    ...      3: {"start-ending": {"number": "2"}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (0, 4), (8, inf)]
    >>> segment_types
    ['pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2']

    Gap between endings and next repeat.

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
    ...      3: {"start-ending": {"number": "2"}},
    ...      5: {"forward": {"times": 2}, "backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (0, 4), (8, 16), (16, 20), (16, 20), (20, inf)]
    >>> segment_types # NB that all music after the start of the second ending goes into 'ending_2'
    ['pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2', 'simple_repeat', 'simple_repeat', 'no_repeat']

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 3}},
    ...      3: {"start-ending": {"number": "2"}},
    ...      5: {"forward": {"times": 2}, "backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (0, 4), (4, 8), (0, 4), (8, 16), (16, 20), (16, 20), (20, inf)]
    >>> segment_types
    ['pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2', 'simple_repeat', 'simple_repeat', 'no_repeat']

    Missing endings after a backward repeat are inferred (note that the second
    ending here is missing):

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (0, 4), (8, inf)]
    >>> segment_types
    ['pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2']

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {1: {"forward": {"times": 2}},
    ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
    ...      4: {"forward": {"times": 2}, "backward": {"times": 2}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (0, 4), (8, 12), (12, 16), (12, 16), (16, inf)]
    >>> segment_types
    ['pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2', 'simple_repeat', 'simple_repeat', 'no_repeat']

    The "number" attribute of the endings doesn't influence `orig_segments`
    or `repeated_segments`. It is included in `segment_types`, however.

    >>> orig_segments, _, segment_types = get_repeat_segments(
    ...     {2: {"forward": {"times": 2}},
    ...      3: {"start-ending": {"number": "1, 2"}, "backward": {"times": 3}},
    ...      4: {"start-ending": {"number": "3"}, "backward": {"times": 2}},
    ...      5: {"start-ending": {"number": "4"}},
    ...      6: {"forward": {"times": 2}},
    ...      8: {"start-ending": {"number": "1"}},
    ...      9: {"backward": {"times": 2}},
    ...      10: {"start-ending": {"number": "2"}}},
    ...     measure_ends)
    >>> orig_segments
    [(0, 4), (4, 8), (8, 12), (4, 8), (8, 12), (4, 8), (12, 16), (4, 8), (16, 20), (20, 28), (28, 36), (20, 28), (36, inf)]
    >>> segment_types
    ['no_repeat', 'pre_ending_repeat', 'ending_1, 2', 'pre_ending_repeat', 'ending_1, 2', 'pre_ending_repeat', 'ending_3', 'pre_ending_repeat', 'ending_4', 'pre_ending_repeat', 'ending_1', 'pre_ending_repeat', 'ending_2']

    We infer missing forward repeats:

    >>> get_repeat_segments(
    ...     {2: {"backward": {"times": 2}}},
    ...     measure_ends)
    ([(0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, inf)], ['simple_repeat', 'simple_repeat', 'no_repeat'])

    >>> get_repeat_segments(
    ...     {2: {"backward": {"times": 2}},
    ...      4: {"backward": {"times": 2}}},
    ...     measure_ends)[0]
    [(0, 8), (0, 8), (8, 16), (8, 16), (16, inf)]

    Missing final repeat:
    >>> get_repeat_segments(
    ...     {1: {"forward": {"times": 2}}},
    ...     measure_ends)
    ([(0, 64), (0, 64)], [(0, 64), (64, 128)], ['simple_repeat', 'simple_repeat'])

    """
    if not repeats:
        return ([(0, float("inf"))], [(0, float("inf"))], ["no_repeat"])
    assert min(repeats) > 0
    assert min(measure_ends) > 0

    last_forward_repeat = 0
    last_backward_repeat = 0
    last_boundary = lambda: max(last_forward_repeat, last_backward_repeat)
    ending_jump_from = None
    segments = []
    segment_types = []
    ending_number = ""
    ending_number_used = True
    for measure_num in sorted(repeats.keys()):
        repeat_dict = repeats[measure_num]
        if "forward" in repeat_dict:
            if "start-ending" in repeat_dict:
                warnings.warn(
                    f"M. {measure_num} has 'forward' repeat bar and "
                    "'start-ending'; ignoring 'start-ending'"
                )
            onset = _get_onset_from_measure_num(measure_num - 1, measure_ends)
            if ending_jump_from is not None:
                # there are incomplete endings...
                segments.extend(
                    [
                        (last_forward_repeat, ending_jump_from),
                        (ending_jump_to, onset),
                    ]
                )
                if ending_number_used:
                    ending_number = _update_ending_number(ending_number)
                segment_types.extend(
                    ["pre_ending_repeat", f"ending_{ending_number}"]
                )
                ending_jump_from = None
            last_forward_repeat = onset
        elif "start-ending" in repeat_dict:
            onset = _get_onset_from_measure_num(measure_num - 1, measure_ends)
            if ending_jump_from is None:
                ending_jump_from = onset
            ending_jump_to = onset
            ending_number = repeat_dict["start-ending"]["number"]
            ending_number_used = False
        if "backward" in repeat_dict:
            onset = _get_onset_from_measure_num(measure_num, measure_ends)
            if not segments:
                if last_forward_repeat != 0:
                    segments.append((0, last_forward_repeat))
                    segment_types.append("no_repeat")
            if ending_jump_from is not None:
                for _ in range(repeat_dict["backward"]["times"] - 1):
                    segments.extend(
                        [
                            (last_forward_repeat, ending_jump_from),
                            (ending_jump_to, onset),
                        ]
                    )
                    segment_types.extend(
                        ["pre_ending_repeat", f"ending_{ending_number}"]
                    )
                    ending_number_used = True
                # we need to set "ending_jump_to" here to catch the case where
                #   there is an omitted 2nd ending after a first ending
                ending_jump_to = onset
            else:
                if last_forward_repeat < last_backward_repeat:
                    # there is a missing forward repeat
                    last_forward_repeat = last_backward_repeat
                if segments and last_forward_repeat != segments[-1][1]:
                    segments.append((segments[-1][1], last_forward_repeat))
                    segment_types.append("no_repeat")
                for _ in range(repeat_dict["backward"]["times"]):
                    segments.append((last_forward_repeat, onset))
                    segment_types.append("simple_repeat")
            last_backward_repeat = onset
    if ending_jump_from is not None:
        segments.extend(
            [
                (last_forward_repeat, ending_jump_from),
                (ending_jump_to, float("inf")),
            ]
        )
        if ending_number_used:
            ending_number = _update_ending_number(ending_number)
        segment_types.extend(["pre_ending_repeat", f"ending_{ending_number}"])
    if last_forward_repeat == last_boundary():
        # There is a missing backward repeat at the end of the score.
        segments.extend(
            [(last_forward_repeat, measure_ends[max(measure_ends)])] * 2
        )
        segment_types.extend(["simple_repeat", "simple_repeat"])
    if segments[-1][1] < measure_ends[max(measure_ends)]:
        segments.append((last_boundary(), float("inf")))
        segment_types.append("no_repeat")
    segment_durs = [segment[1] - segment[0] for segment in segments]
    segment_ends = list(itertools.accumulate(segment_durs))
    segment_starts = [0] + segment_ends[:-1]
    return segments, list(zip(segment_starts, segment_ends)), segment_types


# TODO I'd like to implement processing "special" symbols like da capo, etc.
#   To do that, I believe I need to process the repeats differently. That's
#   largely implemented below. The current hickup is determining the right
#   number of times to repeat at each ending.
# NB that if I do implement the below I need to update it according to
#   changes and additional doctests added to get_repeat_segments above.
# def _get_times_from_endings(repeats):
#     """
#     According to the musicxml standard, the "times" attribute of backward
#     repeat bars is only used for backward repeats without an ending.
#     """
#     pass


# def get_repeat_segments2(
#     repeats: t.Dict[int, t.Dict[str, t.Dict[str, t.Any]]],
#     measure_ends: t.Dict[int, Number],
# ) -> t.Tuple[t.List[t.Tuple[Number, Number]], t.List[t.Tuple[Number, Number]]]:
#     """
#     Args:
#         repeats: a dictionary:
#             - keys are integers indicating bar numbers. (Expected to be
#                 1-indexed.)
#             - values are dictionaries storing repeat attributes where:
#                 - keys indicate the repeat type ("backward", "forward",
#                     or "start-ending")
#                 - values are a dict that stores the other attributes of the
#                     repeat (e.g., "times")
#         measure_ends: a dictionary:
#             - keys are integers indicating bar numbers. (Expected to be
#                 1-indexed.)
#             - values are Fractions indicating the onset time of the closing
#                 barline of each measure

#     Returns:
#         a 2-tuple of "orig_segments" and "repeated_segments". Each item is
#             a list of "start" and "end" times.

#     >>> measure_ends = {i + 1: (i + 1) * 4 for i in range(16)}

#     # >>> get_repeat_segments2({}, measure_ends)
#     # ([(0, inf)], [(0, inf)])

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"backward": {"times": 2}}},
#     # ...     measure_ends)
#     # ([(0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, inf)])

#     # Note that "times" attribute of forward repeats is ignored.

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"backward": {"times": 3}}},
#     # ...     measure_ends)
#     # ([(0, 8), (0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, 24), (24, inf)])

#     # Repeat of a single measure:

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}, "backward": {"times": 2}}},
#     # ...     measure_ends)
#     # ([(0, 4), (0, 4), (4, inf)], [(0, 4), (4, 8), (8, inf)])

#     # First forward repeat not at start.

#     # >>> get_repeat_segments2(
#     # ...     {2: {"forward": {"times": 2}},
#     # ...      3: {"backward": {"times": 2}}},
#     # ...     measure_ends)
#     # ([(0, 4), (4, 12), (4, 12), (12, inf)], [(0, 4), (4, 12), (12, 20), (20, inf)])

#     # Multiple repeats.

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"backward": {"times": 2}},
#     # ...      3: {"forward": {"times": 2}},
#     # ...      6: {"backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 8), (0, 8), (8, 24), (8, 24), (24, inf)]

#     # Repeats with a gap.

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"backward": {"times": 2}},
#     # ...      4: {"forward": {"times": 2}},
#     # ...      6: {"backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 8), (0, 8), (8, 12), (12, 24), (12, 24), (24, inf)]

#     # Endings.

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
#     # ...      3: {"start-ending": {"number": "2"}}},
#     # ...     measure_ends)[0]
#     # [(0, 4), (4, 8), (0, 4), (8, inf)]

#     # Gap between endings and next repeat.

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
#     # ...      3: {"start-ending": {"number": "2"}},
#     # ...      5: {"forward": {"times": 2}, "backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 4), (4, 8), (0, 4), (8, 16), (16, 20), (16, 20), (20, inf)]

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"start-ending": {"number": "1, 2"}, "backward": {"times": 3}},
#     # ...      3: {"start-ending": {"number": "3"}},
#     # ...      5: {"forward": {"times": 2}, "backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 4), (4, 8), (0, 4), (4, 8), (0, 4), (8, 16), (16, 20), (16, 20), (20, inf)]

#     # Missing endings after a backward repeat are inferred (note that the second
#     # ending here is missing):

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 4), (4, 8), (0, 4), (8, inf)]

#     # >>> get_repeat_segments2(
#     # ...     {1: {"forward": {"times": 2}},
#     # ...      2: {"start-ending": {"number": "1"}, "backward": {"times": 2}},
#     # ...      4: {"forward": {"times": 2}, "backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 4), (4, 8), (0, 4), (8, 12), (12, 16), (12, 16), (16, inf)]

#     >>> get_repeat_segments2(
#     ...     {2: {"forward": {"times": 2}},
#     ...      3: {"start-ending": {"number": "1, 2"}, "backward": {"times": 3}},
#     ...      4: {"start-ending": {"number": "3"}, "backward": {"times": 2}},
#     ...      5: {"start-ending": {"number": "4"}},
#     ...      6: {"forward": {"times": 2}},
#     ...      8: {"start-ending": {"number": "1"}},
#     ...      9: {"backward": {"times": 2}},
#     ...      10: {"start-ending": {"number": "2"}}},
#     ...     measure_ends)[0]
#     [(0, 4), (4, 8), (8, 12), (4, 8), (8, 12), (4, 8), (12, 16), (4, 8), (16, 20), (20, 28), (28, 36), (20, 28), (36, inf)]

#     # We infer missing forward repeats:

#     # >>> get_repeat_segments2(
#     # ...     {2: {"backward": {"times": 2}}},
#     # ...     measure_ends)
#     # ([(0, 8), (0, 8), (8, inf)], [(0, 8), (8, 16), (16, inf)])

#     # >>> get_repeat_segments2(
#     # ...     {2: {"backward": {"times": 2}},
#     # ...      4: {"backward": {"times": 2}}},
#     # ...     measure_ends)[0]
#     # [(0, 8), (0, 8), (8, 16), (8, 16), (16, inf)]
#     """
#     if not repeats:
#         return ([(0, float("inf"))], [(0, float("inf"))])
#     assert min(repeats) > 0
#     assert min(measure_ends) > 0

#     last_forward_onset = None
#     last_backward_onset = None
#     last_forward_measure_num = 1
#     continue_from = None
#     continue_from_measure_num = None
#     last_ending_onset = None
#     repeat_time = None
#     n_times = None
#     last_measure = max(repeats)

#     # after a da capo or dal segno initial_repeat_time can become 2
#     initial_repeat_time = 1

#     if 1 not in repeats:
#         repeats[1] = {"continue": True}
#     for measure_num in list(repeats.keys()):
#         if (
#             "backward" in repeats[measure_num]
#             and measure_num + 1 not in repeats
#         ):
#             repeats[measure_num + 1] = {"continue": True}

#     sorted_repeats = sorted(repeats.keys())
#     measure_iter = iter(sorted_repeats)
#     measure_num = next(measure_iter)

#     segments = []

#     while measure_num <= last_measure:
#         print(measure_num)
#         repeat_dict = repeats[measure_num]
#         if "continue" in repeat_dict:
#             onset = _get_onset_from_measure_num(measure_num - 1, measure_ends)
#             continue_from = onset
#             continue_from_measure_num = measure_num
#         if "forward" in repeat_dict:
#             onset = _get_onset_from_measure_num(measure_num - 1, measure_ends)
#             if last_ending_onset is not None and onset > last_ending_onset:
#                 # there was a missing second ending or similar somewhere
#                 segments.append((last_forward_onset, last_ending_onset))
#                 last_ending_onset = None
#                 repeat_time = None
#             if repeat_time is None:
#                 repeat_time = initial_repeat_time
#             if "start-ending" in repeat_dict:
#                 warnings.warn(
#                     f"M. {measure_num} has 'forward' repeat bar and "
#                     "'start-ending'; ignoring 'start-ending'"
#                 )
#             last_forward_onset = onset
#             last_forward_measure_num = measure_num

#             if continue_from is not None and continue_from != onset:
#                 segments.append((continue_from, onset))
#             continue_from = None
#             continue_from_measure_num = None
#         elif "start-ending" in repeat_dict:
#             ending_numbers = [
#                 int(x)
#                 for x in re.findall(
#                     r"\d+", repeat_dict["start-ending"]["number"]
#                 )
#             ]
#             if repeat_time is None:
#                 # missing forward repeat
#                 repeat_time = initial_repeat_time
#             if repeat_time not in ending_numbers:
#                 # go to next ending
#                 while "backward" not in repeats[measure_num]:
#                     measure_num = next(measure_iter)
#                 measure_num = next(measure_iter)
#                 continue
#             onset = _get_onset_from_measure_num(measure_num - 1, measure_ends)
#             if last_ending_onset is None:
#                 segments.append((last_forward_onset, onset))
#             else:
#                 segments.append((last_forward_onset, last_ending_onset))
#             if n_times is not None and repeat_time == n_times:
#                 last_ending_onset = None
#                 repeat_time = None
#                 continue_from = onset
#                 continue_from_measure_num = measure_num
#             else:
#                 last_ending_onset = onset
#         if "backward" in repeat_dict:
#             n_times = repeat_dict["backward"]["times"]
#             if repeat_time is None:
#                 repeat_time = initial_repeat_time
#             if last_forward_onset is None:
#                 last_forward_onset = continue_from
#                 last_forward_measure_num = continue_from_measure_num
#             onset = _get_onset_from_measure_num(measure_num, measure_ends)
#             last_backward_onset = onset
#             if last_ending_onset is not None:
#                 segments.append((last_ending_onset, onset))
#             else:
#                 segments.append((last_forward_onset, onset))

#             if repeat_time < repeat_dict["backward"]["times"]:
#                 repeat_time += 1
#                 measure_iter = iter(sorted_repeats)
#                 measure_num = next(measure_iter)
#                 while measure_num < last_forward_measure_num:
#                     measure_num = next(measure_iter)
#                 continue
#             repeat_time = None
#         if measure_num == last_measure:
#             break
#         measure_num = next(measure_iter)

#     if last_ending_onset is not None:
#         # the last ending was missing
#         segments.append((last_forward_onset, last_ending_onset))
#         continue_from = last_backward_onset
#     if continue_from is not None:
#         segments.append((continue_from, float("inf")))
#     else:
#         segments.append((segments[-1][1], float("inf")))
#     segment_durs = [segment[1] - segment[0] for segment in segments]
#     segment_ends = list(itertools.accumulate(segment_durs))
#     segment_starts = [0] + segment_ends[:-1]
#     return segments, list(zip(segment_starts, segment_ends))


# if __name__ == "__main__":
#     measure_ends = {i + 1: (i + 1) * 4 for i in range(16)}
#     print(
#         get_repeat_segments2(
#             {
#                 2: {"forward": {"times": 2}},
#                 3: {
#                     "start-ending": {"number": "1, 2"},
#                     "backward": {"times": 3},
#                 },
#                 4: {"start-ending": {"number": "3"}, "backward": {"times": 2}},
#                 5: {"start-ending": {"number": "4"}},
#                 6: {"forward": {"times": 2}},
#                 8: {"start-ending": {"number": "1"}},
#                 9: {"backward": {"times": 2}},
#                 10: {"start-ending": {"number": "2"}},
#             },
#             measure_ends,
#         )[0]
#     )
