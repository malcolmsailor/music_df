from __future__ import annotations

import typing as t
import warnings
from collections import defaultdict, deque

from music_df.xml_parser.objects import Note


def merge_ties(notes: t.Iterable[Note]) -> t.List[Note]:
    """
    Assumes that notes are in sorted order by onset.

    >>> len(merge_ties([Note(60, 0.0, 1.0), Note(60, 1.0, 1.0)]))
    2
    >>> len(
    ...     merge_ties(
    ...         [
    ...             Note(60, 0.0, 1.0, tie_to_next=True),
    ...             Note(60, 1.0, 2.0, tie_to_prev=True),
    ...         ]
    ...     )
    ... )
    1
    >>> len(
    ...     merge_ties(
    ...         [
    ...             Note(60, 0.0, 1.0, tie_to_next=True),
    ...             Note(60, 1.0, 2.0, tie_to_next=True, tie_to_prev=True),
    ...             Note(60, 2.0, 3.0, tie_to_prev=True),
    ...             Note(60, 3.0, 4.0, tie_to_next=True),
    ...             Note(60, 4.0, 5.0, tie_to_prev=True),
    ...         ]
    ...     )
    ... )
    2

    Although `tie_to_prev` is redundant for our purposes, it is specified in
    musicxml files and provides an extra check on syntactic correctness so
    it must be provided. Since occasionally dangling ties occur (either due
    to input errors or due to notes in scores that are to be "left ringing")
    dangling ties emit a warning rather than an exception.

    >>> import warnings
    >>> warnings.simplefilter("error")
    >>> merge_ties([Note(60, 0.0, 1.0, tie_to_next=True), Note(60, 1.0, 2.0)])
    Traceback (most recent call last):
    UserWarning: dangling tie at 60:0.0-1.0⌒

    For a similar reason the same warning is emitted if the last Note is
    tied:

    >>> merge_ties([Note(60, 0.0, 1.0), Note(60, 1.0, 2.0, tie_to_next=True)])
    Traceback (most recent call last):
    UserWarning: dangling tie at 60:1.0-2.0⌒

    Notes that would otherwise be tied but for having different piches
    are considered dangling ties.
    >>> merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(62, 1.0, 2.0, tie_to_prev=True),
    ...     ],
    ... )
    Traceback (most recent call last):
    UserWarning: dangling tie at 60:0.0-1.0⌒

    Here we check that when we emit the warning for the last note, we still
    treat any preceding ties appropriately:

    >>> warnings.simplefilter("ignore")
    >>> merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 1.0, 2.0, tie_to_prev=True, tie_to_next=True),
    ...     ]
    ... )
    [Note(pitch=60, onset=0.0, release=2.0, tie_to_next=False, tie_to_prev=False, ...

    There is a tricky case we need to handle where there is a dangling tie
    immediately followed by another tie:
    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 1.0, 2.0, tie_to_prev=True, tie_to_next=True),  # dangling
    ...         Note(61, 2.0, 3.0, tie_to_next=True),
    ...         Note(61, 3.0, 5.0, tie_to_prev=True),
    ...         Note(62, 5.0, 6.0, tie_to_next=True),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=2.0, tie_to_next=False, tie_to_prev=False, ...
    >>> result[1]
    Note(pitch=61, onset=2.0, release=5.0, tie_to_next=False, tie_to_prev=False, ...
    >>> result[2]
    Note(pitch=62, onset=5.0, release=6.0, tie_to_next=False, tie_to_prev=False, ...

    Because it seems that scores sometimes contain tied notes that have gaps
    before the next tied note, we allow the release of the first note to be <
    the onset of the next note, but only if there haven't been any intervening
    notes with the same pitch. We then emit a warning.

    >>> warnings.simplefilter("error")
    >>> merge_ties(
    ...     [
    ...         Note(60, 0.0, 2.0, tie_to_next=True),
    ...         Note(60, 2.5, 3.0, tie_to_prev=True),
    ...     ],
    ... )
    Traceback (most recent call last):
    UserWarning: Release of note at 2.0 < onset of note at 2.5

    If there is an intervening note with the same pitch, we raise an exception:

    >>> merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 1.0, 1.5),
    ...         Note(60, 1.5, 3.0, tie_to_prev=True),
    ...     ],
    ... )
    Traceback (most recent call last):
    ValueError: Release of note at 1.0 < onset of note at 1.5 and intervening note(s).

    However we create a special case for when the intervening note sounds at the
    same time as the note that starts the tie:

    >>> merge_ties(
    ...     [
    ...         Note(
    ...             pitch=44,
    ...             onset=0.0,
    ...             release=0.5,
    ...             tie_to_next=True,
    ...             tie_to_prev=False,
    ...         ),
    ...         Note(
    ...             pitch=44,
    ...             onset=0.0,
    ...             release=1.0,
    ...             tie_to_next=False,
    ...             tie_to_prev=False,
    ...         ),
    ...         Note(
    ...             pitch=44,
    ...             onset=1.0,
    ...             release=2.0,
    ...             tie_to_next=False,
    ...             tie_to_prev=True,
    ...         ),
    ...     ]
    ... )
    Traceback (most recent call last):
    UserWarning: Release of note at 0.5 < onset of note at 1.0

    If the release of the first note is > the onset of the next note, we raise
    an exception.

    >>> merge_ties(
    ...     [
    ...         Note(60, 0.0, 2.0, tie_to_next=True),
    ...         Note(60, 1.5, 3.0, tie_to_prev=True),
    ...     ],
    ... )
    Traceback (most recent call last):
    ValueError: Release of note at 2.0 > onset of note at 1.5

    Check the handling of ties across different "parts":
    >>> warnings.simplefilter("ignore")
    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(67, 0.5, 2.0, tie_to_next=True),
    ...         Note(60, 1.0, 2.0, tie_to_prev=True),
    ...         Note(67, 2.0, 3.0, tie_to_prev=True),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=2.0, tie_to_next=False, tie_to_prev=False, ...
    >>> result[1]
    Note(pitch=67, onset=0.5, release=3.0, tie_to_next=False, tie_to_prev=False, ...

    Check the handling of a tie in one part at the same time as a unison in
    another:
    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 1.0, tie_to_next=True),
    ...         Note(60, 0.0, 1.0),
    ...         Note(60, 1.0, 2.0, tie_to_prev=True),
    ...         Note(67, 1.0, 2.0),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=1.0, tie_to_next=False, tie_to_prev=False, ...
    >>> result[1]
    Note(pitch=60, onset=0.0, release=2.0, tie_to_next=False, tie_to_prev=False, ...
    >>> result[2]
    Note(pitch=67, onset=1.0, release=2.0, tie_to_next=False, tie_to_prev=False, ...

    We can use 'voice' to differentiate ties that would otherwise be
    unparseable:
    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 3.0, tie_to_next=True, voice="1"),
    ...         Note(60, 0.0, 1.5, tie_to_next=True, voice="2"),
    ...         Note(60, 1.5, 1.75, tie_to_prev=True, voice="2"),
    ...         Note(60, 3.0, 4.5, tie_to_prev=True, voice="1"),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=1.75, tie_to_next=False, tie_to_prev=False, grace=False, voice='2', ...
    >>> result[1]
    Note(pitch=60, onset=0.0, release=4.5, tie_to_next=False, tie_to_prev=False, grace=False, voice='1', ...

    On the other hand, when there are cases where there is
        1. a note with `tie_to_next` in one voice
        2. a note with `tie_to_prev` in another voice
        3. the release of the first note <= the onset of the next note
        4. there is no upcoming note at the same onset as
            the note with `tie_to_prev` with the same voice and pitch of
            the note with `tie_to_next`
    I believe we can perform the tie. (This seems to be something like
    what MuseScore does.) In this case the resulting note has the "voice"
    attribute of the first note.

    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 3.0, tie_to_next=True, voice="1"),
    ...         Note(60, 3.0, 4.5, tie_to_prev=True, voice="2"),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=4.5, tie_to_next=False, tie_to_prev=False, grace=False, voice='1', ...

    >>> result = merge_ties(
    ...     [
    ...         Note(60, 0.0, 3.0, tie_to_next=True, voice="2"),
    ...         Note(60, 3.0, 4.5, tie_to_prev=True, voice="1"),
    ...         Note(60, 3.0, 4.0, tie_to_prev=True, voice="2"),
    ...     ]
    ... )
    >>> result[0]
    Note(pitch=60, onset=0.0, release=4.0, tie_to_next=False, tie_to_prev=False, grace=False, voice='2', ...
    >>> result[1]
    Note(pitch=60, onset=3.0, release=4.5, tie_to_next=False, tie_to_prev=False, grace=False, voice='1', ...
    """

    # TODO only tie notes in same part?
    # TODO we may need to handle ties involving grace notes
    def _check_pair(note1: Note, note2: Note):
        assert note1.tie_to_next, "`tie_to_next` is False"
        # it's ok to have dangling ties, but we shouldn't call
        #   _check_pair if the tie is dangling, so this assertion
        #   just checks the correctness of the code.
        assert note2.tie_to_prev, "`tie_to_prev` is False"
        if note1.pitch != note2.pitch:
            raise ValueError(
                f"Tied notes have different pitches {note1.pitch} " f"and {note2.pitch}"
            )
        # if not math.isclose(note1.release, note2.onset):
        if note1.release < note2.onset:
            if allow_gap[note1.pitch]:
                warnings.warn(
                    f"Release of note at {note1.release} < "
                    f"onset of note at {note2.onset}"
                )
            else:
                raise ValueError(
                    f"Release of note at {note1.release} < "
                    f"onset of note at {note2.onset} and intervening note(s)."
                )
        elif note1.release > note2.onset:
            raise ValueError(
                f"Release of note at {note1.release} > "
                f"onset of note at {note2.onset}"
            )

    def _clear_queue(queue):
        if queue[-1].tie_to_next:
            warnings.warn(f"dangling tie at {queue[-1]}")
        first_note = queue.popleft()
        note1 = first_note
        while queue:
            note2 = queue.popleft()
            _check_pair(note1, note2)
            note1 = note2
        # _check_pair(note1, note2)
        new_note = first_note.copy(remove_ties=True)
        new_note.release = note1.release
        out.append(new_note)
        queue.clear()
        if new_note.pitch in allow_gap:
            del allow_gap[new_note.pitch]

    out = []

    notes = sorted(notes, key=lambda note: (note.onset, note.release))

    allow_gap = defaultdict(lambda: True)
    queues = defaultdict(lambda: defaultdict(deque))
    for i, note in enumerate(notes):
        queue = queues[note.voice][note.pitch]
        if queue and not note.tie_to_prev and not note.onset == queue[-1].onset:
            allow_gap[note.pitch] = False
        if note.tie_to_prev:
            if not queue:
                for other_voice in queues:
                    other_queue = queues[other_voice][note.pitch]
                    if other_queue and note.onset >= other_queue[-1].release:
                        # We now want to check whether there is another
                        #   note coming up in the same voice that has
                        #   the same onset. (If it were *earlier* in notes
                        #   it would already have been processed.)
                        j = i
                        other_exists = False
                        while (
                            j < len(notes) and notes[j].onset == other_queue[-1].release
                        ):
                            if (
                                notes[j].voice == other_voice
                                and notes[j].pitch == note.pitch
                            ):
                                other_exists = True
                                break
                            j += 1
                        if not other_exists:
                            # carry out tie
                            queue = other_queue
                            break
            queue.append(note)
            if not note.tie_to_next:
                _clear_queue(queue)
        elif note.tie_to_next:
            if queue and not note.tie_to_prev:
                _clear_queue(queue)
            queue.append(note)
        else:
            out.append(note.copy(remove_ties=True))
    for voice_queue in queues.values():
        for queue in voice_queue.values():
            if queue:
                if queue[-1].tie_to_next:
                    warnings.warn(f"dangling tie at {queue[-1]}")
                _clear_queue(queue)
    return sorted(out, key=lambda note: (note.onset, note.release))
