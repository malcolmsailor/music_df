# Bug: `get_repeat_segments` produces `inf` onsets when endings have no associated repeat barlines

## Symptom

Parsing the MusicXML score at
`When-in-Rome/Corpus/OpenScore-LiederCorpus/Hensel,_Fanny_(Mendelssohn)/6_Lieder,_Op.9/4_Die_frühen_Gräber/score.mxl`
with `xml_parse(..., expand_repeats="yes")` produces `inf` onset and release
values for all notes/events after the endings. This causes downstream crashes.

## Root cause

The MusicXML has `<ending>` elements at measures 20 and 21 but **no `<repeat>`
elements** (neither forward nor backward):

```xml
<!-- measure 20 -->
<barline location="left">
    <ending number="1, 2" type="start" default-y="59.39"/>
</barline>

<!-- measure 21 -->
<barline location="left">
    <ending number="3" type="start" default-y="40.39"/>
</barline>
```

This is an encoding error in the source MuseScore file (endings are meaningless
without a repeat), but the parser should handle it gracefully.

## How `inf` propagates

In `music_df/xml_parser/repeats.py`, `get_repeat_segments` receives:

```python
repeats = {
    20: {'start-ending': {'number': '1, 2'}},
    21: {'start-ending': {'number': '3'}},
}
```

Walking through the logic:

1. M20's `start-ending` sets `ending_jump_from = 76` and `ending_jump_to = 76`
   (lines 251-257).
2. M21's `start-ending` updates `ending_jump_to = 80` (line 255).
3. No `backward` repeat is ever encountered, so the endings are never properly
   resolved inside the loop.
4. After the loop (line 290), since `ending_jump_from is not None`, two segments
   are appended: `(0, 76)` and `(80, inf)`.
5. Then line 300 fires (`last_forward_repeat == last_boundary()`, both 0),
   adding two `simple_repeat` segments of `(0, 84)`.
6. `segment_durs` on line 309 computes `inf - 80 = inf`, and cumulative sums
   propagate `inf` into `repeated_segments`: `[(0, 76), (76, inf), (inf, inf), (inf, inf)]`.
7. `_expand_repeats` then adds `inf` offsets to note onsets.

## Suggested fix

When endings exist without a backward repeat, `get_repeat_segments` should
treat them as a no-op (as if no repeats were present) and optionally warn. One
approach: after the main loop, if `ending_jump_from is not None` and `segments`
is empty (i.e., no backward repeat was ever processed), return the trivial
`([(0, inf)], [(0, inf)], ["no_repeat"])` result (with a warning if
`warn=True`).

## Reproducer

```python
from music_df.xml_parser.repeats import get_repeat_segments
from fractions import Fraction

repeats = {
    20: {'start-ending': {'number': '1, 2'}},
    21: {'start-ending': {'number': '3'}},
}
measure_ends = {i + 1: Fraction(4 * (i + 1)) for i in range(21)}

orig, repeated, types = get_repeat_segments(repeats, measure_ends)
# repeated contains (inf, inf) entries
print(repeated)
# [(0, Fraction(76, 1)), (Fraction(76, 1), inf), (inf, inf), (inf, inf)]
```
