# Bug: `split_notes_at_barlines` doesn't split notes crossing the first bar

## Summary

`split_notes_at_barlines()` in `music_df/split_notes.py` doesn't split notes that start **before** the first bar's onset and extend **past** it. It only checks if `note.release > bar.release` (note extends past the end of the bar's measure), never whether a note starts before `bar.onset` and crosses it.

This works for complete pieces (which always have a bar at onset 0) but fails for mid-measure crops where the first bar isn't at onset 0.

## Minimal reproduction

```python
import pandas as pd
from music_df.split_notes import split_notes_at_barlines

# Note [1.75, 2.0) crosses bar at 1.9375, but there's no bar before it
df = pd.DataFrame({
    "type": ["note", "note", "note", "bar", "note"],
    "onset":   [0.0,    1.5,    1.75,   1.9375, 2.0],
    "release": [1.5,    1.75,   2.0,    4.9375, 2.3125],
    "pitch":   [50,     53,     58,     None,   62],
    "tie_to_next": [False] * 5,
    "tie_to_prev": [False] * 5,
})
result = split_notes_at_barlines(df, min_overhang_dur=1/16)
note58 = result[result["pitch"] == 58]
assert len(note58) == 1  # BUG: should be 2 (split at 1.9375)
```

Adding a bar at onset 0 fixes it:

```python
df_fixed = pd.DataFrame({
    "type": ["bar", "note", "note", "note", "bar", "note"],
    "onset":   [0.0,    0.0,    1.5,    1.75,   1.9375, 2.0],
    "release": [1.9375, 1.5,    1.75,   2.0,    4.9375, 2.3125],
    "pitch":   [None,   50,     53,     58,     None,   62],
    "tie_to_next": [False] * 6,
    "tie_to_prev": [False] * 6,
})
result2 = split_notes_at_barlines(df_fixed, min_overhang_dur=1/16)
note58 = result2[result2["pitch"] == 58]
assert len(note58) == 2  # correctly split into [1.75, 1.9375) and [1.9375, 2.0)
```

## Root cause

In `split_notes.py:88-112`, the function iterates forward through bars for each note:

```python
for final_bars_i in range(bars_i, len(bars)):
    bar_release = bars.loc[final_bars_i].release
    if row.release > bar_release:
        overhang = True
        ...
```

It checks whether `row.release > bar_release`, i.e., whether the note extends past the **end** of the bar's measure. But when a note starts before `bars.loc[0].onset` (the first barline), there's no earlier bar whose `release` equals this barline's `onset`, so the crossing is never detected.

## Downstream effect

The unsplit note produces a kern token whose duration extends past the barline. After humdrum collation (`timebase` + `assemble`), different spines reach barlines at different times, causing `merge_spines` to raise:

```
ValueError: barline token '=' in column N but not all columns are barlines.
This usually means the input spines have misaligned barlines (e.g. from a
mid-measure crop).
```

## Secondary issue: hardcoded `Meter("4/4")` in `_align_pickup_durations`

`_align_pickup_durations` in `humdrum_export.py:182` hardcodes `Meter("4/4")` when decomposing rest durations. This doesn't cause the current bug (the total duration is correct regardless of meter), but it produces suboptimal rest decompositions for non-4/4 time signatures. It should read the meter from the spine's `*M` token instead.

## Suggested fix

In `split_notes_at_barlines`, handle notes that start before `bars.loc[0].onset`. For example, before the main loop, check if any notes start before the first bar and extend past it, then split them at the first bar's onset. Alternatively, add a synthetic bar at the minimum note onset so the existing logic handles it.
