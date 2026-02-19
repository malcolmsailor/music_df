# Bug: Float drift causes KernDurError in humdrum export

## Summary

When rendering Beethoven_B076_annotator=a (tavern corpus) via `rn-corpora notate`,
seven measures are replaced with rests due to `KernDurError` ("Unrecognized
duration"), and the resulting PDF fails to render due to barline misalignment
between spines.

**Root cause:** Floating point imprecision in note onset/release values
propagates into the humdrum export pipeline, producing durations that the
`dur_to_kern` decomposition loop cannot represent within its tolerance.

## Error trace

```
Replacing measure contents with rest due to Unrecognized duration 0.4170000000000016 with offset=1.0829999999999984 in meter=Meter('3/8')
...
PDF rendering failed: Line 637: barline token '=' in column 1 but not all columns are barlines. Full line: '4.r\t=\t=\t=\t='
```

## Analysis

### Where the float drift originates

The `totable` binary outputs onset/release values as decimal strings. For
triplet rhythms (denominator 3, 6, 12, ...), these are truncated decimal
approximations:

- Triplet eighth note: exact duration = 1/6 = 0.16666..., `totable` outputs
  something like `"0.1667"`
- `pd.read_csv` parses this to `float64`, permanently losing exactness

The per-note error is ~3.3e-5 (one rounding unit at 4 decimal places), but it
**accumulates** across consecutive triplet events. After ~15 consecutive
triplets, the drift reaches ~0.0005.

For Beethoven B076:

- 405 note onsets have measurable float drift from a 1/96 grid
- 335 note releases have drift
- Maximum accumulated drift: ~0.0005

### How it causes KernDurError

The drift manifests when computing rests or note durations that involve
triplet-imprecise values. For example:

```
offset = 1.0829999999999984  (should be 13/12 = 1.08333...)
duration = 0.4170000000000016  (should be 5/12 = 0.41666...)
offset + duration = 1.5  (exact: end of 3/8 measure)
```

The duration 5/12 is not a single standard note value --- it requires
decomposition (e.g., 1/4 + 1/8 + 1/24 tied together). The decomposition
happens in the divmod fallback loop in `dur_to_kern`
(`music_df/humdrum_export/dur_to_kern.py:269-293`):

```python
for divisor in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96]:
    whole, remainder = divmod(remainder, 1 / divisor)
    ...
    if remainder < 1e-6:   # <-- tolerance too tight
        break
```

For `d = 0.4170000000000016`:

| Step | divisor | whole | remainder |
|------|---------|-------|-----------|
| 1 | 4 | 0.25 | 0.1670... |
| 2 | 8 | 0.125 | 0.0420... |
| 3 | 24 | 1/24 | **~0.0003** |

The remainder ~0.0003 is 300x larger than the 1e-6 tolerance, so the loop
fails to break and raises `KernDurError`.

### Secondary effect: barline misalignment

When `KernDurError` is caught in `df_to_spines.py:get_kern_spine`,
`skip_measure` is set to `True`, replacing the entire measure with a rest. This
causes barline misalignment after collation (timebase + assemble) because:

1. The skipped spine has a single dotted-quarter rest where other spines have
   multiple note tokens
2. `merge_spines` detects the misalignment and raises `ValueError`

Fixing the primary float drift issue should eliminate this symptom.

### Additional minor bug: wrong `measure_start_time` in skip_measure

In `df_to_spines.py:346-349`, the skip_measure rest handler passes `0.0` as
`measure_start_time`:

```python
handle_rest(
    measure_start,   # start_time
    row.onset,       # end_time
    0.0,             # measure_start_time  <-- should be measure_start
    meter,
    ...
)
```

This makes `measure_offset = measure_start - 0 = measure_start` (an absolute
time) instead of `0` (the correct offset at the start of the measure). For
common meters this is benign because metricker's metric structure repeats, but
it's still incorrect and could cause issues in edge cases.

Compare with the normal rest handler on line 362-365 which correctly passes
`measure_start`.

## Proposed fix

### Primary: snap durations to a grid in `dur_to_kern`

In `music_df/humdrum_export/dur_to_kern.py`, snap `inp` and `offset` to the
nearest 1/96 grid at the entry to `dur_to_kern`. 96 is the finest subdivision
already used in the divmod fallback loop and covers all standard musical
subdivisions (triplets through 32nd-note triplets, regular values through
64th notes).

```python
# At the top of dur_to_kern, after the meter conversion:
SNAP_GRID = 96
inp = round(float(inp) * SNAP_GRID) / SNAP_GRID
offset = round(float(offset) * SNAP_GRID) / SNAP_GRID
```

**Why this works:** For `inp = 0.4170000000000016`:

- `0.4170000000000016 * 96 = 40.032` -> `round(40.032) = 40`
- `40 / 96 = 0.41666...` = exact float representation of 5/12

The subsequent divmod decomposition of `0.41666...` produces remainder ~3.5e-18
(normal float arithmetic noise), well under the 1e-6 tolerance.

**Safety margin:** The maximum observed drift (~0.0005) is far below half a grid
step (1/192 ~ 0.0052), so snapping will always round to the correct grid point.

### Secondary: fix `measure_start_time` in skip_measure

In `df_to_spines.py:349`, change `0.0` to `measure_start`:

```python
handle_rest(
    measure_start,
    row.onset,
    measure_start,   # was 0.0
    meter,
    ...
)
```

### Verified behavior

After snapping, all three failing cases succeed. All existing doctests produce
identical kern notation with and without the snap.

Actual outputs after snapping:

```python
>>> dur_to_kern(0.4170000000000016, 1.0829999999999984, "3/8")  # 5/12 with drift
[(0.25, '16'), (0.16666666666666674, '24')]
# i.e., 16th + triplet 8th tied

>>> dur_to_kern(0.9170000000000016, 0.5829999999999984, "3/8")  # 11/12 with drift
[(0.41666666666666663, '18...'), (0.5, '8')]
# i.e., triple-dotted 18th + 8th tied

>>> dur_to_kern(1.2919999999999732, 0.20800000000002683, "3/8")  # 31/24 with drift
[(0.29166666666666663, '24..'), (0.5, '8'), (0.5, '8')]
```

### Caveat: non-standard recip values for triplet durations

The snapping fix correctly prevents the crash, but exposes a pre-existing
limitation in `duration_float_to_recip`: it can produce non-standard recip
values like `18...` (triple-dotted 18th note) for durations like 5/12. While
mathematically correct (4/18 * 15/8 = 5/12), this is not standard music
notation. The correct notation would be tied standard values (e.g., triplet
quarter + triplet sixteenth = 1/3 + 1/12 = 5/12).

This is a separate, pre-existing issue in `duration_float_to_recip` that
affects any compound triplet duration. Whether verovio renders `18...` correctly
should be tested after applying the fix.

### Tests to add

```python
# In dur_to_kern.py doctests:

# These should not raise KernDurError (previously caused crash)
>>> dur_to_kern(0.4170000000000016, 1.0829999999999984, "3/8")
[(0.25, '16'), (0.16666666666666674, '24')]
>>> dur_to_kern(0.9170000000000016, 0.5829999999999984, "3/8")
[(0.41666666666666663, '18...'), (0.5, '8')]
```

## Files to modify

| File | Change |
|------|--------|
| `music_df/humdrum_export/dur_to_kern.py` | Add grid snapping at top of `dur_to_kern` |
| `music_df/humdrum_export/df_to_spines.py:349` | Fix `measure_start_time` argument |
| `tests/humdrum_export_tests/test_dur_to_kern.py` (or doctests) | Add float-drift test cases |

## Future work

- Investigate whether `duration_float_to_recip` should decompose compound
  triplet durations (like 5/12) into tied standard note values instead of
  producing non-standard recip values like `18...`
- Consider adding grid snapping in `read_krn.py` at the data source to prevent
  float drift from propagating through the entire pipeline
