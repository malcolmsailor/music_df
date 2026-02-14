# music_df: Time signature placed at first note onset instead of piece start

## File

`music_df/conversions/ms3.py`, function `_add_time_sigs` (line 63)

## Bug

`_add_time_sigs()` places time signature rows at the onset of the first note
that has that time signature. For pieces where a rest precedes the first note,
the time signature ends up mid-measure instead of at onset 0. Since bars (added
later by `_add_bars_from_measures_df`) start at onset 0, the time signature
doesn't co-occur with any bar, triggering metricker's "mid-measure
time-signature not supported" warning.

## Affected pieces in chopin_mazurkas

- **BI126-3op41-1**: first note at quarterbeat 1/2 (rest fills beat 0 to 0.5)
- **BI77-4op17-4**: first note at quarterbeat 1 (rest fills beat 0 to 1.0)

## Fix

The first time signature should always start at the beginning of the piece
timeline (onset 0), not at the first note's onset:

```python
# In _add_time_sigs, after creating df_changes:
if len(df_changes) > 0:
    df_changes.iloc[0, df_changes.columns.get_loc("onset")] = min(
        0, df_changes.iloc[0]["onset"]
    )
```
