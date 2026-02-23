# Bug: `_reconstruct_degree_column` drops alterations and mishandles mode suffixes

## Location

`music_df/harmony/modulation.py`, lines 289-295:

```python
def _reconstruct_degree_column(chord_df: pd.DataFrame) -> pd.Series:
    secondary_with_slash = "/" + chord_df["secondary_degree"]
    if "secondary_mode" in chord_df.columns:
        mode_suffix = chord_df["secondary_mode"].fillna("_").replace("_", "")
        secondary_with_slash = secondary_with_slash + mode_suffix
    secondary_with_slash = secondary_with_slash.replace("/I", "")
    return chord_df["primary_degree"] + secondary_with_slash
```

This function is called at the end of `remove_long_tonicizations` (line 865),
`remove_short_modulations` (line 1141), and `remove_phantom_keys` (line 1373).

## Three issues

### 1. Primary alteration is dropped

The function builds the degree string from `primary_degree` alone, ignoring
`primary_alteration`. When the input uses the split-column convention (where
`primary_degree="II"` and `primary_alteration="b"` represent `bII`), the
reconstructed degree becomes `"II"` instead of `"bII"`.

This silently strips every chromatic alteration on primary degrees: `bII` becomes
`II`, `#IV` becomes `IV`, `bVI` becomes `VI`, etc.

### 2. Secondary alteration is dropped

Same problem on the secondary side: `secondary_alteration` is not prepended to
`secondary_degree`. This is less visible because `remove_short_modulations` stores
alterations inside `secondary_degree` directly (via `spelled_pitch_to_rn`, which
returns values like `"bII"`), but any code path that keeps alterations separate
will lose them.

### 3. Mode suffix prevents "/I" stripping

`secondary_with_slash.replace("/I", "")` uses pandas `Series.replace`, which does
**exact value matching**. When a mode suffix has been appended, the value is
`"/IM"` or `"/Im"` rather than `"/I"`, so the replacement silently fails. This
produces malformed degree strings like `"V/bIIM"` instead of `"V/bII"`.

## Impact

These bugs are triggered whenever the input DataFrame has **split columns**
(`primary_degree`, `primary_alteration`, etc.) alongside a `degree` column. The
three normalization functions (`remove_long_tonicizations`,
`remove_short_modulations`, `remove_phantom_keys`) all reconstruct the `degree`
column at the end, overwriting the correct value produced by
`split_degrees_to_single_degree`.

Downstream code that re-parses the `degree` column back into split columns (via
`single_degree_to_split_degrees`) then gets wrong results. In the
`rn_corpora_loader` project, the `normalize_modulations` transform does exactly
this sequence: `split_degrees_to_single_degree` -> three normalization steps ->
`single_degree_to_split_degrees`. The re-parsing at the end reads the broken
degree column.

Observed effects on real corpus data:

- `bII` chords drop from 835 to 1 (dcml) and 68 to 1 (when_in_rome) because the
  `"b"` alteration is silently stripped.
- `#VII`, `bVI`, `#IV`, and all other altered primary degrees are similarly
  affected.
- Spurious lowercase primary degrees appear (`"v"`, `"vi"`, `"iv"`, etc.) because
  `replace_spurious_tonics` moves lowercase secondary degrees (from
  `spelled_pitch_to_rn`) into `primary_degree`, and the degree column preserves
  that case through reconstruction.

## Reproducer

```python
import pandas as pd
from music_df.harmony.modulation import _reconstruct_degree_column

df = pd.DataFrame({
    "primary_degree": ["II", "VII", "V", "I"],
    "primary_alteration": ["b", "#", "_", "_"],
    "secondary_degree": ["I", "I", "II", "I"],
    "secondary_alteration": ["_", "_", "b", "_"],
    "secondary_mode": ["_", "_", "M", "M"],
})

result = _reconstruct_degree_column(df)
print(result.tolist())
# Actual:   ['II',  'VII',  'V/IIM', 'I/IM']
# Expected: ['bII', '#VII', 'V/bIIM', 'I']
```

## Suggested fix

`_reconstruct_degree_column` needs to:

1. Prepend `primary_alteration` (replacing the null sentinel `"_"` with `""`)
   before `primary_degree`.
2. Prepend `secondary_alteration` (same sentinel handling) before
   `secondary_degree`.
3. Use a regex or string-method replacement anchored to the full value (e.g.,
   `str.replace(r"^/I[mM]?$", "", regex=True)`) instead of exact-match
   `Series.replace("/I", "")`, so that mode suffixes on bare `/I` are also
   stripped.

### Additional fix: `replace_spurious_tonics`

`replace_spurious_tonics` (line 873) moves `secondary_degree` into
`primary_degree` but does **not** move `secondary_alteration` into
`primary_alteration` or clear `primary_alteration`. When the secondary degree
includes its alteration (e.g., `"bII"` from `spelled_pitch_to_rn`), this mostly
works through the degree-column round-trip, but the split columns are transiently
inconsistent. It also does not uppercase lowercase Roman numerals that came from
`spelled_pitch_to_rn` (which returns lowercase for minor-mode keys like `"v"` for
G minor relative to C). These lowercase values end up as primary degrees, violating
the convention that primary degrees are always uppercase (`"I"`-`"VII"`).

The fix should:
- Move `secondary_alteration` to `primary_alteration` (and reset
  `secondary_alteration` to `"_"`) for the affected rows.
- Uppercase the Roman numeral portion of the moved degree (preserving any `b`/`#`
  alteration prefix).
