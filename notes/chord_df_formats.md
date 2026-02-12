# chord_df Format Reference

This document describes the three chord annotation formats used by `music_df.chord_df`
and the two RN string dialects used by `music_df.harmony.chords`.

## Formats

### Joined

| Column      | Type                            | Example                                |
|-------------|---------------------------------|----------------------------------------|
| `onset`     | numeric / Fraction              | `0.0`                                  |
| `key`       | str                             | `"C"`, `"f#"`                          |
| `degree`    | str                             | `"I"`, `"V/V"`, `"#VI/bII"`, `"V/Vm"`  |
| `quality`   | str                             | `"M"`, `"m"`, `"d"`, `"Mm7"`, `"aug6"` |
| `inversion` | int (0-3)                       | `0`                                    |
| `release`   | numeric / Fraction *(optional)* | `4.0`                                  |
| `chord_pcs` | str *(optional)*                | `"047"`                                |

The `degree` column encodes the Roman numeral with optional secondary
(tonicization) and secondary mode, but does **not** include quality or
inversion information. Constants: `JOINED_FORMAT_REQUIRED_COLUMNS`.

### Split

| Column                 | Type                            | Example                             |
|------------------------|---------------------------------|-------------------------------------|
| `onset`                | numeric / Fraction              | `0.0`                               |
| `key`                  | str                             | `"C"`                               |
| `primary_degree`       | str                             | `"V"`, `"VII"`                      |
| `primary_alteration`   | str                             | `"_"`, `"#"`, `"b"`, `"##"`, `"bb"` |
| `secondary_degree`     | str                             | `"I"` (default), `"V"`, `"bII"`     |
| `secondary_alteration` | str                             | `"_"`, `"b"`                        |
| `quality`              | str                             | `"M"`                               |
| `inversion`            | int (0-3)                       | `0`                                 |
| `secondary_mode`       | str *(optional)*                | `"_"`, `"m"`, `"M"`                 |
| `release`              | numeric / Fraction *(optional)* | `4.0`                               |
| `chord_pcs`            | str *(optional)*                | `"047"`                             |

Decomposes the joined `degree` into separate columns. `"_"` is the null
alteration character. When no secondary degree exists, defaults are
`secondary_degree="I"`, `secondary_alteration="_"`, `secondary_mode="_"`.
Constants: `SPLIT_FORMAT_REQUIRED_COLUMNS`, `SPLIT_FORMAT_OPTIONAL_COLUMNS`.

### RN

| Column      | Type                            | Example                            |
|-------------|---------------------------------|------------------------------------|
| `onset`     | numeric / Fraction              | `0.0`                              |
| `key`       | str                             | `"C"`, `""` (forward-filled)       |
| `rn`        | str                             | `"I"`, `"V/V"`, `"iv6"`, `"VM/Vm"` |
| `release`   | numeric / Fraction *(optional)* | `4.0`                              |
| `chord_pcs` | str *(optional)*                | `"047"`                            |

A compact single-string representation combining degree, quality, and
inversion figure. The `key` column supports forward-fill: the first row
must have a non-empty key; subsequent rows may be `""` to mean "same key".
Constants: `RN_FORMAT_REQUIRED_COLUMNS`.

## RN String Dialects

The `rn` column string can follow one of two conventions:

| Dialect     | Mode encoding                                    | Examples                               |
|-------------|--------------------------------------------------|----------------------------------------|
| **music21** | case = mode (upper=major, lower=minor)           | `"V"`, `"iv6"`, `"viio7"`, `"V/V"`     |
| **rnbert**  | explicit `M`/`m` suffix, always uppercase degree | `"VM"`, `"IVm6"`, `"VIIo7"`, `"VM/VM"` |

Translation: `translate_rns(rn, src="rnbert", dst="music21")` in
`harmony/chords.py`. Currently **one-way only** (rnbert → music21).

## secondary_mode

The `secondary_mode` column (split format, optional) disambiguates the mode
of the tonicized key:

| Value | Meaning                                             |
|-------|-----------------------------------------------------|
| `"_"` | Use the default mode from the `TONICIZATIONS` table |
| `"m"` | Force minor secondary key                           |
| `"M"` | Force major secondary key                           |

In joined format, secondary mode appears as a suffix on the secondary
degree: `"V/Vm"` (minor), `"V/VM"` (major). When the secondary degree is
`"I"` (no tonicization) and mode is `"_"`, the `/I` suffix is stripped;
when mode is `"m"` or `"M"`, the `/Im` or `/IM` suffix is preserved.

## Conversion Paths

```
joined  <──>  split  ───>  rn
                            │
                     rnbert ──> music21
```

| Conversion | Function | Notes |
|---|---|---|
| joined → split | `single_degree_to_split_degrees()` | Lossless |
| split → joined | `split_degrees_to_single_degree()` | Lossless |
| split → rn | `split_degrees_to_single_degree(quality_col=..., inversion_col=...)` | Requires pre-converting inversion int to figure via `inversion_number_to_figure()` |
| rnbert → music21 | `translate_rns()` | One-way; lossy for augmented sixths |

## Future Work

| Conversion | Status | Notes |
|---|---|---|
| music21 → rnbert | **Missing** | `translate_rns()` raises `NotImplementedError` for reverse |
| rn → split | **Missing** | No parser to decompose an RN string back into split columns |
| rn → joined | **Missing** | Blocked by missing rn → split |
| Augmented sixth round-trip | **Lossy** | Multiple rnbert inputs (`xaug63`, `xaug642`, etc.) collapse to `"It6"` |
| `merge_annotations` quality | **Lossy by design** | Strips `"7"` from quality for display; inversion figure carries 7th-chord info instead |
