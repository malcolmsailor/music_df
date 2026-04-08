"""Transform a music_df to an "absolute time" representation at 120 bpm.

The input onset/release values are interpreted in quarter-note beats with the
timing defined by the df's tempo events. The output has the same events but
with onset/release rewritten so that beats correspond directly to wall-clock
time at a canonical 120 bpm. Tempo events come in two flavors — ``"tempo"``
rows (BPM in ``other["tempo"]``) and ``"set_tempo"`` rows (microseconds per
beat in ``other["tempo"]``, as emitted by MIDI); both are handled via
:func:`music_df.add_feature.make_tempos_explicit`.
"""

from ast import literal_eval

import numpy as np
import pandas as pd

from music_df.transforms import transform

TARGET_BPM = 120.0
TEMPO_TYPES = ("tempo", "set_tempo")


def _extract_bpm(type_val: str, other: object) -> float:
    # "tempo" rows store BPM directly in other["tempo"]; MIDI "set_tempo" rows
    # store microseconds-per-quarter in the same slot. Mirrors the dispatch
    # logic in music_df.add_feature.make_tempos_explicit.
    if isinstance(other, str):
        other = literal_eval(other)
    assert isinstance(other, dict), f"tempo row missing `other` dict: {other!r}"
    raw = other["tempo"]
    if type_val == "set_tempo":
        return 60_000_000.0 / float(raw)
    return float(raw)


def _make_synthetic_tempo_df(template: pd.DataFrame) -> pd.DataFrame:
    # Build a single-row DataFrame whose per-column dtypes match `template`.
    # Returning a properly-typed frame (rather than a dict of Nones) sidesteps
    # the pandas 2.x FutureWarning about all-NA columns being excluded from
    # dtype inference in pd.concat.
    row: dict[str, object] = {}
    for col in template.columns:
        if col == "type":
            row[col] = "tempo"
        elif col == "onset":
            row[col] = 0.0
        elif col == "other":
            row[col] = {"tempo": TARGET_BPM}
        else:
            # NaN for numeric columns, None for object columns.
            kind = template[col].dtype.kind
            row[col] = float("nan") if kind in "fiub" else None
    synthetic = pd.DataFrame([row])
    # Cast to template's dtypes where possible so concat doesn't upcast.
    for col in template.columns:
        try:
            synthetic[col] = synthetic[col].astype(template[col].dtype)
        except (TypeError, ValueError):
            pass
    return synthetic


@transform
def to_absolute_time(df: pd.DataFrame) -> pd.DataFrame:
    """Rewrite onsets/releases so beats measure wall-clock time at 120 bpm.

    At the original tempo of `b` bpm, one quarter note lasts ``60/b`` seconds.
    At 120 bpm, one quarter note lasts 0.5 s. So one original beat maps to
    ``(60/b) / 0.5 = 120/b`` target beats — the slope per tempo segment.

    Tempo changes are treated as step changes (piecewise-constant between
    tempo events). All original tempo rows are dropped and a single synthetic
    ``tempo=120`` row is prepended, so the output always has exactly one tempo
    row at onset 0.

    If the df has no tempo rows (or no ``type`` column), onsets/releases are
    left alone and the synthetic row is prepended.

    A simple 90 → 120 case. Quarter notes at 90 bpm become length-4/3 at 120:

    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["tempo", "note", "note", "note"],
    ...         "pitch": [float("nan"), 60, 62, 64],
    ...         "onset": [0.0, 0.0, 1.0, 2.0],
    ...         "release": [float("nan"), 1.0, 2.0, 3.0],
    ...         "other": [{"tempo": 90.0}, None, None, None],
    ...     }
    ... )
    >>> to_absolute_time(df)[["type", "pitch", "onset", "release"]]
        type  pitch     onset   release
    0  tempo    NaN  0.000000       NaN
    1   note   60.0  0.000000  1.333333
    2   note   62.0  1.333333  2.666667
    3   note   64.0  2.666667  4.000000

    A tempo change partway through. Bar rows are remapped in lockstep:

    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["tempo", "bar", "note", "tempo", "note", "bar"],
    ...         "pitch": [
    ...             float("nan"), float("nan"), 60,
    ...             float("nan"), 62, float("nan"),
    ...         ],
    ...         "onset": [0.0, 0.0, 0.0, 2.0, 2.0, 4.0],
    ...         "release": [
    ...             float("nan"), 4.0, 2.0,
    ...             float("nan"), 4.0, 8.0,
    ...         ],
    ...         "other": [
    ...             {"tempo": 120.0}, None, None,
    ...             {"tempo": 60.0}, None, None,
    ...         ],
    ...     }
    ... )
    >>> to_absolute_time(df)[["type", "pitch", "onset", "release"]]
        type  pitch  onset  release
    0  tempo    NaN    0.0      NaN
    1    bar    NaN    0.0      6.0
    2   note   60.0    0.0      2.0
    3   note   62.0    2.0      6.0
    4    bar    NaN    6.0     14.0

    A df with no tempo rows is returned with a synthetic tempo row prepended
    and onsets unchanged:

    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["note", "note"],
    ...         "pitch": [60.0, 62.0],
    ...         "onset": [0.0, 1.0],
    ...         "release": [1.0, 2.0],
    ...         "other": [None, None],
    ...     }
    ... )
    >>> to_absolute_time(df)[["type", "pitch", "onset", "release"]]
        type  pitch  onset  release
    0  tempo    NaN    0.0      NaN
    1   note   60.0    0.0      1.0
    2   note   62.0    1.0      2.0
    """
    has_tempo = "type" in df.columns and bool(df["type"].isin(TEMPO_TYPES).any())
    if not has_tempo:
        synthetic = _make_synthetic_tempo_df(df)
        out = pd.concat([synthetic, df], ignore_index=True)
        out.attrs = df.attrs.copy()
        out.attrs["absolute_time"] = True
        return out

    # Extract (onset, bpm) for each tempo row. Both "tempo" (BPM in
    # other["tempo"]) and MIDI "set_tempo" (microseconds-per-quarter in
    # other["tempo"]) are handled by _extract_bpm.
    tempo_mask = df["type"].isin(TEMPO_TYPES)
    tempo_onsets_list: list[float] = []
    tempo_bpms_list: list[float] = []
    for _, row in df.loc[tempo_mask].iterrows():
        tempo_onsets_list.append(float(row["onset"]))
        tempo_bpms_list.append(_extract_bpm(str(row["type"]), row.get("other")))
    seg_onsets = np.array(tempo_onsets_list, dtype=float)
    seg_bpms = np.array(tempo_bpms_list, dtype=float)

    # Ensure there's a segment starting at (or before) the smallest onset in
    # the frame. If the first tempo event is at onset > 0, prepend a 120 bpm
    # segment so the opening region is a no-op.
    min_onset = float(df["onset"].min())
    if len(seg_onsets) == 0 or seg_onsets[0] > min_onset:
        seg_onsets = np.concatenate(([min_onset], seg_onsets))
        seg_bpms = np.concatenate(([TARGET_BPM], seg_bpms))

    # Collapse duplicate onsets (e.g., two tempo events at the same beat) —
    # keep the last one, since the later event overrides.
    order = np.argsort(seg_onsets, kind="stable")
    seg_onsets = seg_onsets[order]
    seg_bpms = seg_bpms[order]
    unique_mask = np.concatenate(([True], np.diff(seg_onsets) > 0))
    # For duplicate-onset runs, keep the *last* bpm. np.unique with return_index
    # keeps the first — so reverse, unique, reverse back.
    if not unique_mask.all():
        keep_idx = []
        i = 0
        n = len(seg_onsets)
        while i < n:
            j = i
            while j + 1 < n and seg_onsets[j + 1] == seg_onsets[i]:
                j += 1
            keep_idx.append(j)
            i = j + 1
        seg_onsets = seg_onsets[keep_idx]
        seg_bpms = seg_bpms[keep_idx]

    slopes = TARGET_BPM / seg_bpms

    # Cumulative new-domain start for each segment:
    #   new_start[0] = seg_onsets[0]  (identity at the origin)
    #   new_start[i] = new_start[i-1] + slopes[i-1] * (seg_onsets[i] - seg_onsets[i-1])
    seg_lengths = np.diff(seg_onsets)
    new_starts = np.concatenate(
        ([seg_onsets[0]], seg_onsets[0] + np.cumsum(slopes[:-1] * seg_lengths))
    )

    def remap(values: np.ndarray) -> np.ndarray:
        out = np.full_like(values, np.nan, dtype=float)
        mask = ~np.isnan(values)
        v = values[mask]
        # searchsorted(..., side="right") - 1 gives the segment index whose
        # onset is <= v. Clip to 0 for safety (values < seg_onsets[0] stay in
        # segment 0, matching the 120-bpm prepend).
        idx = np.searchsorted(seg_onsets, v, side="right") - 1
        idx = np.clip(idx, 0, len(seg_onsets) - 1)
        out[mask] = new_starts[idx] + slopes[idx] * (v - seg_onsets[idx])
        return out

    out = df.copy()
    new_onset = remap(out["onset"].to_numpy(dtype=float))
    out["onset"] = new_onset
    if "release" in out.columns:
        out["release"] = remap(out["release"].to_numpy(dtype=float))
    if "duration" in out.columns:
        # Recompute from the new onset/release where both are present;
        # preserve NaN otherwise.
        if "release" in out.columns:
            out["duration"] = out["release"] - out["onset"]

    # Drop all existing tempo rows and prepend the synthetic one.
    out = pd.DataFrame(out[~out["type"].isin(TEMPO_TYPES)]).reset_index(drop=True)
    synthetic = _make_synthetic_tempo_df(out)
    result = pd.concat([synthetic, out], ignore_index=True)

    result.attrs = df.attrs.copy()
    result.attrs["absolute_time"] = True
    return result
