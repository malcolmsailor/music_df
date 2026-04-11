"""Transform a music_df to an "absolute time" representation at 120 bpm.

The input onset/release values are interpreted in quarter-note beats with the
timing defined by the df's tempo events. The output has the same events but
with onset/release rewritten so that beats correspond directly to wall-clock
time at a canonical 120 bpm. Tempo events come in two flavors — ``"tempo"``
rows (BPM in ``other["tempo"]``) and ``"set_tempo"`` rows (microseconds per
beat in ``other["tempo"]``, as emitted by MIDI); both are handled via
:func:`music_df.add_feature.make_tempos_explicit`.

The forward transform :func:`to_absolute_time` is invertible via
:func:`from_absolute_time` provided that ``df.attrs`` is preserved between
the two calls. The forward call stashes the dropped tempo and time-signature
rows under ``df.attrs["to_absolute_time_metadata"]``, and the inverse reads
them back.
"""

from ast import literal_eval

import numpy as np
import pandas as pd

from music_df.transforms import transform

TARGET_BPM = 120.0
TEMPO_TYPES = ("tempo", "set_tempo")
_DROPPED_TYPES = TEMPO_TYPES + ("time_signature",)
_METADATA_ATTR = "to_absolute_time_metadata"


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


def _records_to_frame(
    records: list[dict], template: pd.DataFrame
) -> pd.DataFrame:
    # Build a DataFrame from a list of stashed row dicts, reindexed to the
    # template's columns so concat doesn't introduce NaN gaps. Missing columns
    # are filled NaN/None depending on dtype, mirroring _make_synthetic_tempo_df.
    if not records:
        return pd.DataFrame(columns=template.columns)
    frame = pd.DataFrame.from_records(records)
    for col in template.columns:
        if col not in frame.columns:
            kind = template[col].dtype.kind
            frame[col] = float("nan") if kind in "fiub" else None
    frame = pd.DataFrame(frame[list(template.columns)])
    for col in template.columns:
        try:
            frame[col] = frame[col].astype(template[col].dtype)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            pass
    return frame


def _build_tempo_segments(
    tempo_records: list[dict], min_onset: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the piecewise-linear tempo segment map shared by forward+inverse.

    Returns ``(seg_onsets, seg_bpms, seg_new_starts, slopes)`` where:

    - ``seg_onsets`` are the segment-start onsets in the *original* beat
      domain.
    - ``seg_bpms`` are the bpms in effect over each segment.
    - ``seg_new_starts`` are the segment-start onsets in the *absolute-time*
      (120-bpm) domain.
    - ``slopes`` is ``TARGET_BPM / seg_bpms`` — the per-segment forward slope.

    The forward map is ``new = seg_new_starts[i] + slopes[i] * (orig - seg_onsets[i])``;
    the inverse is the obvious linear inversion.

    A synthetic ``(min_onset, TARGET_BPM)`` segment is prepended whenever the
    first tempo onset is strictly greater than ``min_onset``, so the opening
    region is a no-op slope-1 prefix. Duplicate-onset tempo events collapse
    to the *last* event at that onset (later overrides earlier). The inverse
    must call this with the same `tempo_records` row order as the forward did
    so the collapse picks the same survivors.
    """
    tempo_onsets_list: list[float] = []
    tempo_bpms_list: list[float] = []
    for row in tempo_records:
        tempo_onsets_list.append(float(row["onset"]))
        tempo_bpms_list.append(_extract_bpm(str(row["type"]), row.get("other")))
    seg_onsets = np.array(tempo_onsets_list, dtype=float)
    seg_bpms = np.array(tempo_bpms_list, dtype=float)

    if len(seg_onsets) == 0 or seg_onsets[0] > min_onset:
        seg_onsets = np.concatenate(([min_onset], seg_onsets))
        seg_bpms = np.concatenate(([TARGET_BPM], seg_bpms))

    order = np.argsort(seg_onsets, kind="stable")
    seg_onsets = seg_onsets[order]
    seg_bpms = seg_bpms[order]

    # Collapse duplicate-onset runs, keeping the *last* bpm in each run.
    n = len(seg_onsets)
    if n > 1 and not (np.diff(seg_onsets) > 0).all():
        keep_idx: list[int] = []
        i = 0
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
    seg_new_starts = np.concatenate(
        ([seg_onsets[0]], seg_onsets[0] + np.cumsum(slopes[:-1] * seg_lengths))
    )

    return seg_onsets, seg_bpms, seg_new_starts, slopes


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

    Time-signature rows are also dropped, because their numerator/denominator
    no longer matches the bar spacing in the new domain (e.g., 3/4 at 90 bpm
    has bars 3 beats apart originally, 4 beats apart after the transform).

    Both dropped row types are stashed under
    ``df.attrs["to_absolute_time_metadata"]`` so :func:`from_absolute_time`
    can rebuild the original df.

    Calling ``to_absolute_time`` twice is a no-op: if the input already has
    ``attrs["absolute_time"]`` set, the df is returned unchanged.

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
    if df.attrs.get("absolute_time"):
        return df

    # Stash dropped rows in the *original* beat domain BEFORE any modification.
    # `to_dict("records")` preserves row order, which the forward collapse and
    # inverse rebuild both rely on (later-onset duplicates override earlier).
    if "type" in df.columns:
        tempo_records = df.loc[df["type"].isin(TEMPO_TYPES)].to_dict("records")
        ts_records = df.loc[df["type"] == "time_signature"].to_dict("records")
    else:
        tempo_records = []
        ts_records = []

    min_onset = float(df["onset"].min()) if len(df) else 0.0
    metadata = {
        "tempos": tempo_records,
        "time_signatures": ts_records,
        "min_onset": min_onset,
    }

    has_tempo = bool(tempo_records)
    if not has_tempo:
        # Still need to drop any time_signature rows, since their numerator /
        # denominator no longer matches the absolute-time bar spacing.
        if ts_records:
            df_no_ts = pd.DataFrame(
                df[df["type"] != "time_signature"]
            ).reset_index(drop=True)
        else:
            df_no_ts = df
        synthetic = _make_synthetic_tempo_df(df_no_ts)
        out = pd.concat([synthetic, df_no_ts], ignore_index=True)
        out.attrs = df.attrs.copy()
        out.attrs["absolute_time"] = True
        out.attrs[_METADATA_ATTR] = metadata
        return out

    seg_onsets, _, seg_new_starts, slopes = _build_tempo_segments(
        tempo_records, min_onset
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
        out[mask] = seg_new_starts[idx] + slopes[idx] * (v - seg_onsets[idx])
        return out

    out = df.copy()
    out["onset"] = remap(out["onset"].to_numpy(dtype=float))
    if "release" in out.columns:
        out["release"] = remap(out["release"].to_numpy(dtype=float))
    if "duration" in out.columns and "release" in out.columns:
        out["duration"] = out["release"] - out["onset"]

    out = pd.DataFrame(out[~out["type"].isin(_DROPPED_TYPES)]).reset_index(drop=True)
    synthetic = _make_synthetic_tempo_df(out)
    result = pd.concat([synthetic, out], ignore_index=True)

    result.attrs = df.attrs.copy()
    result.attrs["absolute_time"] = True
    result.attrs[_METADATA_ATTR] = metadata
    return result


@transform
def from_absolute_time(df: pd.DataFrame) -> pd.DataFrame:
    """Inverse of :func:`to_absolute_time`.

    Reads the original tempo/time-signature rows and ``min_onset`` back from
    ``df.attrs["to_absolute_time_metadata"]``, rebuilds the same segment map
    that the forward transform built, and inverse-remaps onsets/releases. The
    synthetic ``tempo=120`` row prepended by the forward is dropped, the
    stashed rows are re-inserted, and the absolute-time attrs are cleared.

    Raises ``ValueError`` if the input is not a forward-transformed df. The
    forward call's metadata must survive any intervening pandas operations:
    ``pd.concat``, ``reset_index``, and most other pandas calls drop
    ``df.attrs`` unless reassigned, so in practice ``from_absolute_time``
    should be called on the direct output of ``to_absolute_time``.

    Round-tripping the simple 90 → 120 example:

    >>> df = pd.DataFrame(
    ...     {
    ...         "type": ["tempo", "note", "note", "note"],
    ...         "pitch": [float("nan"), 60, 62, 64],
    ...         "onset": [0.0, 0.0, 1.0, 2.0],
    ...         "release": [float("nan"), 1.0, 2.0, 3.0],
    ...         "other": [{"tempo": 90.0}, None, None, None],
    ...     }
    ... )
    >>> from_absolute_time(to_absolute_time(df))[["type", "pitch", "onset", "release"]]
        type  pitch  onset  release
    0  tempo    NaN    0.0      NaN
    1   note   60.0    0.0      1.0
    2   note   62.0    1.0      2.0
    3   note   64.0    2.0      3.0
    """
    if not df.attrs.get("absolute_time") or _METADATA_ATTR not in df.attrs:
        raise ValueError(
            "from_absolute_time requires df.attrs['absolute_time'] is True and "
            f"df.attrs['{_METADATA_ATTR}'] is present. Did you call "
            "to_absolute_time first, or did an intervening transform drop attrs?"
        )

    metadata = df.attrs[_METADATA_ATTR]
    tempo_records: list[dict] = metadata["tempos"]
    ts_records: list[dict] = metadata["time_signatures"]
    min_onset: float = metadata["min_onset"]

    out = df.copy()
    # Drop the synthetic tempo=120 row prepended by the forward.
    out = pd.DataFrame(out[out["type"] != "tempo"]).reset_index(drop=True)

    if tempo_records:
        seg_onsets, _, seg_new_starts, slopes = _build_tempo_segments(
            tempo_records, min_onset
        )
        inv_slopes = 1.0 / slopes  # = seg_bpms / TARGET_BPM

        def inverse_remap(values: np.ndarray) -> np.ndarray:
            result = np.full_like(values, np.nan, dtype=float)
            mask = ~np.isnan(values)
            v = values[mask]
            # Locate by NEW-domain onset, since the input is in absolute-time.
            idx = np.searchsorted(seg_new_starts, v, side="right") - 1
            idx = np.clip(idx, 0, len(seg_new_starts) - 1)
            result[mask] = seg_onsets[idx] + inv_slopes[idx] * (
                v - seg_new_starts[idx]
            )
            return result

        out["onset"] = inverse_remap(out["onset"].to_numpy(dtype=float))
        if "release" in out.columns:
            out["release"] = inverse_remap(out["release"].to_numpy(dtype=float))
        if "duration" in out.columns and "release" in out.columns:
            out["duration"] = out["release"] - out["onset"]

    restored_rows = _records_to_frame(tempo_records + ts_records, out)
    if len(restored_rows):
        result = pd.concat([restored_rows, out], ignore_index=True)
        result = result.sort_values("onset", kind="stable").reset_index(drop=True)
    else:
        result = out

    result.attrs = {
        k: v
        for k, v in df.attrs.items()
        if k not in ("absolute_time", _METADATA_ATTR)
    }
    return result
