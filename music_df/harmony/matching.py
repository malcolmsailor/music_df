import io  # noqa: F401
import math
from typing import Sequence

import numpy as np
import pandas as pd

from music_df.harmony.chords import hex_str_to_pc_ints
from music_df.harmony.key_profiles import KEY_PROFILES
from music_df.keys import CHROMATIC_SCALE, MAJOR_KEYS, MINOR_KEYS
from music_df.slice_df import slice_df


def percent_pc_match(
    passage: pd.DataFrame,
    pitch_classes: set[int] | str,
    weight_by_duration: bool = True,
    input_contains_only_notes: bool = False,
) -> float:
    """
    Return the percentage of pitch classes in the passage that match the given pitch
    classes.

    Args:
        passage: A DataFrame with a "type" column and a "pitch" column.
        pitch_classes: A set of pitch classes or a hex string.
        weight_by_duration: If True (default), weight the match by the duration of the
        notes.
        input_contains_only_notes: If True, the input DataFrame is assumed to contain
        only notes, which saves filtering the DataFrame by the type column.

    Returns:
        The percentage of pitch classes in the passage that match the given pitch
        classes.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,2.0
    ... note,64,2.0,3.0
    ... note,67,3.0,4.0
    ... '''
    ...     )
    ... )
    >>> percent_pc_match(df, "047")
    1.0
    >>> percent_pc_match(df, {0, 4, 7})
    1.0
    >>> percent_pc_match(df, "049")
    0.75
    >>> percent_pc_match(df, "049", weight_by_duration=False)  # doctest: +ELLIPSIS
    0.666...
    """
    if isinstance(pitch_classes, str):
        pitch_classes = set(hex_str_to_pc_ints(pitch_classes, return_set=True))

    if not input_contains_only_notes:
        notes = passage.loc[passage["type"] == "note"]
    else:
        notes = passage

    if len(notes) == 0:
        return float("nan")

    matches = (notes["pitch"] % 12).isin(pitch_classes)

    if not weight_by_duration:
        return matches.mean()
    if "duration" in passage.columns:
        durations = notes["duration"]
    else:
        durations = notes["release"] - notes["onset"]

    total_duration = durations.sum()
    if total_duration == 0:
        return float("nan")
    return (matches * durations).sum() / total_duration


def label_pc_matches(
    music_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    chord_df_pc_key: str = "chord_pcs",
    is_sliced: bool = False,
    match_col: str = "is_chord_match",
) -> pd.DataFrame:
    if not is_sliced:
        music_df = slice_df(music_df, chord_df["onset"])

    music_df.loc[:, match_col] = False

    for _, chord_row in chord_df.iterrows():
        chord_notes = music_df.loc[
            (music_df["onset"] >= chord_row["onset"])
            & (music_df["release"] <= chord_row["release"])
            & (music_df["type"] == "note")
        ]
        chord_pcs = chord_row[chord_df_pc_key]
        if isinstance(chord_pcs, str):
            chord_pcs = hex_str_to_pc_ints(chord_pcs, return_set=True)
        matches = (chord_notes["pitch"] % 12).isin(chord_pcs)
        music_df.loc[chord_notes.index, match_col] = matches

    return music_df


def percent_chord_df_match(
    music_df: pd.DataFrame,
    chord_df: pd.DataFrame,
    weight_by_duration: bool = True,
    chord_df_pc_key: str = "chord_pcs",
    is_sliced: bool = False,
    match_col: str = "percent_chord_match",
):
    if is_sliced:
        sliced_notes = music_df.loc[music_df["type"] == "note"]
    else:
        sliced_notes = slice_df(music_df[music_df["type"] == "note"], chord_df["onset"])

    if chord_df.empty:
        return {
            "macroaverage": float("nan"),
            "microaverage": float("nan"),
            "music_df": music_df,
        }

    chord_pc_matches = []

    music_df.loc[:, match_col] = float("nan")
    music_df.loc[:, chord_df_pc_key] = ""

    for _, chord_row in chord_df.iterrows():
        chord_notes = sliced_notes[
            (sliced_notes["onset"] >= chord_row["onset"])
            & (sliced_notes["release"] <= chord_row["release"])
        ]
        chord_pc_match = percent_pc_match(
            chord_notes,
            chord_row[chord_df_pc_key],
            weight_by_duration=weight_by_duration,
            input_contains_only_notes=True,
        )
        chord_pc_matches.append(chord_pc_match)
        music_df.loc[chord_notes.index, match_col] = chord_pc_match
        music_df.loc[chord_notes.index, chord_df_pc_key] = chord_row[chord_df_pc_key]

    valid_matches = [m for m in chord_pc_matches if not math.isnan(m)]
    if len(valid_matches) == 0:
        macroaverage = float("nan")
    else:
        macroaverage = sum(valid_matches) / len(valid_matches)

    return {
        "macroaverage": macroaverage,
        "microaverage": music_df[match_col].mean(skipna=True),
        "music_df": music_df,
    }


def _correlate_pc_counts_with_profile(
    pc_counts: Sequence[float], profile: Sequence[float]
) -> float:
    """
    Compute Pearson correlation between a pitch class distribution and a key profile.

    Args:
        pc_counts: A 12-element sequence of pitch class counts/weights.
        profile: A 12-element key profile where index 0 = tonic.

    Returns:
        Pearson correlation coefficient, or NaN if variance is zero.

    >>> _correlate_pc_counts_with_profile([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    1.0
    >>> _correlate_pc_counts_with_profile([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    nan
    """
    assert len(pc_counts) == 12
    assert len(profile) == 12

    pc_array = np.array(pc_counts, dtype=float)
    profile_array = np.array(profile, dtype=float)

    if np.std(pc_array) == 0 or np.std(profile_array) == 0:
        return float("nan")

    return float(np.corrcoef(pc_array, profile_array)[0, 1])


def _get_pc_distribution(
    passage: pd.DataFrame,
    weight_by_duration: bool = True,
    input_contains_only_notes: bool = False,
) -> list[float]:
    """
    Compute pitch class distribution from a passage.

    Returns a 12-element list where index i is the total duration (or count)
    of pitch class i.
    """
    if not input_contains_only_notes:
        notes = passage.loc[passage["type"] == "note"]
    else:
        notes = passage

    if len(notes) == 0:
        return [0.0] * 12

    pcs = notes["pitch"] % 12

    if weight_by_duration:
        if "duration" in passage.columns:
            durations = notes["duration"]
        else:
            durations = notes["release"] - notes["onset"]
        weights = durations
    else:
        weights = pd.Series([1.0] * len(notes), index=notes.index)

    pc_counts = [0.0] * 12
    for pc, weight in zip(pcs, weights):
        pc_counts[int(pc)] += weight

    return pc_counts


def key_profile_correlation(
    passage: pd.DataFrame,
    profile: str | Sequence[float],
    key: str = "C",
    weight_by_duration: bool = True,
    input_contains_only_notes: bool = False,
) -> float:
    """
    Compute correlation between a passage's pitch class distribution and a key profile.

    Args:
        passage: A music_df DataFrame.
        profile: Either a preset name ("krumhansl_major", "krumhansl_minor",
            "aarden_major", "aarden_minor") or a 12-element sequence where
            position 0 = tonic.
        key: The key to test (e.g., "C", "G", "f#"). Determines which pitch
            class is treated as the tonic.
        weight_by_duration: If True (default), weight by note duration.
        input_contains_only_notes: If True, skip filtering by type column.

    Returns:
        Pearson correlation coefficient, or NaN if insufficient data.

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... note,60,0.0,1.0
    ... note,62,1.0,2.0
    ... note,64,2.0,3.0
    ... note,65,3.0,4.0
    ... note,67,4.0,5.0
    ... note,69,5.0,6.0
    ... note,71,6.0,7.0
    ... '''
    ...     )
    ... )
    >>> # C major scale should correlate positively with Krumhansl major profile at key="C"
    >>> corr_c_major = key_profile_correlation(df, "krumhansl_major", key="C")
    >>> corr_c_major > 0.7
    True
    >>> # Same scale should correlate poorly with minor profile at key="C"
    >>> corr_c_minor = key_profile_correlation(df, "krumhansl_minor", key="C")
    >>> corr_c_major > corr_c_minor
    True
    >>> # Can also use a custom profile
    >>> custom = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    >>> key_profile_correlation(df, custom, key="C") > 0.9
    True
    """
    if isinstance(profile, str):
        profile_values = list(KEY_PROFILES[profile])
    else:
        profile_values = list(profile)

    assert len(profile_values) == 12

    pc_distribution = _get_pc_distribution(
        passage,
        weight_by_duration=weight_by_duration,
        input_contains_only_notes=input_contains_only_notes,
    )

    if sum(pc_distribution) == 0:
        return float("nan")

    key_pc = CHROMATIC_SCALE[key.capitalize()]

    # Rotate pc_distribution left by key_pc so that the key's tonic is at index 0
    rotated_distribution = pc_distribution[key_pc:] + pc_distribution[:key_pc]

    return _correlate_pc_counts_with_profile(rotated_distribution, profile_values)


def all_key_correlations(
    passage: pd.DataFrame,
    profile_type: str = "krumhansl",
    weight_by_duration: bool = True,
    input_contains_only_notes: bool = False,
) -> dict[str, float]:
    """
    Compute correlations for all 24 major/minor keys.

    Args:
        passage: A music_df DataFrame.
        profile_type: "krumhansl" or "aarden". Uses the major profile for major
            keys and minor profile for minor keys.
        weight_by_duration: If True (default), weight by note duration.
        input_contains_only_notes: If True, skip filtering by type column.

    Returns:
        Dict mapping key names to correlation values. Major keys use uppercase
        (e.g., "C", "Db"), minor keys use lowercase (e.g., "c", "c#").

    >>> df = pd.read_csv(
    ...     io.StringIO(
    ...         '''
    ... type,pitch,onset,release
    ... note,60,0.0,1.0
    ... note,62,1.0,2.0
    ... note,64,2.0,3.0
    ... note,65,3.0,4.0
    ... note,67,4.0,5.0
    ... note,69,5.0,6.0
    ... note,71,6.0,7.0
    ... '''
    ...     )
    ... )
    >>> correlations = all_key_correlations(df, profile_type="krumhansl")
    >>> len(correlations)
    24
    >>> # C major scale should have highest correlation with C major
    >>> max_key = max(correlations, key=correlations.get)
    >>> max_key
    'C'
    >>> # Check that all major and minor keys are present
    >>> all(k in correlations for k in ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"))
    True
    >>> all(k in correlations for k in ("c", "c#", "d", "eb", "e", "f", "f#", "g", "g#", "a", "bb", "b"))
    True
    """
    major_profile_name = f"{profile_type}_major"
    minor_profile_name = f"{profile_type}_minor"

    results: dict[str, float] = {}

    for key in MAJOR_KEYS:
        results[key] = key_profile_correlation(
            passage,
            major_profile_name,
            key=key,
            weight_by_duration=weight_by_duration,
            input_contains_only_notes=input_contains_only_notes,
        )

    for key in MINOR_KEYS:
        results[key] = key_profile_correlation(
            passage,
            minor_profile_name,
            key=key,
            weight_by_duration=weight_by_duration,
            input_contains_only_notes=input_contains_only_notes,
        )

    return results
