import io  # noqa: F401

import pandas as pd

from music_df.harmony.chords import hex_str_to_pc_ints
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

    matches = (notes["pitch"] % 12).isin(pitch_classes)

    if not weight_by_duration:
        return matches.mean()
    if "duration" in passage.columns:
        durations = notes["duration"]
    else:
        durations = notes["release"] - notes["onset"]

    return (matches * durations).sum() / durations.sum()


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

    chord_pc_matches = []

    music_df.loc[:, match_col] = float("nan")

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

    return {
        "macroaverage": sum(chord_pc_matches) / len(chord_pc_matches),
        "microaverage": music_df[match_col].mean(skipna=True),
        "music_df": music_df,
    }
