import io  # used by doctests # noqa: F401
import math
import random
from ast import literal_eval
from copy import deepcopy
from typing import Iterable, Iterator

import pandas as pd

from music_df import chromatic_transpose
from music_df.constants import MAX_PIANO_PITCH, MIN_PIANO_PITCH
from music_df.salami_slice import appears_salami_sliced
from music_df.sort_df import sort_df
from music_df.transpose import transpose_to_key
from music_df.utils.rests import return_rests

STANDARD_KEYS = list(range(-6, 7))
ENHARMONICALLY_UNIQUE_KEYS = list(range(-6, 6))


def _n_random_keys(n: int, enh_unique_keys=True) -> list[int]:
    if enh_unique_keys:
        keys = ENHARMONICALLY_UNIQUE_KEYS.copy()
        if n == 12:
            return keys
    else:
        keys = STANDARD_KEYS.copy()
        if n == 13:
            return keys
    return random.sample(keys, k=n)


def aug_by_trans(
    orig_data: pd.DataFrame | Iterable[pd.DataFrame],
    n_keys: int,
    hi: int | None = MAX_PIANO_PITCH,
    low: int | None = MIN_PIANO_PITCH,
) -> Iterator[pd.DataFrame]:
    """
    Augment a music_df by transposing its content to random keys.

    Calls music_df.transpose.transpose_to_key to perform the transpositions. See the
    docstring of that function for more details.

    Args:
        orig_data: a music_df or an iterable of music_dfs. All input dataframes must
            have a "global_key_sig" int attribute in df.attrs, which is used to
            determine the transpositions.
        n_keys: the number of keys to transpose to
        hi: the maximum pitch to allow
        low: the minimum pitch to allow

    Returns:
        An iterator of music_dfs with the pitches transposed to random keys. The
        original key may or may not be included among the yielded keys.

    """
    if isinstance(orig_data, pd.DataFrame):
        orig_data = [orig_data]

    for df in orig_data:
        # we don't guarantee that original key will be included among the keys
        keys = _n_random_keys(n_keys, enh_unique_keys=True)

        for key in keys:
            out = transpose_to_key(df, key, inplace=False)
            if hi is not None:
                max_pitch = df.pitch.max()
                if max_pitch > hi:
                    continue
            if low is not None:
                min_pitch = df.pitch.min()
                if min_pitch < low:
                    continue

            yield out


def aug_within_range(
    df_iter: Iterable[pd.DataFrame],
    n_keys: int,
    hi: int = MAX_PIANO_PITCH,
    low: int = MIN_PIANO_PITCH,
    min_trans: int = -5,
    max_trans: int = 6,
):
    """
    Augment a music_df by transposing its content to random keys within a range.

    Calls music_df.transpose.chromatic_transpose to perform the transpositions. See the
    docstring of that function for more details.

    Args:
        df_iter: an iterable of music_dfs
        n_keys: the number of keys to transpose to. If None, all keys within the range
            will be used.
        hi: the maximum pitch to allow
        low: the minimum pitch to allow
        min_trans: the minimum transposition to allow
        max_trans: the maximum transposition to allow

    Returns:
        An iterator of music_dfs with the pitches transposed to random keys within a range.
    """
    # if n_keys is None, we transpose to every step within range
    avail_range = hi - low
    for df in df_iter:
        if "spelling" in df.columns:
            raise ValueError("need to use 'tranpose_to_key' with spelled data")
        actual_max = int(df.pitch.max())
        actual_min = int(df.pitch.min())
        actual_range = actual_max - actual_min
        n_trans = avail_range - actual_range + 1
        if n_trans <= 0:
            continue
        trans = list(
            range(
                max(low - actual_min, min_trans),
                min(max_trans, low - actual_min + n_trans) + 1,
            )
        )
        if n_keys < n_trans:
            random.shuffle(trans)
            trans = trans[:n_keys]

        for t in trans:
            yield chromatic_transpose(df, t, inplace=False, label=True)


def _to_dict_if_necessary(d):
    if isinstance(d, str):
        return literal_eval(d)
    return d


def scale_time_sigs(music_df: pd.DataFrame, factor: int | float) -> pd.DataFrame:
    """
    Scale a time signature up or down by a power of 2.

    Typically, this simply means multiplying or dividing the denominator. For example,
    if the time signature is 4/4, then
        - scaling by 2 gives 4/2
        - scaling by 0.5 gives 4/8

    However, if the denominator is too small, we may scale up the numerator instead. For
    example, scaling 4/1 by 2.0 gives 8/1. Note that this can change the metric
    implications of the time signature, since 3/1 implies three whole note beats per
    bar, while 6/1 implies two dotted breve beats per bar.

    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [float("nan"), float("nan")],
    ...         "onset": [0, 16],
    ...         "type": ["time_signature", "time_signature"],
    ...         "other": [
    ...             {"numerator": 4, "denominator": 1},
    ...             {"numerator": 3, "denominator": 16},
    ...         ],
    ...     }
    ... )
    >>> pd.set_option(
    ...     "display.width", 200
    ... )  # To avoid issues when the terminal is a different size
    >>> pd.set_option("display.max_columns", None)
    >>> df
       pitch  onset            type                                other
    0    NaN      0  time_signature   {'numerator': 4, 'denominator': 1}
    1    NaN     16  time_signature  {'numerator': 3, 'denominator': 16}

    >>> scale_time_sigs(df, 0.5)
       pitch  onset            type                                other
    0    NaN      0  time_signature   {'numerator': 4, 'denominator': 2}
    1    NaN     16  time_signature  {'numerator': 3, 'denominator': 32}


    If denominator would be fractional, we scale up the numerator (this won't always
    give totally correct results, e.g., 3/1 -> 6/1 with doubled note values which has
    different metric implications.)

    >>> scale_time_sigs(df, 2.0)
       pitch  onset            type                               other
    0    NaN      0  time_signature  {'numerator': 8, 'denominator': 1}
    1    NaN     16  time_signature  {'numerator': 3, 'denominator': 8}

    >>> scale_time_sigs(df, 0.2)  # doctest: +SKIP
    Traceback (most recent call last):
    AssertionError: factor=0.2 must be a power of 2
    """
    assert not math.log2(factor) % 1, f"{factor=} must be a power of 2"
    time_sig_mask = music_df.type == "time_signature"
    if not time_sig_mask.any():
        return music_df
    music_df = music_df.copy()
    time_sigs = music_df[music_df.type == "time_signature"]
    time_sigs.loc[:, "other"] = time_sigs["other"].apply(_to_dict_if_necessary)
    updated_time_sigs = []
    for _, time_sig in time_sigs.iterrows():
        numerator = time_sig["other"]["numerator"]
        # The denominator should get bigger when the notes get shorter
        #   and vice versa
        new_denominator = time_sig["other"]["denominator"] / factor
        while new_denominator % 1:
            numerator *= 2
            new_denominator *= 2
        assert not new_denominator % 1
        new_dict = deepcopy(time_sig["other"])
        new_dict["denominator"] = int(new_denominator)
        new_dict["numerator"] = int(numerator)
        time_sig["other"] = new_dict
        updated_time_sigs.append(time_sig)
    music_df.loc[time_sig_mask] = updated_time_sigs
    return music_df


def scale_df(df: pd.DataFrame, factor: float, metadata: bool = True) -> pd.DataFrame:
    """
    Scale all rhythmic values up/down by a power of 2.

    Besides altering all onsets/releases, this also scales the time signature
    appropriately.

    Args:
        df: a music_df
        factor: must be a power of 2
        metadata: if True, a `rhythms_scaled_by` attribute will be added/updated
            to df.attrs

    Returns:
        A new music_df with the rhythmic values scaled.

    >>> nan = float("nan")  # Alias to simplify below assignments
    >>> df = pd.DataFrame(
    ...     {
    ...         "pitch": [nan, 60, nan, 64],
    ...         "onset": [0, 0, 16, 16.5],
    ...         "release": [nan, 4, nan, 17],
    ...         "type": ["time_signature", "note", "time_signature", "note"],
    ...         "other": [
    ...             {"numerator": 4, "denominator": 1},
    ...             nan,
    ...             {"numerator": 3, "denominator": 16},
    ...             nan,
    ...         ],
    ...     }
    ... )
    >>> pd.set_option(
    ...     "display.width", 200
    ... )  # To avoid issues when the terminal is a different size
    >>> pd.set_option("display.max_columns", None)
    >>> df
       pitch  onset  release            type                                other
    0    NaN    0.0      NaN  time_signature   {'numerator': 4, 'denominator': 1}
    1   60.0    0.0      4.0            note                                  NaN
    2    NaN   16.0      NaN  time_signature  {'numerator': 3, 'denominator': 16}
    3   64.0   16.5     17.0            note                                  NaN

    >>> scale_df(df, 0.5)
       pitch  onset  release            type                                other
    0    NaN   0.00      NaN  time_signature   {'numerator': 4, 'denominator': 2}
    1   60.0   0.00      2.0            note                                  NaN
    2    NaN   8.00      NaN  time_signature  {'numerator': 3, 'denominator': 32}
    3   64.0   8.25      8.5            note                                  NaN

    >>> scale_df(df, 2.0)
       pitch  onset  release            type                               other
    0    NaN    0.0      NaN  time_signature  {'numerator': 8, 'denominator': 1}
    1   60.0    0.0      8.0            note                                 NaN
    2    NaN   32.0      NaN  time_signature  {'numerator': 3, 'denominator': 8}
    3   64.0   33.0     34.0            note                                 NaN

    >>> scale_df(df, 0.2)  # doctest: +SKIP
    Traceback (most recent call last):
    AssertionError: factor=0.2 must be a power of 2

    """
    assert not math.log2(factor) % 1, "factor must be a power of 2"
    aug_df = df.copy()
    aug_df["onset"] *= factor
    aug_df["release"] *= factor

    aug_df = scale_time_sigs(aug_df, factor)

    if metadata:
        if "rhythms_scaled_by" in aug_df.attrs:
            aug_df.attrs["rhythms_scaled_by"] *= factor
        else:
            aug_df.attrs["rhythms_scaled_by"] = factor
    return aug_df


def aug_rhythms(
    orig_data: pd.DataFrame | Iterable[pd.DataFrame],
    n_augs: int,
    n_possibilities: int = 2,
    threshold: float = 0.6547,
    metadata: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Augment one or more music_dfs rhythmically by scaling rhythms up or down.

    Example: if n_augs is 1 and n_possibilities is 2, then the returned values will
    be scaled by one of (1, 2) or (depending on the threshold) (0.5, 1), but only
    one of these values will be chosen.

    Args:
        orig_data: a music_df or an iterable of music_dfs
        n_augs: specifies the actual number of augmentations performed for each
            dataframe. Can be 1.
        n_possibilities: specifies the number of "scalings" from which the actual
            augmentations are (uniformly randomly) chosen. Must be >= n_augs.
        threshold: the threshold for the mean duration of a note to determine
            whether to scale up or down. The default value of 0.6547 was empirically
            calculated from a sample of 177 scores of Classical music.
        metadata: if True, a `rhythms_scaled_by` attribute will be added/updated
            to df.attrs

    Returns:
        An iterator of music_dfs with rhythmic values scaled. The original unscaled
        dataframe may be among the yielded values.
    """

    if isinstance(orig_data, pd.DataFrame):
        orig_data = [orig_data]

    for df in orig_data:
        mean_dur = (df.release - df.onset).mean()

        possible_pows_of_2 = [
            x
            - (n_possibilities // 2)
            + (mean_dur < threshold and n_possibilities % 2 == 0)
            for x in range(n_possibilities)
        ]

        actual_pows_of_2 = random.choices(possible_pows_of_2, k=n_augs)

        for pow_of_2 in actual_pows_of_2:
            if not pow_of_2:
                yield df
            else:
                scale_factor = 2.0**pow_of_2
                yield scale_df(df, scale_factor, metadata=metadata)


def shuffle_pitches(df: pd.DataFrame, inplace=False):
    """
    Shuffle the pitches of the notes of a music_df.

    This can be used to get a random baseline for some tasks.

    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,1.0
    ... note,64,1.0,2.0
    ... note,67,2.0,3.0
    ... note,72,3.0,4.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   60.0    0.0      1.0
    2  note   64.0    1.0      2.0
    3  note   67.0    2.0      3.0
    4  note   72.0    3.0      4.0
    >>> shuffle_pitches(df)  # doctest: +SKIP
       type  pitch  onset  release
    0   bar    NaN    0.0      4.0
    1  note   72.0    0.0      1.0
    2  note   67.0    1.0      2.0
    3  note   60.0    2.0      3.0
    4  note   64.0    3.0      4.0
    """
    note_mask = df["type"] == "note"
    pitches = df.loc[note_mask, "pitch"].tolist()
    if not inplace:
        df = df.copy()
    random.shuffle(pitches)
    df.loc[note_mask, "pitch"] = pitches
    return df


def shuffle_slices(df: pd.DataFrame, check: bool = False, include_rests: bool = True):
    """
    Shuffle the slices of a salami-sliced music_df.

    This can be used to get a random baseline for some tasks.

    Args:
        df: a music_df
        check: if True, will raise an AssertionError if the music_df is not
            salami-sliced.
        include_rests: if True, will include the rests in the shuffling.

    Returns:
        A new music_df with the slices shuffled.

    >>> csv_table = '''
    ... type,pitch,onset,release
    ... bar,,0.0,4.0
    ... note,60,0.0,0.5
    ... note,64,0.0,0.5
    ... note,60,2.0,3.0
    ... note,65,2.0,3.0
    ... bar,,4.0,8.0
    ... note,60,4.75,5.0
    ... note,67,4.75,5.0
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release
    0   bar    NaN   0.00      4.0
    1  note   60.0   0.00      0.5
    2  note   64.0   0.00      0.5
    3  note   60.0   2.00      3.0
    4  note   65.0   2.00      3.0
    5   bar    NaN   4.00      8.0
    6  note   60.0   4.75      5.0
    7  note   67.0   4.75      5.0
    >>> shuffle_slices(df)  # doctest: +SKIP
       type  pitch  onset  release
    0   bar    NaN   0.00      4.0
    1  note   60.0   1.50      2.0
    2  note   64.0   1.50      2.0
    3  note   60.0   2.00      3.0
    4  note   65.0   2.00      3.0
    5   bar    NaN   4.00      8.0
    6  note   60.0   4.75      5.0
    7  note   67.0   4.75      5.0
    """
    if check:
        assert appears_salami_sliced(df)

    note_mask = df["type"] == "note"
    note_df = df[note_mask]

    if include_rests:
        rests = return_rests(note_df)
        rest_df = pd.DataFrame(
            [
                {"type": "rest", "onset": onset, "release": release}
                for (onset, release) in rests
            ]
        )
        # I don't think it is necessary to sort the result so that the rests
        # occur in the correct location since we are shuffling immediately afterwards
        note_df = pd.concat([note_df, rest_df], axis=0)

    note_df["duration"] = note_df["release"] - note_df["onset"]

    note_groups = [group for _, group in note_df.groupby("onset")]

    random.shuffle(note_groups)
    onset = 0

    for note_group in note_groups:
        note_group["onset"] = onset
        onset += note_group.iloc[0]["duration"]

    shuffled_note_df = pd.concat(note_groups)
    shuffled_note_df["release"] = (
        shuffled_note_df["onset"] + shuffled_note_df["duration"]
    )
    shuffled_note_df = shuffled_note_df.drop("duration", axis=1)

    if include_rests:
        # drop the added rests
        shuffled_note_df = shuffled_note_df[shuffled_note_df["type"] == "note"]

    out_df = pd.concat((shuffled_note_df, df[~note_mask]))
    return sort_df(out_df)
