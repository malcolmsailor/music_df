import random
from ast import literal_eval
from typing import Iterable, Iterator

import pandas as pd

from music_df.transpose import transpose_to_key

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
    hi: int | None = None,
    low: int | None = None,
) -> Iterator[pd.DataFrame]:
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


def _to_dict_if_necessary(d):
    if isinstance(d, str):
        return literal_eval(d)
    return d


def scale_time_sigs(music_df: pd.DataFrame, factor: int | float) -> pd.DataFrame:
    time_sig_mask = music_df.type == "time_signature"
    if not time_sig_mask.any():
        return music_df
    time_sigs = music_df[music_df.type == "time_signature"]
    time_sigs.loc[:, "other"] = time_sigs["other"].apply(_to_dict_if_necessary)
    for _, time_sig in time_sigs.iterrows():
        new_denominator = factor * time_sig["other"]["denominator"]
        assert not new_denominator % 1
        time_sig["other"]["denominator"] = int(new_denominator)
    music_df.loc[time_sig_mask] = time_sigs
    return music_df


def aug_rhythms(
    orig_data: pd.DataFrame | Iterable[pd.DataFrame],
    n_augs: int,
    n_possibilities: int = 2,
    threshold: float = 0.6547667782160375,
    metadata: bool = True,
) -> Iterator[pd.DataFrame]:
    """
    Example: if n_augs is 1 and n_possibilities is 2, then the returned values will
    be scaled by one of (1, 2) or (depending on the threshold) (0.5, 1), but only
    one of these values will be chosen.

    default threshold was empirically calculated from a sample of 177 scores

    Args:
        n_augs: specifies the actual number of augmentations. Can be 1.
        n_possibilities: specifies the number of "scalings" from which the actual
            augmentations are chosen. Must be >= n_augs.
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

                aug_df = df.copy()
                aug_df["onset"] *= scale_factor
                aug_df["release"] *= scale_factor

                # The denominator should get bigger when the notes get shorter
                #   and vice versa
                scale_time_sigs(aug_df, 1 / scale_factor)

                if metadata:
                    if "rhythms_scaled_by" in aug_df.attrs:
                        aug_df.attrs["rhythms_scaled_by"] *= scale_factor
                    else:
                        aug_df.attrs["rhythms_scaled_by"] = scale_factor
                yield aug_df
