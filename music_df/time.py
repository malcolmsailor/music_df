from fractions import Fraction

import pandas as pd

from music_df.utils.search import get_index_to_item_leq


def appears_to_have_pickup_measure(music_df: pd.DataFrame) -> bool:
    time_sig_mask = music_df.type == "time_signature"
    if not time_sig_mask.any() or music_df[time_sig_mask].iloc[0].onset != 0:
        # Return whether first measure is shorter than second measure
        pass
    # TODO: (Malcolm 2023-12-24)

    raise NotImplementedError


def time_to_bar_number_and_offset(
    music_df: pd.DataFrame, x: float | Fraction | int
) -> tuple[int, float]:
    """ """
    bar_mask = music_df.type == "bar"
    assert bar_mask.any()
    bars = music_df[bar_mask]
    bar_i = get_index_to_item_leq(bars.onset, val=x)
    assert isinstance(bar_i, int)
    bar_number = bar_i if appears_to_have_pickup_measure(music_df) else bar_i + 1
    bar_onset = float(bars.loc[bar_i, "onset"])  # type:ignore
    offset = float(x - bar_onset)
    return bar_number, offset
