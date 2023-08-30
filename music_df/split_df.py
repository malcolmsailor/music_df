from typing import Any, Literal

import pandas as pd


def split_musicdf(
    music_df: pd.DataFrame, split_by_track: bool = True, split_by_channel: bool = True
) -> dict[Any, pd.DataFrame]:
    if split_by_channel and split_by_track:
        if "channel" in music_df.columns:
            grouping = ["track", "channel"]
        else:
            grouping = "track"
    elif split_by_channel:
        grouping = "channel"
    elif split_by_track:
        grouping = "track"
    else:
        raise ValueError

    grouped = music_df.groupby(grouping)

    output = {key: df.copy() for key, df in grouped}
    return output
