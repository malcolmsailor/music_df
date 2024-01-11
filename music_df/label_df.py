from typing import Any, Sequence

import pandas as pd


def label_df(
    music_df: pd.DataFrame,
    labels: Sequence[Any],
    label_indices: Sequence[int] | None,
    label_col_name: str,
    inplace: bool = False,
) -> pd.DataFrame:
    if not inplace:
        music_df = music_df.copy()
    if label_indices is None:
        label_indices = range(len(labels))

    music_df[label_col_name] = None
    for label, i in zip(labels, label_indices):
        music_df.loc[i, label_col_name] = label

    return music_df
