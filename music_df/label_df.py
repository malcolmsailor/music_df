from typing import Any, Sequence

import pandas as pd


def label_df(
    music_df: pd.DataFrame,
    labels: Sequence[Any],
    label_indices: pd.Series | pd.Index | Sequence[int] | None,
    label_col_name: str,
    inplace: bool = False,
    null_label: Any = None,
) -> pd.DataFrame:
    if not inplace:
        music_df = music_df.copy()
    if label_indices is None:
        # label_indices = range(len(labels))
        label_indices = music_df.index

    music_df[label_col_name] = null_label

    if isinstance(label_indices, pd.Series) or isinstance(label_indices, pd.Index):
        music_df.loc[label_indices, label_col_name] = labels  # type:ignore
    else:
        for label, i in zip(labels, label_indices):
            music_df.loc[i, label_col_name] = label

    return music_df
