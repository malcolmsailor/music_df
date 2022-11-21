from types import MappingProxyType

import pandas as pd


DF_TYPE_SORT_ORDER = MappingProxyType(
    {"bar": 0, "time_signature": 1, "note": 2}
)


def sort_df(df: pd.DataFrame, inplace: bool = False):
    if not inplace:
        df = df.sort_values(
            by="release",
            axis=0,
            inplace=False,
            ignore_index=True,
            key=lambda x: 0 if x is None else x,
        )
    else:
        df.sort_values(
            by="release",
            axis=0,
            inplace=True,
            ignore_index=True,
            key=lambda x: 0 if x is None else x,
        )
    df.sort_values(
        by="pitch",
        axis=0,
        inplace=True,
        ignore_index=True,
        key=lambda x: 128 if x is None else x,
        kind="mergesort",  # default sort is not stable
    )
    if "type" in df.columns:
        df.sort_values(
            by="type",
            axis=0,
            inplace=True,
            ignore_index=True,
            key=lambda x: x.map(DF_TYPE_SORT_ORDER),
            kind="mergesort",  # default sort is not stable
        )
    df.sort_values(
        by="onset",
        axis=0,
        inplace=True,
        ignore_index=True,
        kind="mergesort",  # default sort is not stable
    )
    return df
