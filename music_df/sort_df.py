from types import MappingProxyType

import pandas as pd


class SortOrderMapping(dict):
    def __init__(self, *args, missing_value, **kwargs):
        super().__init__(*args, **kwargs)
        self._missing_value = missing_value

    def __missing__(self, key):
        return self._missing_value


DF_TYPE_SORT_ORDER = SortOrderMapping(
    {"bar": 1, "time_signature": 0, "note": 3}, missing_value=2
)


def sort_df(df: pd.DataFrame, inplace: bool = False, ignore_index: bool = True):
    if not inplace:
        df = df.sort_values(
            by="release",
            axis=0,
            inplace=False,
            ignore_index=ignore_index,
            key=lambda x: 0 if x is None else x,
            kind="mergesort",  # default sort is not stable
        )
    else:
        df.sort_values(
            by="release",
            axis=0,
            inplace=True,
            ignore_index=ignore_index,
            key=lambda x: 0 if x is None else x,
            kind="mergesort",  # default sort is not stable
        )
    df.sort_values(
        by="pitch",
        axis=0,
        inplace=True,
        ignore_index=ignore_index,
        key=lambda x: 128 if x is None else x,
        kind="mergesort",  # default sort is not stable
    )
    if "type" in df.columns:
        # We first sort by type so that result of sort will always be the same (i.e.,
        #   the "all other" rows below will be sorted)
        df.sort_values(
            by="type", axis=0, inplace=True, ignore_index=ignore_index, kind="mergesort"
        )
        # Then we sort by type again to make sure we have rows in the following order:
        #   time signature
        #   bar
        #   all other
        #   note
        df.sort_values(
            by="type",
            axis=0,
            inplace=True,
            ignore_index=ignore_index,
            key=lambda x: x.map(DF_TYPE_SORT_ORDER),
            kind="mergesort",  # default sort is not stable
        )
    df.sort_values(
        by="onset",
        axis=0,
        inplace=True,
        ignore_index=ignore_index,
        kind="mergesort",  # default sort is not stable
    )
    return df
