import io  # used by doctest
from collections import defaultdict

import pandas as pd

from music_df import sort_df


class MergeGroup:
    def __init__(self):
        self.rows = []
        self.max_release = float("-inf")

    def add_row(self, row):
        self.rows.append(row)
        self.max_release = max(row.release, self.max_release)


def merge_notes(
    df: pd.DataFrame,
    attrs_to_merge_on: tuple[str, ...] = ("type", "pitch"),
    store_unmerged_indices: bool = True,
):
    """
    >>> csv_table = '''
    ... type,pitch,onset,release
    ... note,60,0.0,0.5
    ... note,60,0.0,1.5
    ... note,60,1.0,2.0
    ... note,60,2.0,3.0
    ... note,64,2.0,3.0
    ... note,64,3.5,4.0
    ... note,64,3.6,3.7
    ... note,60,4.0,4.4
    ... note,60,4.0,4.1
    ... '''
    >>> df = pd.read_csv(io.StringIO(csv_table.strip()))
    >>> df
       type  pitch  onset  release
    0  note     60    0.0      0.5
    1  note     60    0.0      1.5
    2  note     60    1.0      2.0
    3  note     60    2.0      3.0
    4  note     64    2.0      3.0
    5  note     64    3.5      4.0
    6  note     64    3.6      3.7
    7  note     60    4.0      4.4
    8  note     60    4.0      4.1
    >>> merged_df = merge_notes(df)
    >>> merged_df
       type  pitch  onset  release unmerged_indices
    0  note     60    0.0      3.0          0,1,2,3
    1  note     64    2.0      3.0                4
    2  note     64    3.5      4.0              5,6
    3  note     60    4.0      4.4              7,8
    >>> merged_df.attrs["merged_notes"]
    True
    >>> merge_notes(df, store_unmerged_indices=False)
       type  pitch  onset  release
    0  note     60    0.0      3.0
    1  note     64    2.0      3.0
    2  note     64    3.5      4.0
    3  note     60    4.0      4.4
    """
    working_area = defaultdict(MergeGroup)
    groups = []

    # assumes df is sorted by onset
    for _, row in df.iterrows():
        row_attrs = tuple(row[attr] for attr in attrs_to_merge_on)
        if row_attrs not in working_area:
            working_area[row_attrs].add_row(row)
            continue
        else:
            current_release = working_area[row_attrs].max_release
            if row.onset <= current_release:
                working_area[row_attrs].add_row(row)
            else:
                groups.append(working_area.pop(row_attrs))
                working_area[row_attrs].add_row(row)

    groups.extend(list(working_area.values()))
    row_accumulator = []
    for group in groups:
        new_row = group.rows[0].copy()
        new_row.release = group.max_release
        if store_unmerged_indices:
            new_row["unmerged_indices"] = ",".join(str(row.name) for row in group.rows)
        row_accumulator.append(new_row)

    new_df = pd.DataFrame(row_accumulator)

    new_df.attrs = df.attrs.copy()
    new_df.attrs["merged_notes"] = True
    new_df = sort_df(new_df)
    return new_df
