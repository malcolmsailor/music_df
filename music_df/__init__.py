"""
.. include:: ../README.md
   :end-before: Documentation

Among the more useful parts of the package worth highlighting:

- `music_df.read`: read in a variety of music file formats to a dataframe
- `music_df.add_feature`: infer/adjust features like barlines, time signatures, etc.
- `music_df.salami_slice`: salami-slice a dataframe
- `music_df.augmentations`: apply augmentations (e.g., transposition) to a dataframe
- `music_df.transpose`: transpose the pitches of a dataframe

The documentation is currently a work in progress.
"""

# TODO: (Malcolm 2025-03-06) the selection of names that are imported here is
#   somewhat arbitrary.
from .quantize_df import quantize_df
from .read_csv import read_csv
from .read_krn import read_krn, read_krn_via_xml
from .segment_df import (
    get_df_segment_indices,
    get_eligible_onsets,
    get_eligible_releases,
    segment_df,
)
from .sort_df import sort_df
from .split_df import split_musicdf
from .transpose import chromatic_transpose

__all__ = [
    "quantize_df",
    "read_csv",
    "read_krn",
    "read_krn_via_xml",
    "segment_df",
    "sort_df",
    "split_musicdf",
    "chromatic_transpose",
    "get_df_segment_indices",
    "get_eligible_onsets",
    "get_eligible_releases",
]

# Unfortunately the following approach means that the documentation of the submodules
#    isn't generated either.
# __pdoc__ = {
#     # because the selection of names that are imported here is somewhat arbitrary,
#     # I prefer that the documentation be found in the sub-modules rather than here.
#     "quantize_df": False,
#     "read_csv": False,
#     "read_krn": False,
#     "read_krn_via_xml": False,
#     "segment_df": False,
#     "sort_df": False,
#     "split_musicdf": False,
#     "chromatic_transpose": False,
#     "get_df_segment_indices": False,
#     "get_eligible_onsets": False,
#     "get_eligible_releases": False,
# }
