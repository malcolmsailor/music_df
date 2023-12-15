import pandas as pd

from music_df import sort_df
from music_df.xml_parser import xml_parse


def read_xml(input_path: str, sort: bool = True) -> pd.DataFrame:
    music_df = xml_parse(input_path)
    # TODO: (Malcolm 2023-12-15) sort inside xml_parse ?
    if sort:
        return sort_df(music_df)
    return music_df
