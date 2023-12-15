import pandas as pd

from music_df import sort_df
from music_df.xml_parser import xml_parse


def read_xml(input_path: str, sort: bool = True) -> pd.DataFrame:
    music_df = xml_parse(input_path, sort=sort)
    return music_df
