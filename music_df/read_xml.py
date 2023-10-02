import pandas as pd

from music_df import sort_df


def read_xml(input_path: str, sort: bool = True) -> pd.DataFrame:
    from xml_to_note_table import parse  # type:ignore

    music_df = parse(input_path)
    if sort:
        return sort_df(music_df)
    return music_df
