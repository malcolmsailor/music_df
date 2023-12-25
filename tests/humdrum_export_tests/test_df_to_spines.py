from music_df.humdrum_export.df_to_spines import df_to_spines
from music_df.xml_parser import xml_parse


# TODO: (Malcolm 2023-09-28) update not to use hardcoded paths outside of this repo
def test_df_to_spines():
    df = xml_parse(
        "/Users/malcolm/datasets/OpenScore/Lieder/scores/Elgar,_Edward/_/Is_She_Not_Passing_Fair/lc6546527.mscx"
    )
    df_to_spines(df)
