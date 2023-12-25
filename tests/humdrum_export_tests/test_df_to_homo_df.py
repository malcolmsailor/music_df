from music_df.humdrum_export.df_to_homo_df import df_to_homo_df
from music_df.xml_parser import xml_parse


def test_df_to_homo_df():
    df = xml_parse(
        "/Users/malcolm/datasets/OpenScore/Lieder/scores/Elgar,_Edward/_/Is_She_Not_Passing_Fair/lc6546527.mscx"
    )
    homo_dfs = df_to_homo_df(df)
    for homo_df in homo_dfs:
        for (_, note1), (_, note2) in zip(
            homo_df.iterrows(), homo_df.iloc[1:].iterrows()
        ):
            if note1.type != "note" or note2.type != "note":
                continue
            assert (note1.onset == note2.onset and note1.release == note2.release) or (
                note2.onset >= note1.release
            )
