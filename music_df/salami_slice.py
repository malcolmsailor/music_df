import pandas as pd
from music_df.sort_df import sort_df


def salami_slice(df: pd.DataFrame) -> pd.DataFrame:
    # TODO implement a "minimum tolerance" so that onsets/releases don't
    #   have to be *exactly* simultaneous
    # any zero-length notes will be omitted.
    # Given that all onsets/releases will be homophonic after running
    #   this function, there would be a more efficient way of storing notes
    #   than storing each one individually, but then we would have to rewrite
    #   repr functions for the output.
    if len(df) == 0:
        return df.copy()
    moments = sorted(
        set(df[df.type == "note"].onset) | set(df[df.type == "note"].release)
    )
    moment_iter = enumerate(moments)
    moment_i, moment = next(moment_iter)
    out = []
    for _, note in df.iterrows():
        if note.type != "note":
            out.append(note.copy())
            continue
        while note.onset > moment:
            moment_i, moment = next(moment_iter)
        onset = note.onset
        release_i = moment_i + 1
        while release_i < len(moments) and moments[release_i] <= note.release:
            new_note = note.copy()
            new_note.onset = onset
            new_note.release = onset = moments[release_i]
            out.append(new_note)
            release_i += 1
    new_df = pd.DataFrame(out)
    sort_df(new_df, inplace=True)
    return new_df
