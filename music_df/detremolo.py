from typing import Iterable
import pandas as pd


def detremolo(
    df: pd.DataFrame,
    max_tremolo_note_length: float = 0.25,
    max_tremolo_note_gap: float = 0.125,
    instrument_columns: Iterable[str] = (
        "instrument",
        "midi_instrument",
        "track",
        "channel",
    ),
) -> pd.DataFrame:
    tremoli = []
    for _, instr in df[df["type"] == "note"].groupby(
        [c for c in instrument_columns if c in df.columns]
    ):
        for _, group in instr.groupby(["pitch"]):
            current_tremolo = None

            # I think that this will miss the last note in the case where tremolo extends to the
            #   last note

            for (row1_i, note1), (row2_i, note2) in zip(
                group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()
            ):
                if (note1.release - note1.onset <= max_tremolo_note_length) and (
                    note2.onset - note1.release <= max_tremolo_note_gap
                ):
                    if current_tremolo is None:
                        current_tremolo = []
                    current_tremolo.append(row1_i)
                else:
                    if current_tremolo is not None:
                        tremoli.append(current_tremolo)
                        current_tremolo = None
            if current_tremolo is not None:
                tremoli.append(current_tremolo)

    out_df = df.copy()

    to_drop = []

    for tremolo in tremoli:
        out_df.loc[tremolo[0], "release"] = out_df.loc[tremolo[-1], "release"]
        to_drop.extend(tremolo[1:])

    return out_df.drop(to_drop, axis=0)
