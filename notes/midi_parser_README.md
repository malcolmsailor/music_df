A library for reading midi files into Pandas dataframes.

The main function for reading files is `midi_to_table`. By default it returns a Pandas dataframe but for legacy reasons it can also return a list of lists. (Possibly I could deprecate this.)

There is also an inverse function, `df_to_midi`. I believe as of 2022-11-28 the only events types that this writes are `note_on`, `note_off`, `time_signature`, and `pitchwheel`.
