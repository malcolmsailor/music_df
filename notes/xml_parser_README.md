This is a musicxml parser. It is not very full-featured because I have implemented it purely for my own purposes.

The return format is a Pandas DataFrame.

Only certain types of events are returned, namely: <!-- TODO Keep this list updated! -->

- notes
- time signatures
- barlines

Supported formats are:

- .xml (musicxml)
- .mxl (compressed musicxml)
- .mscx or .mscz (MuseScore format; these are first converted to xml format using MuseScore; if `mscore` is not in your path, will raise a ValueError)

# Usage

```
>>> import xml_to_note_table
>>> from xml_to_note_table import parse
>>> parse("tests/resources/temp.xml")
/Users/malcolm/google_drive/python/malmus/xml_to_note_table/xml_to_note_table/ties.py:215: UserWarning: Release of note at 76 < onset of note at 140
  warnings.warn(
      pitch onset release tie_to_next tie_to_prev  grace voice part spelling instrument  midi_instrument            type                               other
0       NaN     0       4         NaN         NaN    NaN   NaN  NaN      NaN        NaN              NaN             bar                                 NaN
1       NaN     0    None         NaN         NaN    NaN   NaN  NaN      NaN        NaN              NaN  time_signature  {'numerator': 4, 'denominator': 4}
2      55.0     0       3       False       False  False     1   P3        G       Alto              1.0            note                                 NaN
3      55.0     3       4       False       False  False     1   P3        G       Alto              1.0            note                                 NaN
4       NaN     4       8         NaN         NaN    NaN   NaN  NaN      NaN        NaN              NaN             bar                                 NaN
...     ...   ...     ...         ...         ...    ...   ...  ...      ...        ...              ...             ...                                 ...
1221   50.0   324     328       False       False  False     1   P6        D   Continuo              1.0            note                                 NaN
1222   57.0   324     328       False       False  False     1   P3        A       Alto              1.0            note                                 NaN
1223   66.0   324     328       False       False  False     1   P4       F#      Tenor              1.0            note                                 NaN
1224   74.0   324     328       False       False  False     1   P1        D      Canto              1.0            note                                 NaN
1225   74.0   324     328       False       False  False     1   P2        D     Quinto              1.0            note                                 NaN

[1226 rows x 11 columns]
```

# TODO

- text-based repeats (da capo, dal segna, etc.) are not yet implemented
- I _think_ first/second endings are implemented
