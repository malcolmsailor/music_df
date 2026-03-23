 Italian augmented sixths are triads (3 notes); French and German augmented
 sixths are four-note chords (like seventh chords).

 Changes needed in music_df

 1. music_df/chord_df.py:1072 — inversion_number_to_figure()

 Current:
 if "7" in quality or quality == "aug6":

 Change to:
 if "7" in quality or quality in {"Fr", "Ger", "aug6"}:

 This ensures:
 - "It" → triad inversions (0→"", 1→"6", 2→"64")
 - "Fr", "Ger" → seventh chord inversions (0→"7", 1→"65", 2→"43", 3→"42")
 - "aug6" remains for backwards compatibility

 Update the comment on line 1065-1067 — with the new schema we can
 distinguish German/French/Italian, so the note about not being able to
 distinguish them is no longer accurate.

 Update doctests to use the new quality values.

 2. music_df/harmony/modulation.py:284-285

 Current regex pattern:
 "Mm7|aug6|o|\\+"

 Change to:
 "Mm7|It|Fr|Ger|aug6|o|\\+"

 (Both the non_dominant_mask and dominant_mask lines.)

 Also update the docstring example at line 205/219 that shows aug6.

 3. music_df/harmony/chords.py:312-320 — _translate_single_rnbert_part()

 Discuss update with user.

 4. music_df/chord_df.py:822-831 — get_quality_for_merging()

 Comment mentions removing "6 from augmented 6 chords". Review whether this
 function needs updating for the new quality names.

 5. Test data

 - tests/resources/example_dfs/K330-3_chords.csv has rows with aug6 quality.
 If these are used in tests that check quality values, update them to use
 "It", "Fr", or "Ger" as appropriate.
 - tests/test_chord_df_round_trips.py:335-338 tests rnbert aug6 round-trips.
