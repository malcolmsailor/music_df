from music_df.humdrum_export.merge_spines import _sort_measure_spines


def test_sort_measure_spines_with_chord_tokens():
    """Chord tokens (space-separated sub-tokens) should have all their
    pitches averaged when sorting spines, not just the last pitch letter."""
    # Spine 0 has a chord: CC (~36) and ee (~76), correct mean ~56
    # Spine 1 has a single note: dd (~74)
    # Correct sort: spine 1 (74) should come before spine 0 (56)
    # Bug: without splitting, regex extracts just 'ee' (76) from the chord
    #   token, so spine 0 (76) incorrectly sorts first
    measure = [
        ["4CC 4ee"],  # chord: low + high, mean ~56
        ["4dd"],  # single note, ~74
    ]
    _sort_measure_spines(measure)
    assert measure[0] == ["4dd"], (
        "Spine with higher mean pitch should sort first; "
        "chord tokens must be split before averaging"
    )
