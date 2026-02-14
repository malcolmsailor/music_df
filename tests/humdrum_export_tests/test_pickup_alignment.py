"""Tests for pickup alignment and related bug fixes.

Fix 1: _align_pickup_durations pads shorter pickups with rests
Fix 2: merge_spines gives a descriptive error on misaligned barlines
"""

import pytest

from music_df.humdrum_export.merge_spines import merge_spines


class TestMergeSpinesMisalignedBarlines:
    """Fix 2: merge_spines should raise ValueError (not bare AssertionError)
    when barlines are misaligned across spines."""

    def test_raises_value_error(self):
        # Spine 0 has a rest where spine 1 has a barline
        humdrum = "\n".join(
            [
                "\t".join(["**kern", "**kern"]),
                "\t".join(["4r", "="]),
                "\t".join(["=", "4c"]),
                "\t".join(["1c", "1e"]),
                "\t".join(["=", "="]),
            ]
        )
        with pytest.raises(ValueError, match="barline"):
            merge_spines(humdrum)


class TestAlignPickupDurations:
    """Fix 1: spines with different pickup durations should be aligned
    by prepending rests to shorter pickups."""

    def test_no_change_when_aligned(self):
        from music_df.humdrum_export.humdrum_export import _align_pickup_durations

        spines = [
            ["*M4/4", "4c", "=", "1d", "="],
            ["*M4/4", "4e", "=", "1f", "="],
        ]
        result = _align_pickup_durations(spines)
        assert result == spines

    def test_pads_shorter_pickup(self):
        from music_df.humdrum_export.humdrum_export import _align_pickup_durations
        from music_df.humdrum_export.df_to_spines import kern_to_float_dur

        # Spine 0: 2.0-beat pickup (half note)
        # Spine 1: 1.0-beat pickup (quarter note)
        spines = [
            ["*M4/4", "2cc", "=", "1dd", "="],
            ["*M4/4", "4ee", "=", "1ff", "="],
        ]
        result = _align_pickup_durations(spines)

        # Spine 0 unchanged
        assert result[0] == spines[0]

        # Spine 1 should now have 2.0 beats before the barline
        first_barline_idx = result[1].index("=")
        pickup_tokens = [
            t for t in result[1][:first_barline_idx] if not t.startswith("*")
        ]
        pickup_dur = sum(kern_to_float_dur(t.split(" ")[0]) for t in pickup_tokens)
        assert abs(pickup_dur - 2.0) < 1e-9

    def test_pads_64th_note_difference(self):
        """The typical case from the bug report: 0.0625 beat difference."""
        from music_df.humdrum_export.humdrum_export import _align_pickup_durations
        from music_df.humdrum_export.df_to_spines import kern_to_float_dur

        # Spine 0: 1.9375 beats (4r + 8...r)
        # Spine 1: 2.0 beats (4.D + 16F + 16B-)
        spines = [
            ["*M3/4", "4r", "8...r", "=", "2.cc", "="],
            ["*M3/4", "4.dd", "16ff", "16b-", "=", "2.ee", "="],
        ]
        result = _align_pickup_durations(spines)

        # Spine 1 unchanged (it has the longer pickup)
        assert result[1] == spines[1]

        # Spine 0 should now have 2.0 beats before the barline
        first_barline_idx = result[0].index("=")
        pickup_tokens = [
            t for t in result[0][:first_barline_idx] if not t.startswith("*")
        ]
        pickup_dur = sum(kern_to_float_dur(t.split(" ")[0]) for t in pickup_tokens)
        assert abs(pickup_dur - 2.0) < 1e-9

    def test_no_pickup(self):
        """If all spines start with a barline, no padding needed."""
        from music_df.humdrum_export.humdrum_export import _align_pickup_durations

        spines = [
            ["*M4/4", "=", "1c", "="],
            ["*M4/4", "=", "1e", "="],
        ]
        result = _align_pickup_durations(spines)
        assert result == spines

    def test_single_spine(self):
        """A single spine needs no alignment."""
        from music_df.humdrum_export.humdrum_export import _align_pickup_durations

        spines = [["*M4/4", "2cc", "=", "1dd", "="]]
        result = _align_pickup_durations(spines)
        assert result == spines

    def test_integration_collate_and_merge(self):
        """Aligned spines should collate and merge successfully."""
        from music_df.humdrum_export.collate_spines import collate_spines
        from music_df.humdrum_export.humdrum_export import (
            _align_pickup_durations,
            _get_temp_paths,
            _write_spine,
        )

        # Spines with misaligned pickups
        spines = [
            ["*M4/4", "2cc", "=", "1dd", "="],
            ["*M4/4", "4ee", "=", "1ff", "="],
        ]

        aligned = _align_pickup_durations(spines)

        with _get_temp_paths(len(aligned)) as paths:
            for path, spine in zip(paths, aligned):
                _write_spine(spine, path)
            collated = collate_spines(paths)
            merged = merge_spines(collated)

        assert len(merged) > 0
