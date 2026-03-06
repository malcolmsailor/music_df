from fractions import Fraction

import pytest
from metricker import Meter

from music_df.humdrum_export.dur_to_kern import _Dur, dur_to_kern, duration_float_to_recip
from music_df.humdrum_export.fallback_meter import BeatBoundaryMeter, make_meter


class TestBeatBoundaryMeterInit:
    def test_5_4_beat_dur(self):
        m = BeatBoundaryMeter("5/4")
        assert m._beat_dur == Fraction(1)

    def test_7_8_beat_dur(self):
        m = BeatBoundaryMeter("7/8")
        assert m._beat_dur == Fraction(1, 2)

    def test_7_4_beat_dur(self):
        m = BeatBoundaryMeter("7/4")
        assert m._beat_dur == Fraction(1)


class TestSplitAtMetricStrongPoints:
    def test_no_split_within_beat(self):
        m = BeatBoundaryMeter("5/4")
        items = [_Dur(0.0, 0.5)]
        result = m.split_at_metric_strong_points(items)
        assert len(result) == 1

    def test_split_across_beats_5_4(self):
        m = BeatBoundaryMeter("5/4")
        items = [_Dur(0.0, 3.0)]
        result = m.split_at_metric_strong_points(items)
        assert len(result) == 3
        assert result[0].onset == 0.0
        assert result[0].release == 1.0
        assert result[1].onset == 1.0
        assert result[1].release == 2.0
        assert result[2].onset == 2.0
        assert result[2].release == 3.0

    def test_split_across_beats_7_8(self):
        m = BeatBoundaryMeter("7/8")
        items = [_Dur(0.0, 1.5)]
        result = m.split_at_metric_strong_points(items)
        # Beat boundaries at 0.5, 1.0 inside (0, 1.5)
        assert len(result) == 3

    def test_min_split_dur_filters(self):
        m = BeatBoundaryMeter("5/4")
        # Duration from 0.0 to 1.25: beat boundary at 1.0
        # Left fragment = 1.0, right = 0.25; with min_split_dur=0.5, right is too short
        items = [_Dur(0.0, 1.25)]
        result = m.split_at_metric_strong_points(items, min_split_dur=0.5)
        assert len(result) == 1

    def test_force_split(self):
        m = BeatBoundaryMeter("5/4")
        items = [_Dur(0.0, 1.25)]
        result = m.split_at_metric_strong_points(
            items, min_split_dur=0.5, force_split=True
        )
        assert len(result) == 2

    def test_multiple_items(self):
        m = BeatBoundaryMeter("5/4")
        items = [_Dur(0.0, 2.0), _Dur(2.0, 4.0)]
        result = m.split_at_metric_strong_points(items)
        assert len(result) == 4

    def test_does_not_mutate_original(self):
        m = BeatBoundaryMeter("5/4")
        item = _Dur(0.0, 3.0)
        result = m.split_at_metric_strong_points([item])
        assert item.onset == 0.0
        assert item.release == 3.0
        assert len(result) == 3


class TestSplitOddDuration:
    def test_representable_unchanged(self):
        m = BeatBoundaryMeter("5/4")
        item = _Dur(0.0, 1.0)
        result = m.split_odd_duration(item)
        assert len(result) == 1

    def test_unrepresentable_gets_split(self):
        m = BeatBoundaryMeter("5/4")
        # 5 quarter notes is not representable as a single kern duration
        item = _Dur(0.0, 5.0)
        result = m.split_odd_duration(item)
        assert len(result) > 1
        total = sum(r.release - r.onset for r in result)
        assert abs(total - 5.0) < 1e-9


class TestMakeMeter:
    def test_standard_returns_meter(self):
        m = make_meter("4/4")
        assert isinstance(m, Meter)

    def test_standard_3_4(self):
        m = make_meter("3/4")
        assert isinstance(m, Meter)

    def test_unsupported_returns_fallback(self):
        m = make_meter("5/4")
        assert isinstance(m, BeatBoundaryMeter)

    def test_unsupported_7_8(self):
        m = make_meter("7/8")
        assert isinstance(m, BeatBoundaryMeter)


class TestDurToKernEndToEnd:
    def test_5_4_no_crash(self):
        result = dur_to_kern(2.0, 0.0, "5/4")
        total = sum(d for d, _ in result)
        assert abs(total - 2.0) < 1e-9
        for _, kern in result:
            assert not kern.startswith("q"), f"Unrepresentable kern duration: {kern}"

    def test_7_8_no_crash(self):
        result = dur_to_kern(1.5, 0.0, "7/8")
        total = sum(d for d, _ in result)
        assert abs(total - 1.5) < 1e-9
        for _, kern in result:
            assert not kern.startswith("q"), f"Unrepresentable kern duration: {kern}"

    def test_5_4_full_measure(self):
        result = dur_to_kern(5.0, 0.0, "5/4")
        total = sum(d for d, _ in result)
        assert abs(total - 5.0) < 1e-9
        for _, kern in result:
            assert not kern.startswith("q"), f"Unrepresentable kern duration: {kern}"

    def test_7_4_no_crash(self):
        result = dur_to_kern(3.0, 0.0, "7/4")
        total = sum(d for d, _ in result)
        assert abs(total - 3.0) < 1e-9

    def test_standard_meter_still_works(self):
        result = dur_to_kern(2.0, 0.0, "4/4")
        total = sum(d for d, _ in result)
        assert abs(total - 2.0) < 1e-9
