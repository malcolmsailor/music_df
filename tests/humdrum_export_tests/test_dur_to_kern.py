import pytest

from music_df.humdrum_export.dur_to_kern import KernDurError, dur_to_kern


class TestZeroDuration:
    """dur_to_kern should return [] for durations that snap to zero."""

    @pytest.mark.parametrize("inp", [0.0, 1e-9, 0.001])
    def test_zero_or_near_zero_duration_returns_empty(self, inp):
        result = dur_to_kern(inp, 0.0, "2/4")
        assert result == []

    @pytest.mark.parametrize("inp", [0.0, 1e-9, 0.001])
    def test_zero_duration_with_raise_flag(self, inp):
        result = dur_to_kern(
            inp, 0.0, "2/4", raise_exception_on_unrecognized_duration=True,
        )
        assert result == []


class TestTripletFloatDriftAccumulation:
    """Simulate the pipeline that produces drifted values from truncated
    triplet durations (as totable outputs them)."""

    def test_accumulated_drift_does_not_raise(self):
        # totable truncates 1/6 to 4 decimal places
        truncated_triplet_eighth = round(1 / 6, 4)  # 0.1667

        # Simulate 15 consecutive triplet eighths, accumulating onsets
        onsets = [0.0]
        for i in range(15):
            onsets.append(onsets[-1] + truncated_triplet_eighth)

        # After 6 triplet eighths the onset has drifted from the exact
        # value of 1.0 (= 6/6). Compute a rest spanning from that drifted
        # onset to a later barline in 3/8 meter (barline at 1.5).
        drifted_onset = onsets[6]  # ~1.0002, should be 1.0
        measure_end = 1.5
        drifted_dur = measure_end - drifted_onset  # ~0.4998, should be 0.5

        assert drifted_onset != 1.0, "Expected drifted onset"

        result = dur_to_kern(
            drifted_dur, drifted_onset, "3/8",
            raise_exception_on_unrecognized_duration=True,
        )
        assert len(result) > 0
        total = sum(d for d, _ in result)
        assert abs(total - 0.5) < 0.01


class TestDriftedValuesFromBeethovenB076:
    """Regression tests with the specific drifted values from Beethoven B076."""

    CASES = [
        (0.4170000000000016, 1.0829999999999984, "3/8"),
        (0.9170000000000016, 0.5829999999999984, "3/8"),
        (1.2919999999999732, 0.20800000000002683, "3/8"),
    ]

    @pytest.mark.parametrize("inp,offset,meter", CASES)
    def test_drifted_values_do_not_raise(self, inp, offset, meter):
        result = dur_to_kern(
            inp, offset, meter,
            raise_exception_on_unrecognized_duration=True,
        )
        assert len(result) > 0

    @pytest.mark.parametrize("inp,offset,meter", CASES)
    def test_output_durations_sum_to_input(self, inp, offset, meter):
        result = dur_to_kern(
            inp, offset, meter,
            raise_exception_on_unrecognized_duration=True,
        )
        total = sum(d for d, _ in result)
        assert abs(total - inp) < 0.01, (
            f"Output durations sum {total} != input {inp}"
        )
