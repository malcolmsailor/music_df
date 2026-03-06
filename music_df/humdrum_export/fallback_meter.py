import copy
from fractions import Fraction

try:
    from metricker import Meter, MeterError
except ImportError:
    pass

from .dur_to_kern import duration_float_to_recip


class BeatBoundaryMeter:
    """Fallback meter for time signatures not supported by metricker (e.g., 5/4, 7/8).

    Splits durations at every beat boundary, where beat = 4/denom quarter notes.
    """

    def __init__(self, ts_str: str):
        parts = ts_str.split("/")
        numer = int(parts[0])
        denom = int(parts[1])
        self._beat_dur = Fraction(4, denom)
        self._numer = numer
        self._denom = denom

    def split_at_metric_strong_points(self, items, min_split_dur=None, force_split=False):
        result = []
        for item in items:
            result.extend(
                self._split_item_at_beats(item, min_split_dur, force_split)
            )
        return result

    def _split_item_at_beats(self, item, min_split_dur, force_split):
        onset = Fraction(item.onset).limit_denominator(1000)
        release = Fraction(item.release).limit_denominator(1000)
        beat = self._beat_dur

        # Find beat boundaries strictly inside (onset, release)
        first_beat_after_onset = (onset // beat + 1) * beat
        boundaries = []
        b = first_beat_after_onset
        while b < release:
            boundaries.append(b)
            b += beat

        if not boundaries:
            return [item]

        # Filter out boundaries that create fragments shorter than min_split_dur
        if min_split_dur is not None:
            min_dur = Fraction(min_split_dur).limit_denominator(1000)
            filtered = []
            for bd in boundaries:
                # Check both sides: left fragment and right fragment
                left_start = filtered[-1] if filtered else onset
                if bd - left_start >= min_dur and release - bd >= min_dur:
                    filtered.append(bd)

            if force_split and not filtered and boundaries:
                filtered = [boundaries[0]]

            boundaries = filtered

        if not boundaries:
            return [item]

        # Split at boundaries
        splits = []
        points = [onset] + boundaries + [release]
        for i in range(len(points) - 1):
            new_item = copy.copy(item)
            new_item.onset = float(points[i])
            new_item.release = float(points[i + 1])
            splits.append(new_item)
        return splits

    def split_odd_duration(self, item, min_split_dur=None):
        dur = float(item.release - item.onset)
        recip = duration_float_to_recip(dur)
        if not recip.startswith("q"):
            return [item]
        return self.split_at_metric_strong_points(
            [item], min_split_dur, force_split=True
        )


def make_meter(ts_str: str):
    """Create a Meter, falling back to BeatBoundaryMeter for unsupported time signatures."""
    try:
        return Meter(ts_str)
    except MeterError:
        return BeatBoundaryMeter(ts_str)
