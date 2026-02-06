import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from music_df.plot_piano_rolls.plot import plot_piano_roll


@pytest.fixture
def simple_note_df():
    return pd.DataFrame(
        {
            "onset": [0.0, 1.0, 2.0, 3.0],
            "release": [1.0, 2.0, 3.0, 4.0],
            "pitch": [60.0, 62.0, 64.0, 65.0],
            "type": ["note"] * 4,
        }
    )


class TestAnnotations:
    def test_no_annotations_unchanged(self, simple_note_df):
        """Passing annotations=None should not change y-limits."""
        ax = plot_piano_roll(simple_note_df, annotations=None)
        low, hi = ax.get_ylim()
        assert low == 60.0
        assert hi == 66.0
        plt.close("all")

    def test_annotations_expands_ylim(self, simple_note_df):
        annotations = [(0.0, "C"), (2.0, "F")]
        ax = plot_piano_roll(simple_note_df, annotations=annotations)
        low, hi = ax.get_ylim()
        assert low == 60.0 - 4  # annotation_height = 4
        assert hi == 66.0
        plt.close("all")

    def test_annotations_text_objects(self, simple_note_df):
        annotations = [(0.0, "Cmaj"), (2.0, "Fmin")]
        ax = plot_piano_roll(simple_note_df, annotations=annotations)
        texts = [t for t in ax.texts if t.get_text() in ("Cmaj", "Fmin")]
        assert len(texts) == 2
        labels = {t.get_text() for t in texts}
        assert labels == {"Cmaj", "Fmin"}
        plt.close("all")

    def test_annotations_text_positions(self, simple_note_df):
        annotations = [(1.5, "test")]
        ax = plot_piano_roll(simple_note_df, annotations=annotations)
        text_obj = [t for t in ax.texts if t.get_text() == "test"][0]
        x, y = text_obj.get_position()
        assert x == 1.5
        # y should be low - 0.5 = 60.0 - 0.5
        assert y == 59.5
        plt.close("all")

    def test_yticks_filtered_to_note_range(self, simple_note_df):
        annotations = [(0.0, "C")]
        ax = plot_piano_roll(simple_note_df, annotations=annotations)
        yticks = ax.get_yticks()
        assert all(t >= 60.0 for t in yticks)
        plt.close("all")

    def test_separator_line_drawn(self, simple_note_df):
        annotations = [(0.0, "C")]
        ax = plot_piano_roll(simple_note_df, annotations=annotations)
        # There should be a gray separator line at y=low (60.0)
        separator_lines = [
            line for line in ax.get_lines()
            if list(line.get_ydata()) == [60.0, 60.0]
            and line.get_color() == "gray"
        ]
        assert len(separator_lines) == 1
        plt.close("all")

    def test_annotations_with_barlines(self):
        df = pd.DataFrame(
            {
                "onset": [0.0, 1.0, 2.0, 3.0, 0.0, 2.0],
                "release": [1.0, 2.0, 3.0, 4.0, float("nan"), float("nan")],
                "pitch": [60.0, 62.0, 64.0, 65.0, float("nan"), float("nan")],
                "type": ["note", "note", "note", "note", "bar", "bar"],
            }
        )
        annotations = [(0.0, "I"), (2.0, "V")]
        ax = plot_piano_roll(df, annotations=annotations, barlines=True)
        low, hi = ax.get_ylim()
        assert low == 60.0 - 4
        plt.close("all")

    def test_empty_annotations_list(self, simple_note_df):
        """An empty annotations list should still expand y-limits."""
        ax = plot_piano_roll(simple_note_df, annotations=[])
        low, hi = ax.get_ylim()
        assert low == 60.0 - 4
        plt.close("all")


class TestRegions:
    def test_full_height_region(self, simple_note_df):
        """A 2-tuple region should span the full pitch range."""
        ax = plot_piano_roll(simple_note_df, regions=[(1.0, 3.0)])
        region_patches = [
            p for p in ax.patches
            if getattr(p, "get_alpha", lambda: None)() == 0.3
        ]
        assert len(region_patches) == 1
        verts = region_patches[0].get_xy()
        # low=60, hi=66
        assert verts[0][1] == 60.0
        assert verts[2][1] == 66.0
        plt.close("all")

    def test_bounded_region(self, simple_note_df):
        """A 4-tuple region should have the specified pitch bounds."""
        ax = plot_piano_roll(simple_note_df, regions=[(1.0, 3.0, 62.0, 65.0)])
        region_patches = [
            p for p in ax.patches
            if getattr(p, "get_alpha", lambda: None)() == 0.3
        ]
        assert len(region_patches) == 1
        verts = region_patches[0].get_xy()
        assert verts[0] == pytest.approx([1.0, 62.0])
        assert verts[1] == pytest.approx([3.0, 62.0])
        assert verts[2] == pytest.approx([3.0, 65.0])
        assert verts[3] == pytest.approx([1.0, 65.0])
        plt.close("all")

    def test_multiple_regions(self, simple_note_df):
        """Multiple regions should all be rendered."""
        regions = [(0.0, 1.0), (2.0, 3.0), (1.0, 2.0, 61.0, 63.0)]
        ax = plot_piano_roll(simple_note_df, regions=regions)
        region_patches = [
            p for p in ax.patches
            if getattr(p, "get_alpha", lambda: None)() == 0.3
        ]
        assert len(region_patches) == 3
        plt.close("all")

    def test_regions_with_annotations(self, simple_note_df):
        """Regions and annotations should coexist without error."""
        ax = plot_piano_roll(
            simple_note_df,
            regions=[(0.5, 2.5)],
            annotations=[(0.0, "Cmaj"), (2.0, "Fmin")],
        )
        region_patches = [
            p for p in ax.patches
            if getattr(p, "get_alpha", lambda: None)() == 0.3
        ]
        assert len(region_patches) == 1
        texts = [t for t in ax.texts if t.get_text() in ("Cmaj", "Fmin")]
        assert len(texts) == 2
        plt.close("all")
