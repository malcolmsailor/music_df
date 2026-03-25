"""Tests for music_df.playback."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Soundfont discovery
# ---------------------------------------------------------------------------


class TestFindSoundfont:
    def test_from_env_var(self, tmp_path: Path) -> None:
        sf2 = tmp_path / "test.sf2"
        sf2.touch()
        with patch.dict(os.environ, {"GM_SOUNDFONT": str(sf2)}):
            from music_df.playback._soundfont import find_soundfont

            assert find_soundfont() == str(sf2)

    def test_from_search_path(self, tmp_path: Path) -> None:
        sf2 = tmp_path / "GeneralUser.sf2"
        sf2.touch()
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "music_df.playback._soundfont._SOUNDFONT_SEARCH_PATHS",
                [tmp_path],
            ),
        ):
            os.environ.pop("GM_SOUNDFONT", None)
            from music_df.playback._soundfont import find_soundfont

            assert find_soundfont() == str(sf2)

    def test_not_found_raises(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "music_df.playback._soundfont._SOUNDFONT_SEARCH_PATHS",
                [],
            ),
        ):
            os.environ.pop("GM_SOUNDFONT", None)
            from music_df.playback._soundfont import find_soundfont

            with pytest.raises(FileNotFoundError, match="No SoundFont"):
                find_soundfont()


# ---------------------------------------------------------------------------
# Player (mocked fluidsynth)
# ---------------------------------------------------------------------------


def _simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "type": ["note", "note"],
            "onset": [0.0, 1.0],
            "release": [1.0, 2.0],
            "pitch": [60, 64],
            "velocity": [64, 64],
            "track": [0, 0],
            "channel": [0, 0],
            "other": [None, None],
        }
    )


@pytest.fixture
def mock_fluidsynth():
    """Patch fluidsynth module inside _player with mocks."""
    mock_synth_cls = MagicMock()

    synth_instance = mock_synth_cls.return_value
    synth_instance.sfload.return_value = 1

    # fluid_player_stop signals the player to finish; fluid_player_join
    # blocks until that signal arrives.  We simulate this so the cleanup
    # thread (which calls join without a prior stop) blocks until
    # _stop_unlocked calls fluid_player_stop, which opens the gate.
    join_gate = threading.Event()

    with (
        patch("music_df.playback._player.fluidsynth") as mock_mod,
        patch("music_df.playback._player.find_soundfont", return_value="/fake.sf2"),
        patch("music_df.playback._player.df_to_midi"),
    ):
        mock_mod.Synth = mock_synth_cls
        mock_mod.FLUID_PLAYER_PLAYING = 1
        fluid_player = MagicMock(name="fluid_player_handle")
        mock_mod.new_fluid_player.return_value = fluid_player
        mock_mod.fluid_player_get_status.return_value = 1
        mock_mod.fluid_player_join.side_effect = lambda p: join_gate.wait()
        mock_mod.fluid_player_stop.side_effect = lambda p: join_gate.set()
        yield {
            "module": mock_mod,
            "synth_cls": mock_synth_cls,
            "synth": synth_instance,
            "fluid_player": fluid_player,
            "join_gate": join_gate,
        }
        join_gate.set()  # unblock cleanup thread on teardown


class TestPlayer:
    def test_play_creates_temp_midi(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player, df_to_midi

        p = Player()
        p.play(_simple_df())

        # df_to_midi should have been called with a .mid temp path
        df_to_midi = p  # just to reference; the real check is the mock
        from music_df.playback._player import df_to_midi as mocked_fn

        assert mocked_fn.called
        call_args = mocked_fn.call_args
        assert call_args[0][1].endswith(".mid")

    def test_stop_calls_player_stop(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player

        mod = mock_fluidsynth["module"]
        p = Player()
        p.play(_simple_df())
        p.stop()

        fp = mock_fluidsynth["fluid_player"]
        mod.fluid_player_stop.assert_called_with(fp)
        mod.fluid_player_join.assert_called_with(fp)
        mod.delete_fluid_player.assert_called_with(fp)

    def test_stop_silences_all_channels(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player

        mod = mock_fluidsynth["module"]
        synth_handle = mock_fluidsynth["synth"].synth
        p = Player()
        p.play(_simple_df())
        mod.fluid_synth_all_sounds_off.reset_mock()
        p.stop()

        assert mod.fluid_synth_all_sounds_off.call_count == 16
        for chan in range(16):
            mod.fluid_synth_all_sounds_off.assert_any_call(synth_handle, chan)

    def test_play_with_start_offset(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player, _TICKS_PER_QUARTER

        mod = mock_fluidsynth["module"]
        fp = mock_fluidsynth["fluid_player"]
        p = Player()
        p.play(_simple_df(), start=4.0)

        expected_ticks = int(4.0 * _TICKS_PER_QUARTER)
        mod.fluid_player_seek.assert_called_with(fp, expected_ticks)

    def test_play_no_seek_at_zero(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player

        mod = mock_fluidsynth["module"]
        p = Player()
        p.play(_simple_df(), start=0.0)

        mod.fluid_player_seek.assert_not_called()

    def test_is_playing(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player

        p = Player()
        assert not p.is_playing()

        p.play(_simple_df())
        assert p.is_playing()

    def test_cleanup_deletes_synth(self, mock_fluidsynth: dict) -> None:
        from music_df.playback._player import Player

        p = Player()
        p.play(_simple_df())
        p.cleanup()

        mock_fluidsynth["synth_cls"].return_value.delete.assert_called()


# ---------------------------------------------------------------------------
# Integration test (requires real fluidsynth)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("MUSIC_DF_TEST_PLAYBACK") != "1",
    reason="Set MUSIC_DF_TEST_PLAYBACK=1 to run integration playback tests",
)
def test_real_playback() -> None:
    """Smoke test: play a short score, stop after 1 second."""
    import time

    from music_df.playback import play, stop

    play(_simple_df())
    time.sleep(1)
    stop()
