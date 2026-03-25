"""Player class wrapping pyfluidsynth."""

from __future__ import annotations

import os
import platform
import tempfile
import threading
from typing import TYPE_CHECKING

try:
    import fluidsynth
except ImportError:
    raise ImportError(
        "pyfluidsynth is required for MIDI playback. "
        "Install with: pip install 'music_df[playback]'\n"
        "You also need the FluidSynth C library: brew install fluid-synth"
    ) from None

from music_df.midi_parser import df_to_midi
from music_df.playback._soundfont import find_soundfont

if TYPE_CHECKING:
    import pandas as pd

# Must match df_to_symusic_score default
_TICKS_PER_QUARTER = 480


def _default_audio_driver() -> str:
    system = platform.system()
    if system == "Darwin":
        return "coreaudio"
    if system == "Windows":
        return "dsound"
    return "pulseaudio"


class Player:
    """Non-blocking MIDI playback via FluidSynth.

    Playback runs in FluidSynth's internal C thread, so ``play()``
    returns immediately.
    """

    def __init__(
        self,
        soundfont: str | None = None,
        gain: float = 0.5,
        audio_driver: str | None = None,
    ) -> None:
        self._soundfont_path = soundfont
        self._resolved_soundfont = soundfont or find_soundfont()

        self._synth = fluidsynth.Synth(gain=gain)
        driver = audio_driver or _default_audio_driver()
        self._synth.start(driver=driver)
        self._sfid = self._synth.sfload(self._resolved_soundfont)

        self._lock = threading.Lock()
        self._fluid_player: object | None = None
        self._tmp_path: str | None = None
        self._cleanup_thread: threading.Thread | None = None

    def play(self, df: pd.DataFrame, start: float = 0.0) -> None:
        """Begin playback of *df*.

        Stops any current playback first.

        Args:
            df: A music_df DataFrame.
            start: Start offset in quarter notes.
        """
        with self._lock:
            self._stop_unlocked()

            tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False)
            tmp.close()
            self._tmp_path = tmp.name
            df_to_midi(df, self._tmp_path)

            player = fluidsynth.new_fluid_player(self._synth.synth)
            fluidsynth.fluid_player_add(player, self._tmp_path.encode())
            fluidsynth.fluid_player_play(player)

            if start > 0:
                ticks = int(start * _TICKS_PER_QUARTER)
                fluidsynth.fluid_player_seek(player, ticks)

            self._fluid_player = player
            self._start_cleanup_thread()

    def stop(self) -> None:
        """Stop playback immediately."""
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self._fluid_player is not None:
            fluidsynth.fluid_player_stop(self._fluid_player)
            fluidsynth.fluid_player_join(self._fluid_player)
            fluidsynth.delete_fluid_player(self._fluid_player)
            self._fluid_player = None
        # Silence any notes still sounding in the synth
        for chan in range(16):
            fluidsynth.fluid_synth_all_sounds_off(self._synth.synth, chan)
        self._remove_tmp_file()

    def is_playing(self) -> bool:
        """Return True if playback is active."""
        with self._lock:
            if self._fluid_player is None:
                return False
            return (
                fluidsynth.fluid_player_get_status(self._fluid_player)
                == fluidsynth.FLUID_PLAYER_PLAYING
            )

    def cleanup(self) -> None:
        """Release all FluidSynth resources."""
        with self._lock:
            self._stop_unlocked()
        self._synth.delete()

    def _remove_tmp_file(self) -> None:
        if self._tmp_path is not None and os.path.exists(self._tmp_path):
            os.unlink(self._tmp_path)
            self._tmp_path = None

    def _start_cleanup_thread(self) -> None:
        def _wait_and_cleanup() -> None:
            player = self._fluid_player
            if player is not None:
                fluidsynth.fluid_player_join(player)
            with self._lock:
                # Only clean up if this is still the active player
                if self._fluid_player is player:
                    fluidsynth.delete_fluid_player(player)
                    self._fluid_player = None
                    self._remove_tmp_file()

        t = threading.Thread(target=_wait_and_cleanup, daemon=True)
        t.start()
        self._cleanup_thread = t

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass
