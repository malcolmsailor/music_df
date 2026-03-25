"""MIDI playback for music_df DataFrames via FluidSynth.

Requires the ``pyfluidsynth`` Python package and the FluidSynth C library::

    pip install 'music_df[playback]'
    brew install fluid-synth   # macOS
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_df.playback._player import Player

if TYPE_CHECKING:
    import pandas as pd

_default_player: Player | None = None


def play(
    df: pd.DataFrame,
    soundfont: str | None = None,
    start: float = 0.0,
    gain: float = 0.5,
) -> Player:
    """Play a music_df DataFrame.

    Creates (or reuses) a module-level Player and begins non-blocking
    playback. Call :func:`stop` to halt playback.

    Args:
        df: A music_df DataFrame.
        soundfont: Path to a ``.sf2`` SoundFont file. Auto-detected if None.
        start: Start offset in quarter notes.
        gain: Master volume (0.0--1.0).

    Returns:
        The active :class:`Player` instance.
    """
    global _default_player
    if _default_player is None or _default_player._soundfont_path != soundfont:
        if _default_player is not None:
            _default_player.cleanup()
        _default_player = Player(soundfont=soundfont, gain=gain)
    _default_player.play(df, start=start)
    return _default_player


def stop() -> None:
    """Stop the default player, if playing."""
    if _default_player is not None:
        _default_player.stop()
