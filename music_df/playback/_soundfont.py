"""SoundFont discovery."""

from __future__ import annotations

import os
from pathlib import Path

_ENV_VAR = "GM_SOUNDFONT"

_SOUNDFONT_SEARCH_PATHS = [
    # macOS Homebrew
    Path("/opt/homebrew/share/soundfonts"),
    Path("/opt/homebrew/share/fluidsynth"),
    Path("/usr/local/share/soundfonts"),
    Path("/usr/local/share/fluidsynth"),
    # Linux
    Path("/usr/share/soundfonts"),
    Path("/usr/share/sounds/sf2"),
    Path("/usr/share/sound/sf2"),
]


def find_soundfont() -> str:
    """Locate a SoundFont (.sf2) file.

    Search order:
        1. ``GM_SOUNDFONT`` environment variable
        2. Common system directories

    Returns:
        Absolute path to a ``.sf2`` file.

    Raises:
        FileNotFoundError: If no SoundFont is found.
    """
    env_path = os.environ.get(_ENV_VAR)
    if env_path and Path(env_path).is_file():
        return env_path

    for search_dir in _SOUNDFONT_SEARCH_PATHS:
        if search_dir.is_dir():
            sf2_files = sorted(search_dir.glob("*.sf2"))
            if sf2_files:
                return str(sf2_files[0])

    raise FileNotFoundError(
        "No SoundFont (.sf2) file found. Either:\n"
        f"  - Set the {_ENV_VAR} environment variable to a .sf2 file path\n"
        "  - Install FluidSynth with a bundled SoundFont: "
        "brew install fluid-synth\n"
        "  - Download a SoundFont such as FluidR3_GM.sf2 or "
        "MuseScore_General.sf2"
    )
