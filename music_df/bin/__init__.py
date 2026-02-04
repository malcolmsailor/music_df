"""
Platform-specific binary management for music_df.

This module provides access to compiled binaries (like totable) that are
bundled with the package for different platforms.
"""

import platform
import sys
from pathlib import Path

# Map (system, machine) to binary suffix
PLATFORM_MAP = {
    ("Darwin", "arm64"): "macos-arm64",
    ("Darwin", "x86_64"): "macos-x86_64",
    ("Linux", "x86_64"): "linux-x86_64",
    ("Linux", "aarch64"): "linux-arm64",
}


def get_platform_suffix() -> str:
    """Get the platform suffix for the current system."""
    system = platform.system()
    machine = platform.machine()
    key = (system, machine)

    if key not in PLATFORM_MAP:
        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            f"Supported platforms: {list(PLATFORM_MAP.values())}"
        )

    return PLATFORM_MAP[key]


def get_binary_path(name: str) -> Path:
    """
    Get the path to a platform-specific binary.

    Args:
        name: The base name of the binary (e.g., "totable")

    Returns:
        Path to the binary for the current platform

    Raises:
        RuntimeError: If the platform is unsupported or binary is missing
    """
    suffix = get_platform_suffix()
    binary_name = f"{name}-{suffix}"

    bin_dir = Path(__file__).parent
    binary_path = bin_dir / binary_name

    if not binary_path.exists():
        raise RuntimeError(
            f"Binary not found: {binary_path}. "
            f"The {name} binary for your platform ({suffix}) may not be included. "
            "See the music_df documentation for building from source."
        )

    return binary_path


def get_totable_path() -> Path:
    """Get the path to the totable binary for the current platform."""
    return get_binary_path("totable")
