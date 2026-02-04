#!/bin/bash
# Build totable for the current platform
#
# Prerequisites:
#   - C++ compiler (g++ or clang++)
#
# Usage (from repository root):
#   ./scripts/build_totable.sh
#
# The script will detect the platform and output a binary with the appropriate
# suffix (e.g., totable-linux-x86_64, totable-macos-arm64) to music_df/bin/.
#
# For cross-compilation on macOS to x86_64:
#   ARCH=x86_64 ./scripts/build_totable.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HUMLIB_DIR="${SCRIPT_DIR}/humlib"
OUTPUT_DIR="${SCRIPT_DIR}/../music_df/bin"

# Detect platform
OS=$(uname -s)
ARCH="${ARCH:-$(uname -m)}"

case "$OS" in
    Darwin)
        PLATFORM="macos"
        CXX="${CXX:-clang++}"
        ;;
    Linux)
        PLATFORM="linux"
        CXX="${CXX:-g++}"
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

case "$ARCH" in
    x86_64)
        ARCH_SUFFIX="x86_64"
        ;;
    arm64|aarch64)
        ARCH_SUFFIX="arm64"
        ;;
    *)
        echo "Unsupported architecture: $ARCH"
        exit 1
        ;;
esac

SUFFIX="${PLATFORM}-${ARCH_SUFFIX}"
OUTPUT_NAME="totable-${SUFFIX}"

echo "Building for: ${SUFFIX}"
echo "Output: ${OUTPUT_DIR}/${OUTPUT_NAME}"

# Check for required files
if [ ! -f "${HUMLIB_DIR}/humlib.h" ]; then
    echo "Error: humlib.h not found in ${HUMLIB_DIR}"
    echo "Make sure the humlib sources are in scripts/humlib/"
    exit 1
fi

# Create output directory if needed
mkdir -p "$OUTPUT_DIR"

# Build command
CXXFLAGS="-O3 -std=c++11 -Wall"

# Add architecture flag for macOS cross-compilation
if [ "$OS" = "Darwin" ] && [ "$ARCH" != "$(uname -m)" ]; then
    CXXFLAGS="$CXXFLAGS -arch $ARCH"
fi

echo "Compiling with ${CXX}..."
$CXX $CXXFLAGS \
    -I"${HUMLIB_DIR}" \
    -o "${OUTPUT_DIR}/${OUTPUT_NAME}" \
    "${SCRIPT_DIR}/totable.cpp" \
    "${HUMLIB_DIR}/humlib.cpp" \
    "${HUMLIB_DIR}/pugixml.cpp"

chmod +x "${OUTPUT_DIR}/${OUTPUT_NAME}"

echo ""
echo "Success! Built ${OUTPUT_DIR}/${OUTPUT_NAME}"
file "${OUTPUT_DIR}/${OUTPUT_NAME}"
