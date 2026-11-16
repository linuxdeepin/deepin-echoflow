#!/usr/bin/env bash
# dev-setup.sh — install build dependencies for fast iterative
# CrispASR development. Per-platform: macOS uses Homebrew, Linux uses
# apt/dnf, Windows is documented but not auto-installed (use the
# matching dev-setup.ps1).
#
# Idempotent — safe to re-run; existing installs are skipped.

set -e

OS="$(uname -s)"

case "$OS" in
  Darwin)
    if ! command -v brew >/dev/null; then
      echo "ERROR: Homebrew required. Install from https://brew.sh"
      exit 1
    fi
    # libomp     — Apple clang needs explicit OpenMP for ggml multi-threaded matmul
    # ninja      — much faster than Unix Makefiles for incremental builds
    # ccache     — caches compiler output across rebuilds (huge win after first build)
    # cmake      — required (no-op if already installed)
    # NOTE: mold doesn't ship a Mach-O linker on macOS — skipped.
    brew install --quiet libomp ninja ccache cmake
    echo
    echo "✓ macOS dev deps installed."
    echo "  Build with: scripts/dev-build.sh"
    ;;

  Linux)
    if command -v apt-get >/dev/null; then
      sudo apt-get update -qq
      # mold — much faster linker than GNU ld; Linux-only on this script
      # libomp-dev — explicit OpenMP runtime headers
      sudo apt-get install -y -qq build-essential cmake ninja-build ccache mold libomp-dev
    elif command -v dnf >/dev/null; then
      sudo dnf install -y cmake ninja-build ccache mold libomp-devel gcc-c++
    elif command -v pacman >/dev/null; then
      sudo pacman -S --noconfirm cmake ninja ccache mold openmp gcc
    else
      echo "ERROR: unsupported Linux distro — install cmake/ninja/ccache/mold/libomp manually."
      exit 1
    fi
    echo
    echo "✓ Linux dev deps installed."
    echo "  Build with: scripts/dev-build.sh"
    ;;

  MINGW*|MSYS*|CYGWIN*)
    echo "Detected Windows shell. For native Windows builds:"
    echo "  - Install Visual Studio 2022 with C++ workload"
    echo "  - Install ninja:  choco install ninja  (or scoop install ninja)"
    echo "  - Install ccache: choco install ccache"
    echo "  - Run scripts/dev-build.ps1 from PowerShell"
    exit 0
    ;;

  *)
    echo "ERROR: unsupported platform: $OS"
    echo "Install cmake + ninja + ccache + (libomp-dev or libomp) by hand and run dev-build.sh."
    exit 1
    ;;
esac
