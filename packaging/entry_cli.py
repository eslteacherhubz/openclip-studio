"""PyInstaller entry for the headless CLI (openclip console script)."""

from __future__ import annotations

import sys

from openclip.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
