"""PyInstaller entry for the desktop app (keeps the GUI import explicit)."""

from __future__ import annotations

import sys

from openclip.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
