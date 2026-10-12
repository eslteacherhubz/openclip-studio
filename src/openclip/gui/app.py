"""GUI entry point (optional ``gui`` extra)."""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print(
            "PySide6 is not installed. Run: pip install eterna-openclip-studio[gui]",
            file=sys.stderr,
        )
        return 2

    # Headless/CI-safe: allow forcing the offscreen platform via env.
    if os.environ.get("OPENCLIP_GUI_PLATFORM"):
        os.environ["QT_QPA_PLATFORM"] = os.environ["OPENCLIP_GUI_PLATFORM"]

    from openclip.gui.main_window import MainWindow

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Eterna OpenClip Studio")
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
