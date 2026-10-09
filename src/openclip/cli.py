"""CLI entry point. Subcommands grow with the stages (see docs/ROADMAP.md)."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="openclip",
        description="Eterna OpenClip Studio — bilingual teaching-video editor",
    )
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from openclip import __version__

        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
