# PyInstaller spec for the Eterna OpenClip Studio desktop app.
# Build (from repo root):  pyinstaller packaging/openclip.spec
# Output: dist/OpenClipStudio/ (one-folder; zip it for distribution).
# Contains OpenClipStudio.exe (GUI) and OpenClipCLI.exe (headless CLI).
# Spec mechanics validated on Linux in the dev sandbox; produced for
# windows-latest in CI.

import os

from PyInstaller.utils.hooks import collect_submodules

block_cipher = None
spec_dir = SPECPATH  # directory containing this spec file
src_dir = os.path.normpath(os.path.join(spec_dir, "..", "src"))


def _optional_datas() -> list[tuple[str, str]]:
    """Package data files required by optional features at runtime.

    - faster-whisper: silero VAD ONNX model loaded from its package assets.
    - opencv: Haar cascade XMLs loaded via cv2.data.haarcascades.
    Without these the frozen app fails on first transcription/gaze use.
    """
    datas: list[tuple[str, str]] = []
    try:
        import faster_whisper

        fw_assets = os.path.join(os.path.dirname(faster_whisper.__file__), "assets")
        if os.path.isdir(fw_assets):
            datas.append((fw_assets, os.path.join("faster_whisper", "assets")))
    except Exception:
        pass
    try:
        import cv2

        cv2_data = os.path.join(os.path.dirname(cv2.__file__), "data")
        if os.path.isdir(cv2_data):
            datas.append((cv2_data, os.path.join("cv2", "data")))
    except Exception:
        pass
    return datas


def _optional_modules(package: str) -> list[str]:
    """Collect submodules of an optional dependency if it is installed.

    The packaging environment installs [gui,asr,gaze] so whisper + gaze ship
    inside the bundle; the guard keeps the spec buildable without them.
    """
    try:
        return collect_submodules(package)
    except Exception:
        return []


hiddenimports = [
    *collect_submodules("openclip"),
    # Optional features that must ship in the frozen app. The packaging smoke
    # tests verify their presence — never claim a feature as bundled without
    # passing those.
    *_optional_modules("faster_whisper"),
    *_optional_modules("ctranslate2"),
    *_optional_modules("tokenizers"),
    *_optional_modules("av"),
    *_optional_modules("huggingface_hub"),
    "cv2",
    "numpy",
]

common = dict(
    pathex=[src_dir],
    binaries=[],
    datas=_optional_datas(),
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "pandas", "IPython"],
    cipher=block_cipher,
    noarchive=False,
)

gui = Analysis([os.path.join(spec_dir, "entry_gui.py")], **common)
cli = Analysis([os.path.join(spec_dir, "entry_cli.py")], **common)

gui_pyz = PYZ(gui.pure, gui.zipped_data, cipher=block_cipher)
cli_pyz = PYZ(cli.pure, cli.zipped_data, cipher=block_cipher)

gui_exe = EXE(
    gui_pyz,
    gui.scripts,
    [],
    exclude_binaries=True,
    name="OpenClipStudio",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # GUI app; engine errors surface via dialogs
)

cli_exe = EXE(
    cli_pyz,
    cli.scripts,
    [],
    exclude_binaries=True,
    name="OpenClipCLI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,  # headless CLI: same engine, for automation
)

coll = COLLECT(
    gui_exe,
    cli_exe,
    gui.binaries,
    gui.zipfiles,
    gui.datas,
    strip=False,
    upx=False,
    name="OpenClipStudio",
)
