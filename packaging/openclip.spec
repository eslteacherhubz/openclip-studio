# PyInstaller spec for the Eterna OpenClip Studio desktop app.
# Build (from repo root):  pyinstaller packaging/openclip.spec
# Output: dist/OpenClipStudio/ (one-folder; zip it for distribution).
# Contains OpenClipStudio.exe (GUI) and OpenClipCLI.exe (headless CLI).
# Spec mechanics validated on Linux in the dev sandbox; produced for
# windows-latest in CI.

import os

block_cipher = None
spec_dir = SPECPATH  # directory containing this spec file
src_dir = os.path.normpath(os.path.join(spec_dir, "..", "src"))

common = dict(
    pathex=[src_dir],
    binaries=[],
    datas=[],
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
