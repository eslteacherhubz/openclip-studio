def test_smoke() -> None:
    from openclip import __version__

    assert __version__.startswith("0.")


def test_cli_version(capsys: object) -> None:
    from openclip.cli import main

    assert main(["--version"]) == 0
