def test_smoke() -> None:
    from openclip import __version__

    assert __version__.startswith("0.")


def test_cli_version() -> None:
    import pytest

    from openclip.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
