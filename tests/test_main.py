"""CLI-level tests for babymon.__main__.

Kept minimal and process-free: main() is called directly with an argv list
so nothing spawns a subprocess, and every source/store that could block or
write to disk is patched to raise if it's ever constructed - so a test that
would otherwise hang (e.g. an infinite webcam retry loop) fails fast and
loudly instead.
"""

from babymon import __main__ as cli


def test_replay_without_a_file_source_fails_fast(monkeypatch, capsys):
    def _must_not_construct(name):
        def _raise(*args, **kwargs):
            raise AssertionError(f"{name} must not be constructed")
        return _raise

    # No --source and no config means cfg.source.kind defaults to "webcam",
    # which is an infinite generator. If the fail-fast guard is ever removed
    # or misplaced, these patches turn the resulting hang into an immediate,
    # obvious assertion failure instead.
    monkeypatch.setattr(cli, "WebcamSource", _must_not_construct("WebcamSource"))
    monkeypatch.setattr(cli, "SqliteStore", _must_not_construct("SqliteStore"))
    monkeypatch.setattr(
        cli, "YoloPersonModel", _must_not_construct("YoloPersonModel")
    )

    rc = cli.main(["--replay"])

    assert rc != 0
    err = capsys.readouterr().err
    assert "--source" in err
