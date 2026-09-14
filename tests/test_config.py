import pytest
from pydantic import ValidationError

from babymon.config import Config, ZoneConfig


def test_defaults_load_without_a_file():
    cfg = Config()
    assert cfg.source.kind == "webcam"
    assert cfg.rules.lost_track_after_s == 60.0
    assert cfg.storage.retention_days == 7


def test_load_reads_yaml(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(
        "source:\n"
        "  kind: file\n"
        "  path: clip.mp4\n"
        "zones:\n"
        "  - name: crib\n"
        "    polygon: [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]]\n"
        "rules:\n"
        "  lost_track_after_s: 30\n"
    )
    cfg = Config.load(path)
    assert cfg.source.kind == "file"
    assert cfg.source.path == "clip.mp4"
    assert cfg.zones[0].name == "crib"
    assert cfg.rules.lost_track_after_s == 30.0


def test_polygon_needs_at_least_three_points():
    with pytest.raises(ValidationError):
        ZoneConfig(name="crib", polygon=[(0.1, 0.1), (0.9, 0.9)])


def test_polygon_coordinates_must_be_normalised():
    with pytest.raises(ValidationError):
        ZoneConfig(name="crib", polygon=[(0, 0), (2.0, 0), (1, 1)])


def test_file_source_requires_a_path():
    with pytest.raises(ValidationError):
        Config(source={"kind": "file"})


def test_source_override_is_applied_before_validation(tmp_path):
    """A config declaring a file source without a path is valid once --source
    supplies one. Validating first would reject a config the override was
    about to make valid."""
    path = tmp_path / "c.yaml"
    path.write_text("source:\n  kind: file\n")
    cfg = Config.load(path, source="clip.avi")
    assert cfg.source.kind == "file"
    assert cfg.source.path == "clip.avi"


def test_source_override_replaces_a_configured_path(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source:\n  kind: file\n  path: old.avi\n")
    cfg = Config.load(path, source="new.avi")
    assert cfg.source.path == "new.avi"


def test_source_override_preserves_other_source_fields(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source:\n  kind: file\n  source_id: nursery\n  realtime: false\n")
    cfg = Config.load(path, source="clip.avi")
    assert cfg.source.source_id == "nursery"
    assert cfg.source.realtime is False
    assert cfg.source.path == "clip.avi"


def test_source_override_without_a_config_file():
    cfg = Config.load(None, source="clip.avi")
    assert cfg.source.kind == "file"
    assert cfg.source.path == "clip.avi"


def test_load_without_a_path_or_override_gives_defaults():
    assert Config.load(None).source.kind == "webcam"


def test_a_file_config_with_no_path_and_no_override_still_fails(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("source:\n  kind: file\n")
    with pytest.raises(ValidationError):
        Config.load(path)
