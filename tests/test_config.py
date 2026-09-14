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
