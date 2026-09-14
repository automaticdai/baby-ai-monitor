import numpy as np

from babymon.config import ZoneConfig
from babymon.rules.zones import Zone

SQUARE = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)]


def test_point_inside_polygon():
    assert Zone(name="crib", polygon=SQUARE).contains((0.5, 0.5))


def test_point_outside_polygon():
    assert not Zone(name="crib", polygon=SQUARE).contains((0.05, 0.5))


def test_point_outside_on_the_other_axis():
    assert not Zone(name="crib", polygon=SQUARE).contains((0.5, 0.95))


def test_concave_polygon_excludes_the_notch():
    # An L-shape occupying the left column and bottom row.
    l_shape = [(0.0, 0.0), (0.4, 0.0), (0.4, 0.6), (1.0, 0.6), (1.0, 1.0), (0.0, 1.0)]
    zone = Zone(name="l", polygon=l_shape)
    assert zone.contains((0.2, 0.2))
    assert not zone.contains((0.8, 0.2))


def test_mask_marks_interior_pixels():
    mask = Zone(name="crib", polygon=SQUARE).mask(100, 100)
    assert mask.shape == (100, 100)
    assert mask[50, 50] == 255
    assert mask[1, 1] == 0
    assert mask.dtype == np.uint8


def test_from_config():
    zone = Zone.from_config(ZoneConfig(name="crib", polygon=SQUARE))
    assert zone.name == "crib"
    assert zone.contains((0.5, 0.5))
