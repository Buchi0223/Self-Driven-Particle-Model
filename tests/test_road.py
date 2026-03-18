"""
road モジュールのテスト
"""

import pytest
from src.road import Road


class TestRoad:

    def test_default_values(self):
        r = Road()
        assert r.y_right == 0.0
        assert r.y_left == 12.0
        assert r.road_length == 500.0

    def test_width(self):
        r = Road()
        assert r.width == 12.0

    def test_center(self):
        r = Road()
        assert r.center == 6.0

    def test_custom_road(self):
        r = Road(y_right=2.0, y_left=8.0, road_length=1000.0)
        assert r.width == 6.0
        assert r.center == 5.0

    def test_frozen(self):
        r = Road()
        with pytest.raises(AttributeError):
            r.y_left = 99.0
