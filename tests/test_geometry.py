"""
geometry モジュールのテスト

車両間の幾何量計算が正しいことを検証する。
"""

import pytest
from src.geometry import delta_x, delta_y, W_bar, sy_gap, sx_gap


class TestDeltaX:
    """縦方向距離のテスト"""

    def test_leader_ahead(self):
        assert delta_x(xi=0.0, xl=50.0) == 50.0

    def test_same_position(self):
        assert delta_x(xi=10.0, xl=10.0) == 0.0

    def test_leader_behind(self):
        """リーダーが後方の場合は負"""
        assert delta_x(xi=50.0, xl=30.0) == -20.0


class TestDeltaY:
    """横方向変位のテスト"""

    def test_leader_left(self):
        """リーダーが左側 → 正"""
        assert delta_y(yi=3.0, yl=5.0) == 2.0

    def test_leader_right(self):
        """リーダーが右側 → 負"""
        assert delta_y(yi=5.0, yl=3.0) == -2.0

    def test_same_lane(self):
        assert delta_y(yi=3.0, yl=3.0) == 0.0


class TestWBar:
    """平均車幅のテスト"""

    def test_same_width(self):
        assert W_bar(1.7, 1.7) == 1.7

    def test_different_widths(self):
        """Car (1.7) と Motorcycle (0.6)"""
        assert W_bar(1.7, 0.6) == pytest.approx(1.15)

    def test_symmetry(self):
        assert W_bar(1.7, 0.6) == W_bar(0.6, 1.7)


class TestSyGap:
    """横方向ギャップのテスト — 式(8) s^y_{il}"""

    def test_inline_overlap(self):
        """インライン (Δy=0) → sy = -W̄ < 0 (完全オーバーラップ)"""
        w_bar = 1.7
        assert sy_gap(dy=0.0, w_bar=w_bar) == pytest.approx(-1.7)

    def test_partial_overlap(self):
        """部分オーバーラップ: |Δy| < W̄"""
        w_bar = 1.7  # Car 同士
        dy = 1.0     # 1m 横にずれている
        sy = sy_gap(dy, w_bar)
        assert sy < 0.0  # まだオーバーラップ
        assert sy == pytest.approx(1.0 - 1.7)

    def test_just_touching(self):
        """ちょうど接触: |Δy| = W̄ → sy = 0"""
        w_bar = 1.7
        assert sy_gap(dy=1.7, w_bar=w_bar) == pytest.approx(0.0)

    def test_clear_gap(self):
        """クリアランスあり: |Δy| > W̄ → sy > 0"""
        w_bar = 1.7
        sy = sy_gap(dy=3.0, w_bar=w_bar)
        assert sy > 0.0
        assert sy == pytest.approx(3.0 - 1.7)

    def test_symmetry_sign(self):
        """左右対称: sy_gap(+dy) == sy_gap(-dy)"""
        w_bar = 1.15
        assert sy_gap(dy=2.0, w_bar=w_bar) == sy_gap(dy=-2.0, w_bar=w_bar)

    def test_motorcycle_car(self):
        """Motorcycle (0.6) と Car (1.7): W̄=1.15"""
        w_bar = W_bar(0.6, 1.7)
        # 2m 離れている
        assert sy_gap(dy=2.0, w_bar=w_bar) == pytest.approx(2.0 - 1.15)


class TestSxGap:
    """縦方向ギャップのテスト"""

    def test_normal_following(self):
        """通常の追従: リーダーが前方"""
        # i at x=0, leader at x=20, leader length=4.2
        gap = sx_gap(xi=0.0, xl=20.0, Ll=4.2)
        assert gap == pytest.approx(15.8)

    def test_close_following(self):
        """密接追従"""
        gap = sx_gap(xi=0.0, xl=6.0, Ll=4.2)
        assert gap == pytest.approx(1.8)

    def test_overlap(self):
        """縦方向オーバーラップ (衝突状態) → 負"""
        gap = sx_gap(xi=0.0, xl=3.0, Ll=4.2)
        assert gap < 0.0
        assert gap == pytest.approx(-1.2)

    def test_just_touching(self):
        """ちょうど接触: gap = 0"""
        gap = sx_gap(xi=0.0, xl=4.2, Ll=4.2)
        assert gap == pytest.approx(0.0)
