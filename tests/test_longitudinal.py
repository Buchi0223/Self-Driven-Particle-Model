"""
longitudinal モジュールのテスト

MTM 縦方向ダイナミクスの各関数を検証する。
"""

import math
import pytest

from src.longitudinal import (
    alpha,
    f_interaction_single,
    find_most_interacting_leader,
    f_boundary_longitudinal,
    longitudinal_acceleration,
)
from src.cf_models import cf_free, cf_interaction, idm_acceleration
from src.parameters import CAR, MOTORCYCLE, DEFAULT_MTM_PARAMS
from src.vehicle import Vehicle
from src.road import Road


# ---------------------------------------------------------------------------
# α (横方向減衰因子) — 式(8)
# ---------------------------------------------------------------------------

class TestAlpha:

    def test_full_overlap(self):
        """完全オーバーラップ (Δy=0) → α = 1"""
        assert alpha(dy=0.0, w_bar=1.7, sy0=0.15) == 1.0

    def test_partial_overlap(self):
        """部分オーバーラップ (|Δy| < W̄) → α = 1"""
        assert alpha(dy=1.0, w_bar=1.7, sy0=0.15) == 1.0

    def test_just_touching(self):
        """接触 (|Δy| = W̄) → α = 1 (sy=0, exp(0)=1)"""
        assert alpha(dy=1.7, w_bar=1.7, sy0=0.15) == pytest.approx(1.0)

    def test_exponential_decay(self):
        """クリアランスあり → 指数減衰"""
        sy0 = 0.15
        dy = 2.0
        w_bar = 1.7
        sy = abs(dy) - w_bar  # 0.3
        expected = math.exp(-sy / sy0)
        assert alpha(dy, w_bar, sy0) == pytest.approx(expected)

    def test_large_gap_near_zero(self):
        """大きな横方向ギャップ → α ≈ 0"""
        a = alpha(dy=10.0, w_bar=1.7, sy0=0.15)
        assert a < 0.01

    def test_symmetry(self):
        """左右対称: α(+dy) == α(-dy)"""
        assert alpha(dy=2.5, w_bar=1.7, sy0=0.15) == \
               alpha(dy=-2.5, w_bar=1.7, sy0=0.15)


# ---------------------------------------------------------------------------
# f_interaction_single — 式(7)
# ---------------------------------------------------------------------------

class TestFInteractionSingle:

    def test_inline_reverts_to_cf(self):
        """インライン (Δy=0) → 純粋な CF interaction に一致"""
        xi, yi, vi = 0.0, 5.0, 15.0
        xl, yl, vl = 30.0, 5.0, 10.0  # same y → inline
        Wi, Wl, Ll = CAR.width, CAR.width, CAR.length

        f = f_interaction_single(
            xi, yi, vi, Wi, xl, yl, vl, Wl, Ll,
            CAR.v0, CAR.T, CAR.s0, CAR.a, CAR.b, CAR.delta, CAR.b_max,
            sy0=0.15,
        )
        # α = 1 (inline), so f = cf_interaction(gap, vi, vl)
        gap = xl - xi - Ll
        expected = cf_interaction(gap, vi, vl, CAR.v0, CAR.T, CAR.s0,
                                  CAR.a, CAR.b, CAR.delta, CAR.b_max)
        assert f == pytest.approx(expected)

    def test_offset_attenuates(self):
        """横方向オフセット → |f| が小さくなる"""
        xi, vi = 0.0, 15.0
        xl, vl = 30.0, 10.0
        Wi, Wl, Ll = CAR.width, CAR.width, CAR.length

        f_inline = f_interaction_single(
            xi, 5.0, vi, Wi, xl, 5.0, vl, Wl, Ll,
            CAR.v0, CAR.T, CAR.s0, CAR.a, CAR.b, CAR.delta, CAR.b_max, 0.15,
        )
        f_offset = f_interaction_single(
            xi, 5.0, vi, Wi, xl, 8.0, vl, Wl, Ll,
            CAR.v0, CAR.T, CAR.s0, CAR.a, CAR.b, CAR.delta, CAR.b_max, 0.15,
        )
        assert abs(f_offset) < abs(f_inline)


# ---------------------------------------------------------------------------
# find_most_interacting_leader — 式(2)
# ---------------------------------------------------------------------------

class TestFindMostInteractingLeader:

    def test_no_leaders(self):
        """前方に車両なし → (None, 0.0)"""
        car_i = Vehicle.create("Car", x=100.0, y=6.0, v=15.0)
        car_behind = Vehicle.create("Car", x=50.0, y=6.0, v=15.0, vehicle_id=1)
        leader, f = find_most_interacting_leader(car_i, [car_behind], sy0=0.15)
        assert leader is None
        assert f == 0.0

    def test_single_leader(self):
        """リーダー1台 → そのリーダーが選択される"""
        car_i = Vehicle.create("Car", x=0.0, y=6.0, v=15.0)
        leader_1 = Vehicle.create("Car", x=30.0, y=6.0, v=10.0, vehicle_id=1)
        leader, f = find_most_interacting_leader(car_i, [leader_1], sy0=0.15)
        assert leader is leader_1
        assert f < 0.0  # 減速方向

    def test_selects_most_interacting(self):
        """複数リーダー → |f| 最大のものを選択"""
        car_i = Vehicle.create("Car", x=0.0, y=6.0, v=15.0)
        # 近くの遅いリーダー (強い相互作用)
        leader_close = Vehicle.create("Car", x=15.0, y=6.0, v=5.0, vehicle_id=1)
        # 遠くのリーダー (弱い相互作用)
        leader_far = Vehicle.create("Car", x=100.0, y=6.0, v=10.0, vehicle_id=2)
        leader, f = find_most_interacting_leader(
            car_i, [leader_close, leader_far], sy0=0.15
        )
        assert leader is leader_close

    def test_ignores_self(self):
        """自分自身は除外"""
        car_i = Vehicle.create("Car", x=0.0, y=6.0, v=15.0, vehicle_id=0)
        leader, f = find_most_interacting_leader(car_i, [car_i], sy0=0.15)
        assert leader is None


# ---------------------------------------------------------------------------
# f_boundary_longitudinal — 式(9)
# ---------------------------------------------------------------------------

class TestFBoundaryLongitudinal:

    def test_center_of_road_negligible(self):
        """道路中央では境界力はほぼゼロ"""
        f = f_boundary_longitudinal(
            vi=15.0, yi=6.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        assert abs(f) < 0.01

    def test_near_left_boundary(self):
        """左境界接近 (yi=1, y_left=0) → 負の力 (減速)"""
        f = f_boundary_longitudinal(
            vi=15.0, yi=1.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        assert f < 0.0

    def test_near_right_boundary(self):
        """右境界接近 (yi=11, y_right=12) → 負の力 (減速)"""
        f = f_boundary_longitudinal(
            vi=15.0, yi=11.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        assert f < 0.0

    def test_zero_speed_no_force(self):
        """v=0 → 境界力なし (式9 の vi/v0 因子)"""
        f = f_boundary_longitudinal(
            vi=0.0, yi=0.5, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        assert f == 0.0

    def test_increases_near_boundary(self):
        """境界に近づくほど力が強くなる"""
        f_far = f_boundary_longitudinal(
            vi=15.0, yi=3.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        f_near = f_boundary_longitudinal(
            vi=15.0, yi=1.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb=3.0, sy0b=0.15,
        )
        assert abs(f_near) > abs(f_far)


# ---------------------------------------------------------------------------
# longitudinal_acceleration — 式(2) 全体
# ---------------------------------------------------------------------------

class TestLongitudinalAcceleration:

    def test_free_road(self):
        """空道路 → 自由加速度に近い"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        road = Road()
        acc = longitudinal_acceleration(car, [], road, DEFAULT_MTM_PARAMS)
        expected_free = cf_free(car.v, car.v0, car.a, car.delta)
        # 道路中央なので境界力はほぼゼロ
        assert acc == pytest.approx(expected_free, abs=0.01)

    def test_inline_following(self):
        """インライン追従 → CF モデルの加速度に近い"""
        follower = Vehicle.create("Car", x=0.0, y=6.0, v=15.0, vehicle_id=0)
        leader = Vehicle.create("Car", x=20.0, y=6.0, v=10.0, vehicle_id=1)
        road = Road()
        acc = longitudinal_acceleration(
            follower, [leader], road, DEFAULT_MTM_PARAMS
        )
        # インラインなので α=1, 純粋な IDM
        gap = 20.0 - 0.0 - CAR.length
        expected = idm_acceleration(
            gap, 15.0, 10.0, CAR.v0, CAR.T, CAR.s0, CAR.a, CAR.b,
        )
        assert acc == pytest.approx(expected, abs=0.1)
