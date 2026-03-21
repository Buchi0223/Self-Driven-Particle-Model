"""
lateral モジュールのテスト

MTM 横方向ダイナミクスの各関数を検証する。
"""

import math
import pytest

from src.lateral import (
    alpha_tilde,
    w0_ij_from_leader,
    w0_ij_from_follower,
    w0_desired,
    g_boundary_lateral,
    lateral_acceleration,
)
from src.parameters import CAR, DEFAULT_MTM_PARAMS
from src.vehicle import Vehicle
from src.road import Road


# ---------------------------------------------------------------------------
# α̃ (横方向減衰因子) — 式(16)
# ---------------------------------------------------------------------------

class TestAlphaTilde:

    def test_inline_zero(self):
        """Δy=0 → α̃ = 0 (インラインでは横方向力なし)"""
        assert alpha_tilde(dy=0.0, w_bar=1.7, sy0_tilde=0.30) == 0.0

    def test_positive_dy_positive_result(self):
        """Δy > 0 (リーダーが右) → α̃ > 0"""
        at = alpha_tilde(dy=3.0, w_bar=1.7, sy0_tilde=0.30)
        assert at > 0.0

    def test_negative_dy_negative_result(self):
        """Δy < 0 (リーダーが左) → α̃ < 0"""
        at = alpha_tilde(dy=-3.0, w_bar=1.7, sy0_tilde=0.30)
        assert at < 0.0

    def test_antisymmetry(self):
        """反対称: α̃(-Δy) = -α̃(+Δy)"""
        at_pos = alpha_tilde(dy=2.0, w_bar=1.7, sy0_tilde=0.30)
        at_neg = alpha_tilde(dy=-2.0, w_bar=1.7, sy0_tilde=0.30)
        assert at_pos == pytest.approx(-at_neg)

    def test_full_overlap_edges(self):
        """完全オーバーラップの端: |Δy| = 0 → α̃ = 0"""
        # dy=0 は already tested. dy = tiny but overlap
        at = alpha_tilde(dy=0.1, w_bar=1.7, sy0_tilde=0.30)
        # sy = 0.1 - 1.7 = -1.6, overlap region
        # α̃ = sign(0.1) * (1 + (-1.6)/1.7) = 1 * (1 - 0.941) ≈ 0.059
        assert 0.0 < at < 0.1

    def test_overlap_linear(self):
        """オーバーラップ領域で線形"""
        w_bar = 1.7
        # sy = |dy| - w_bar, for overlap sy < 0
        # α̃ = sign(dy) * (1 + sy / w_bar)
        dy = 1.0  # sy = 1.0 - 1.7 = -0.7
        at = alpha_tilde(dy, w_bar, sy0_tilde=0.30)
        expected = 1.0 * (1.0 + (-0.7) / 1.7)
        assert at == pytest.approx(expected)

    def test_clearance_exponential(self):
        """クリアランス領域で指数減衰"""
        w_bar = 1.7
        sy0_tilde = 0.30
        dy = 3.0  # sy = 3.0 - 1.7 = 1.3
        at = alpha_tilde(dy, w_bar, sy0_tilde)
        expected = 1.0 * math.exp(-1.3 / 0.30)
        assert at == pytest.approx(expected)

    def test_at_boundary_continuity(self):
        """sy=0 (接触点) で連続: overlap側→1, clearance側→exp(0)=1"""
        w_bar = 1.7
        dy = w_bar  # sy = 0 exactly
        at = alpha_tilde(dy, w_bar, sy0_tilde=0.30)
        # overlap: 1 + 0/1.7 = 1, clearance: exp(0) = 1
        assert at == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# w0_ij_from_leader — 式(15)
# ---------------------------------------------------------------------------

class TestW0FromLeader:

    def test_inline_zero(self):
        """インライン (Δy=0) → w0=0 (α̃=0 より)"""
        w0 = w0_ij_from_leader(
            xi=0, yi=5.0, vi=15.0, wi=0.0, Wi=1.7,
            xj=20.0, yj=5.0, vj=10.0, wj=0.0, Wj=1.7, Lj=4.2,
            v0=18.0, T=0.8, s0=2.0, a=1.0, b=1.0,
            delta=4.0, b_max=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7,
        )
        assert w0 == 0.0

    def test_no_cf_interaction_no_lateral(self):
        """CF相互作用なし (大きなgap) → 横方向力なし"""
        w0 = w0_ij_from_leader(
            xi=0, yi=5.0, vi=10.0, wi=0.0, Wi=1.7,
            xj=10000.0, yj=7.0, vj=10.0, wj=0.0, Wj=1.7, Lj=4.2,
            v0=18.0, T=0.8, s0=2.0, a=1.0, b=1.0,
            delta=4.0, b_max=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7,
        )
        assert w0 == pytest.approx(0.0, abs=1e-4)

    def test_slow_leader_right_pushes_left(self):
        """遅いリーダーが右にいる (Δy > 0) → 左に避ける (w0 < 0, y減少方向)

        α̃ > 0 (Δy > 0), a_CF_int < 0 → w0 = λ * (+) * (-) < 0
        """
        w0 = w0_ij_from_leader(
            xi=0, yi=5.0, vi=15.0, wi=0.0, Wi=1.7,
            xj=15.0, yj=7.0, vj=5.0, wj=0.0, Wj=1.7, Lj=4.2,
            v0=18.0, T=0.8, s0=2.0, a=1.0, b=1.0,
            delta=4.0, b_max=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7,
        )
        assert w0 < 0.0

    def test_slow_leader_left_pushes_right(self):
        """遅いリーダーが左にいる (Δy < 0) → 右に避ける (w0 > 0, y増加方向)

        α̃ < 0 (Δy < 0), a_CF_int < 0 → w0 = λ * (-) * (-) > 0
        """
        w0 = w0_ij_from_leader(
            xi=0, yi=5.0, vi=15.0, wi=0.0, Wi=1.7,
            xj=15.0, yj=3.0, vj=5.0, wj=0.0, Wj=1.7, Lj=4.2,
            v0=18.0, T=0.8, s0=2.0, a=1.0, b=1.0,
            delta=4.0, b_max=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7,
        )
        assert w0 > 0.0


# ---------------------------------------------------------------------------
# w0_ij_from_follower — actio=reactio
# ---------------------------------------------------------------------------

class TestW0FromFollower:

    def test_p_zero_no_contribution(self):
        """politeness=0 → フォロワーからの寄与なし"""
        w0 = w0_ij_from_follower(
            xi=50.0, yi=5.0, vi=10.0, wi=0.0, Wi=1.7, Li=4.2,
            xj=30.0, yj=6.0, vj=15.0, wj=0.0, Wj=1.7,
            v0_j=18.0, T_j=0.8, s0_j=2.0, a_j=1.0, b_j=1.0,
            delta_j=4.0, b_max_j=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7, p=0.0,
        )
        assert w0 == 0.0

    def test_has_contribution_with_p(self):
        """p > 0 → フォロワーからの寄与あり"""
        w0 = w0_ij_from_follower(
            xi=50.0, yi=5.0, vi=10.0, wi=0.0, Wi=1.7, Li=4.2,
            xj=30.0, yj=6.0, vj=15.0, wj=0.0, Wj=1.7,
            v0_j=18.0, T_j=0.8, s0_j=2.0, a_j=1.0, b_j=1.0,
            delta_j=4.0, b_max_j=9.0,
            sy0_tilde=0.30, lam=0.4, lam_dw=0.7, p=0.2,
        )
        assert w0 != 0.0


# ---------------------------------------------------------------------------
# g_boundary_lateral — 式(11)
# ---------------------------------------------------------------------------

class TestGBoundaryLateral:

    def test_center_near_zero(self):
        """道路中央 → 左右の力が打ち消し合ってほぼゼロ"""
        g = g_boundary_lateral(
            vi=15.0, yi=6.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        assert abs(g) < 0.01

    def test_near_left_pushes_right(self):
        """左境界接近 (yi=1, y_left=0) → 右へ押す (g > 0, y増加方向)"""
        g = g_boundary_lateral(
            vi=15.0, yi=1.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        assert g > 0.0

    def test_near_right_pushes_left(self):
        """右境界接近 (yi=11, y_right=12) → 左へ押す (g < 0, y減少方向)"""
        g = g_boundary_lateral(
            vi=15.0, yi=11.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        assert g < 0.0

    def test_zero_speed_no_force(self):
        """v=0 → 境界力なし"""
        g = g_boundary_lateral(
            vi=0.0, yi=0.5, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        assert g == 0.0

    def test_increases_near_boundary(self):
        """境界に近づくほど力が強くなる"""
        g_far = g_boundary_lateral(
            vi=15.0, yi=3.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        g_near = g_boundary_lateral(
            vi=15.0, yi=1.0, Wi=1.7, v0=18.0,
            y_left=0.0, y_right=12.0, bb_tilde=5.0, sy0b_tilde=0.25,
        )
        assert abs(g_near) > abs(g_far)


# ---------------------------------------------------------------------------
# w0_desired — 式(13)
# ---------------------------------------------------------------------------

class TestW0Desired:

    def test_no_neighbors(self):
        """近傍なし → w0 = 0"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=15.0)
        w0 = w0_desired(car, [], DEFAULT_MTM_PARAMS)
        assert w0 == 0.0

    def test_leader_contribution(self):
        """リーダーがいると w0 ≠ 0"""
        follower = Vehicle.create("Car", x=0.0, y=5.0, v=15.0, vehicle_id=0)
        leader = Vehicle.create("Car", x=15.0, y=7.0, v=5.0, vehicle_id=1)
        w0 = w0_desired(follower, [leader], DEFAULT_MTM_PARAMS)
        # 遅いリーダーが右 (Δy > 0) → 左に避ける → w0 < 0
        assert w0 < 0.0


# ---------------------------------------------------------------------------
# lateral_acceleration — 式(10), (12)
# ---------------------------------------------------------------------------

class TestLateralAcceleration:

    def test_no_neighbors_relaxes_to_zero(self):
        """近傍なし, w=0.5 → g = (0 - 0.5)/τ = -0.5 (w を 0 に緩和)"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=15.0)
        car.w = 0.5
        road = Road()
        g = lateral_acceleration(car, [], road, DEFAULT_MTM_PARAMS)
        # g ≈ (0 - 0.5) / 1.0 = -0.5 + boundary (negligible at center)
        assert g == pytest.approx(-0.5, abs=0.05)

    def test_no_neighbors_w_zero(self):
        """近傍なし, w=0, 道路中央 → g ≈ 0"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=15.0)
        car.w = 0.0
        road = Road()
        g = lateral_acceleration(car, [], road, DEFAULT_MTM_PARAMS)
        assert abs(g) < 0.05

    def test_near_boundary_pushes_away(self):
        """境界近くでは境界から離れる方向に力"""
        car = Vehicle.create("Car", x=0.0, y=1.0, v=15.0)
        car.w = 0.0
        road = Road()
        g = lateral_acceleration(car, [], road, DEFAULT_MTM_PARAMS)
        assert g > 0.0  # 左境界から離れる = y 増加方向 (右へ)
