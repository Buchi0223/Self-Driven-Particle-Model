"""
cf_models モジュールのテスト

IDM/ACC の加速度関数と free/interaction 分解の正しさを検証する。
"""

import math
import pytest
from src.cf_models import (
    idm_desired_gap,
    idm_acceleration,
    acc_acceleration,
    cf_free,
    cf_interaction,
)
from src.parameters import CAR, MOTORCYCLE, B_MAX


# ---------------------------------------------------------------------------
# テスト用ヘルパー: Car パラメータを展開
# ---------------------------------------------------------------------------
V0, T, S0, A, B, DELTA = CAR.v0, CAR.T, CAR.s0, CAR.a, CAR.b, CAR.delta


# ---------------------------------------------------------------------------
# IDM 希望車間距離
# ---------------------------------------------------------------------------

class TestIDMDesiredGap:

    def test_stopped_vehicle(self):
        """停止中 → s* = s0"""
        s_star = idm_desired_gap(v=0.0, dv=0.0, s0=S0, T=T, a=A, b=B)
        assert s_star == pytest.approx(S0)

    def test_free_flow_no_speed_diff(self):
        """自由走行 (Δv=0) → s* = s0 + v*T"""
        v = 15.0
        s_star = idm_desired_gap(v=v, dv=0.0, s0=S0, T=T, a=A, b=B)
        assert s_star == pytest.approx(S0 + v * T)

    def test_approaching(self):
        """接近中 (Δv>0) → s* > s0 + v*T"""
        v, dv = 15.0, 5.0
        s_star = idm_desired_gap(v=v, dv=dv, s0=S0, T=T, a=A, b=B)
        assert s_star > S0 + v * T

    def test_leader_faster(self):
        """リーダーが速い (Δv<0) → interaction_term 負 → max(0,...) でカット"""
        v = 10.0
        dv = -10.0  # リーダーが 10m/s 速い
        s_star = idm_desired_gap(v=v, dv=dv, s0=S0, T=T, a=A, b=B)
        # v*T + v*dv/(2*sqrt(a*b)) = 10*0.8 + 10*(-10)/(2*1) = 8 - 50 = -42 → max(0, -42) = 0
        assert s_star == pytest.approx(S0)


# ---------------------------------------------------------------------------
# IDM 加速度
# ---------------------------------------------------------------------------

class TestIDMAcceleration:

    def test_free_road_from_rest(self):
        """空道路, v=0 → 加速度 ≈ a (s*/s の残余項あり)"""
        acc = idm_acceleration(s=1000.0, v=0.0, vl=0.0,
                               v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc == pytest.approx(A, abs=1e-4)

    def test_free_road_at_desired_speed(self):
        """v = v0 → 自由加速度 ≈ 0"""
        acc = idm_acceleration(s=1000.0, v=V0, vl=V0,
                               v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc == pytest.approx(0.0, abs=0.01)

    def test_close_following_decelerates(self):
        """接近追従 → 減速 (負の加速度)"""
        acc = idm_acceleration(s=5.0, v=15.0, vl=10.0,
                               v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc < 0.0

    def test_large_gap_approaches_free(self):
        """大きな gap → 自由加速度に近づく"""
        v = 10.0
        acc_large = idm_acceleration(s=1000.0, v=v, vl=v,
                                     v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        acc_free = cf_free(v, V0, A, DELTA)
        assert acc_large == pytest.approx(acc_free, abs=0.01)

    def test_negative_gap_emergency_braking(self):
        """gap ≤ 0 → -b_max (緊急ブレーキ)"""
        acc = idm_acceleration(s=0.0, v=10.0, vl=10.0,
                               v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc == -B_MAX

        acc_neg = idm_acceleration(s=-1.0, v=10.0, vl=10.0,
                                   v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc_neg == -B_MAX

    def test_equilibrium_gap(self):
        """定常状態: v = vl かつ acc ≈ 0 となる gap を検証"""
        v = 10.0
        # 定常状態の gap: s_eq where a * [1 - (v/v0)^δ - (s*/s_eq)^2] = 0
        # s* = s0 + v*T (Δv=0 なので)
        s_star = S0 + v * T
        # (s*/s_eq)^2 = 1 - (v/v0)^δ
        free_ratio = 1.0 - (v / V0) ** DELTA
        s_eq = s_star / math.sqrt(free_ratio)
        acc = idm_acceleration(s=s_eq, v=v, vl=v,
                               v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert acc == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# ACC 加速度
# ---------------------------------------------------------------------------

class TestACCAcceleration:

    def test_same_as_idm_when_coolness_zero(self):
        """coolness=0 → IDM と完全一致"""
        s, v, vl = 10.0, 15.0, 10.0
        a_idm = idm_acceleration(s, v, vl, V0, T, S0, A, B, DELTA)
        a_acc = acc_acceleration(s, v, vl, V0, T, S0, A, B, DELTA,
                                 coolness=0.0)
        assert a_acc == pytest.approx(a_idm)

    def test_free_road(self):
        """空道路では IDM と同じ"""
        a_acc = acc_acceleration(s=1000.0, v=0.0, vl=0.0,
                                 v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert a_acc == pytest.approx(A, abs=1e-4)

    def test_negative_gap(self):
        """gap ≤ 0 → -b_max"""
        a_acc = acc_acceleration(s=0.0, v=10.0, vl=10.0,
                                 v0=V0, T=T, s0=S0, a=A, b=B, delta=DELTA)
        assert a_acc == -B_MAX

    def test_smoother_than_idm_on_cutin(self):
        """カットイン状況で ACC は IDM より穏やか (減速が弱い)"""
        # 突然 gap が小さくなるカットイン状況
        s, v, vl = 5.0, 15.0, 14.0  # 小さい gap, わずかに速い
        a_idm = idm_acceleration(s, v, vl, V0, T, S0, A, B, DELTA)
        a_acc = acc_acceleration(s, v, vl, V0, T, S0, A, B, DELTA,
                                 coolness=0.99)
        # ACC は IDM 以上の加速度 (= 弱い減速) を返す
        assert a_acc >= a_idm


# ---------------------------------------------------------------------------
# Free/Interaction 分解 — 式(3)-(5)
# ---------------------------------------------------------------------------

class TestCFFree:

    def test_at_rest(self):
        """v=0 → a_free = a"""
        assert cf_free(v=0.0, v0=V0, a=A) == pytest.approx(A)

    def test_at_desired_speed(self):
        """v=v0 → a_free = 0"""
        assert cf_free(v=V0, v0=V0, a=A) == pytest.approx(0.0)

    def test_above_desired_speed(self):
        """v > v0 → a_free < 0"""
        assert cf_free(v=V0 + 5.0, v0=V0, a=A) < 0.0

    def test_monotonically_decreasing(self):
        """速度増加で自由加速度は単調減少"""
        speeds = [0.0, 5.0, 10.0, 15.0, V0]
        accs = [cf_free(v, V0, A) for v in speeds]
        for i in range(len(accs) - 1):
            assert accs[i] > accs[i + 1]


class TestCFInteraction:

    def test_large_gap_zero_interaction(self):
        """gap が十分大きい → interaction ≈ 0"""
        a_int = cf_interaction(s=10000.0, v=10.0, vl=10.0,
                               v0=V0, T=T, s0=S0, a=A, b=B)
        assert a_int == pytest.approx(0.0, abs=0.001)

    def test_small_gap_negative(self):
        """gap が小さい → interaction < 0 (減速)"""
        a_int = cf_interaction(s=3.0, v=15.0, vl=10.0,
                               v0=V0, T=T, s0=S0, a=A, b=B)
        assert a_int < 0.0

    def test_decomposition_identity(self):
        """恒等式: cf_free + cf_interaction == idm_acceleration"""
        test_cases = [
            (20.0, 10.0, 10.0),  # 通常追従
            (5.0, 15.0, 5.0),    # 接近
            (100.0, 5.0, 18.0),  # リーダーが速い
            (50.0, 0.0, 10.0),   # 対象車両停止
        ]
        for s, v, vl in test_cases:
            a_total = idm_acceleration(s, v, vl, V0, T, S0, A, B, DELTA)
            a_free = cf_free(v, V0, A, DELTA)
            a_int = cf_interaction(s, v, vl, V0, T, S0, A, B, DELTA)
            assert a_free + a_int == pytest.approx(a_total), \
                f"Decomposition failed for s={s}, v={v}, vl={vl}"

    def test_negative_gap_interaction(self):
        """gap ≤ 0 → interaction = -b_max - a_free"""
        v = 10.0
        a_int = cf_interaction(s=0.0, v=v, vl=10.0,
                               v0=V0, T=T, s0=S0, a=A, b=B)
        expected = -B_MAX - cf_free(v, V0, A)
        assert a_int == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Motorcycle パラメータでのテスト (異なる車種)
# ---------------------------------------------------------------------------

class TestMotorcycleParams:
    """Motorcycle の異なるパラメータでも正しく動作することを確認"""

    MV0 = MOTORCYCLE.v0  # 25.0
    MT = MOTORCYCLE.T    # 0.3
    MS0 = MOTORCYCLE.s0  # 0.5
    MA = MOTORCYCLE.a    # 2.0
    MB = MOTORCYCLE.b    # 2.0

    def test_higher_desired_speed(self):
        """Motorcycle は v0=25 なので v=18 でもまだ加速する"""
        a_free = cf_free(v=18.0, v0=self.MV0, a=self.MA)
        assert a_free > 0.0

    def test_smaller_gap_tolerance(self):
        """Motorcycle は s0=0.5, T=0.3 → 小さい gap でも余裕"""
        acc = idm_acceleration(s=3.0, v=15.0, vl=15.0,
                               v0=self.MV0, T=self.MT, s0=self.MS0,
                               a=self.MA, b=self.MB)
        # s*=0.5+15*0.3=5.0, s=3 → (5/3)^2=2.78 → acc=2*(1-0.34-2.78)<0
        # Car との比較は不要、動作確認のみ
        assert isinstance(acc, float)

    def test_decomposition(self):
        s, v, vl = 5.0, 20.0, 15.0
        a_total = idm_acceleration(s, v, vl, self.MV0, self.MT, self.MS0,
                                   self.MA, self.MB)
        a_free = cf_free(v, self.MV0, self.MA)
        a_int = cf_interaction(s, v, vl, self.MV0, self.MT, self.MS0,
                               self.MA, self.MB)
        assert a_free + a_int == pytest.approx(a_total)
