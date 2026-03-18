"""
parameters モジュールのテスト

パラメータ定義が正しく読み込め、値が論文 Table 1, 2 と一致することを確認する。
"""

import pytest
from src.parameters import (
    B_MAX,
    MOTORCYCLE, CAR, BUS, AUTO_RICKSHAW,
    CF_PARAMS_BY_TYPE,
    DEFAULT_MTM_PARAMS,
    MTMParams,
)


# ---------------------------------------------------------------------------
# Table 1: CF パラメータ
# ---------------------------------------------------------------------------

class TestCFParams:
    """車種別 CF パラメータが Table 1 の値と一致するか検証"""

    def test_b_max(self):
        assert B_MAX == 9.0

    def test_motorcycle_dimensions(self):
        assert MOTORCYCLE.length == 1.8
        assert MOTORCYCLE.width == 0.6

    def test_motorcycle_cf_params(self):
        assert MOTORCYCLE.v0 == 25.0
        assert MOTORCYCLE.T == 0.3
        assert MOTORCYCLE.s0 == 0.5
        assert MOTORCYCLE.a == 2.0
        assert MOTORCYCLE.b == 2.0

    def test_car_dimensions(self):
        assert CAR.length == 4.2
        assert CAR.width == 1.7

    def test_car_cf_params(self):
        assert CAR.v0 == 18.0
        assert CAR.T == 0.8
        assert CAR.s0 == 2.0
        assert CAR.a == 1.0
        assert CAR.b == 1.0

    def test_bus_dimensions(self):
        assert BUS.length == 10.3
        assert BUS.width == 2.1

    def test_bus_cf_params(self):
        assert BUS.v0 == 14.0
        assert BUS.T == 1.0

    def test_auto_rickshaw_dimensions(self):
        assert AUTO_RICKSHAW.length == 2.6
        assert AUTO_RICKSHAW.width == 0.9

    def test_auto_rickshaw_cf_params(self):
        assert AUTO_RICKSHAW.v0 == 6.0
        assert AUTO_RICKSHAW.T == 1.0

    def test_all_types_have_b_max(self):
        """全車種で b_max = 9.0 が設定されていること"""
        for name, params in CF_PARAMS_BY_TYPE.items():
            assert params.b_max == B_MAX, f"{name}: b_max mismatch"

    def test_all_types_have_default_delta(self):
        """IDM 標準の加速指数 δ=4 が全車種に設定されていること"""
        for name, params in CF_PARAMS_BY_TYPE.items():
            assert params.delta == 4.0, f"{name}: delta mismatch"

    def test_cf_params_by_type_keys(self):
        expected = {"Motorcycle", "Car", "Bus", "Auto-Rickshaw"}
        assert set(CF_PARAMS_BY_TYPE.keys()) == expected

    def test_frozen_immutability(self):
        """パラメータが不変であること"""
        with pytest.raises(AttributeError):
            CAR.v0 = 999.0


# ---------------------------------------------------------------------------
# Table 2: MTM パラメータ
# ---------------------------------------------------------------------------

class TestMTMParams:
    """MTM パラメータが Table 2 の値と一致するか検証"""

    def test_theta(self):
        assert DEFAULT_MTM_PARAMS.theta == 0.2

    def test_sy0(self):
        assert DEFAULT_MTM_PARAMS.sy0 == 0.15

    def test_sy0_tilde(self):
        assert DEFAULT_MTM_PARAMS.sy0_tilde == 0.30

    def test_sy0b(self):
        assert DEFAULT_MTM_PARAMS.sy0b == 0.15

    def test_sy0b_tilde(self):
        assert DEFAULT_MTM_PARAMS.sy0b_tilde == 0.25

    def test_lam(self):
        assert DEFAULT_MTM_PARAMS.lam == 0.4

    def test_tau(self):
        assert DEFAULT_MTM_PARAMS.tau == 1.0

    def test_lam_dw(self):
        assert DEFAULT_MTM_PARAMS.lam_dw == 0.7

    def test_p(self):
        assert DEFAULT_MTM_PARAMS.p == 0.2

    def test_boundary_params_are_estimated(self):
        """bb, bb_tilde は推定値 (論文未明記)"""
        assert DEFAULT_MTM_PARAMS.bb == 3.0
        assert DEFAULT_MTM_PARAMS.bb_tilde == 5.0

    def test_custom_mtm_params(self):
        """カスタム値で MTMParams を生成できること"""
        custom = MTMParams(theta=0.3, lam=0.5)
        assert custom.theta == 0.3
        assert custom.lam == 0.5
        # 未指定のパラメータはデフォルト値
        assert custom.sy0 == 0.15

    def test_frozen_immutability(self):
        with pytest.raises(AttributeError):
            DEFAULT_MTM_PARAMS.theta = 999.0
