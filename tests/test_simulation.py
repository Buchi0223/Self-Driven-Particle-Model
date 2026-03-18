"""
simulation モジュールのテスト

シミュレーションループ全体の動作を検証する。
"""

import math
import pytest

from src.simulation import (
    compute_accelerations,
    euler_step,
    run_simulation,
    SimulationResult,
)
from src.vehicle import Vehicle
from src.road import Road
from src.parameters import CAR, MOTORCYCLE, DEFAULT_MTM_PARAMS, MTMParams
from src.cf_models import cf_free


# ---------------------------------------------------------------------------
# compute_accelerations
# ---------------------------------------------------------------------------

class TestComputeAccelerations:

    def test_single_vehicle_free_road(self):
        """単独車両 → 自由加速度のみ"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        road = Road()
        accs = compute_accelerations([car], road, DEFAULT_MTM_PARAMS)
        f, g = accs[0]
        expected_f = cf_free(car.v, car.v0, car.a, car.delta)
        assert f == pytest.approx(expected_f, abs=0.01)
        assert abs(g) < 0.1  # 道路中央、近傍なし → ほぼゼロ

    def test_two_vehicles(self):
        """2台 → 各車両に加速度が計算される"""
        v0 = Vehicle.create("Car", x=0.0, y=6.0, v=15.0, vehicle_id=0)
        v1 = Vehicle.create("Car", x=30.0, y=6.0, v=10.0, vehicle_id=1)
        road = Road()
        accs = compute_accelerations([v0, v1], road, DEFAULT_MTM_PARAMS)
        assert len(accs) == 2
        # v0 (follower) は減速
        assert accs[0][0] < 0.0
        # v1 (leader, no one ahead) は加速
        assert accs[1][0] > 0.0


# ---------------------------------------------------------------------------
# euler_step
# ---------------------------------------------------------------------------

class TestEulerStep:

    def test_position_update(self):
        """位置が速度 × dt で更新される"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        car.w = 0.0
        dt = 0.05
        # 加速度ゼロ → 等速
        euler_step([car], [(0.0, 0.0)], dt, theta=0.2)
        assert car.x == pytest.approx(10.0 * dt)
        assert car.y == pytest.approx(6.0)
        assert car.v == 10.0
        assert car.w == 0.0

    def test_acceleration_applied(self):
        """加速度が速度に反映される"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        car.w = 0.0
        dt = 0.05
        euler_step([car], [(1.0, 0.5)], dt, theta=0.2)
        assert car.v == pytest.approx(10.0 + 1.0 * dt)
        # w_max = v_new * tan(0.2) ≈ 10.05 * 0.2027 ≈ 2.04
        # w_new = 0 + 0.5 * 0.05 = 0.025 < w_max → そのまま
        assert car.w == pytest.approx(0.0 + 0.5 * dt)

    def test_no_reverse(self):
        """後退禁止: v は 0 以下にならない"""
        car = Vehicle.create("Car", x=100.0, y=6.0, v=0.1)
        car.w = 0.0
        dt = 0.05
        euler_step([car], [(-10.0, 0.0)], dt, theta=0.2)
        assert car.v == 0.0
        assert car.w == 0.0  # v=0 なら w=0 にクランプ

    def test_heading_constraint(self):
        """heading 制約: |w| ≤ v * tan(θ)"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        car.w = 0.0
        dt = 0.05
        theta = 0.2
        # 巨大な横方向加速度
        euler_step([car], [(0.0, 1000.0)], dt, theta=theta)
        v_new = 10.0  # f=0 → v 変化なし
        w_max = v_new * math.tan(theta)
        assert car.w == pytest.approx(w_max)

    def test_heading_constraint_negative(self):
        """heading 制約 (負方向)"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0)
        car.w = 0.0
        dt = 0.05
        theta = 0.2
        euler_step([car], [(0.0, -1000.0)], dt, theta=theta)
        w_max = 10.0 * math.tan(theta)
        assert car.w == pytest.approx(-w_max)

    def test_zero_speed_zero_lateral(self):
        """v=0 のとき w=0 に強制"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=0.0)
        car.w = 1.0
        dt = 0.05
        euler_step([car], [(0.0, 5.0)], dt, theta=0.2)
        assert car.v == 0.0
        assert car.w == 0.0


# ---------------------------------------------------------------------------
# run_simulation: 空道路で単独車両
# ---------------------------------------------------------------------------

class TestRunSimulationFreeRoad:

    def test_accelerates_from_rest(self):
        """空道路 v=0 → v0 へ加速 (Car: a=1.0, IDM曲線で30秒程度で到達)"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=0.0, vehicle_id=0)
        road = Road()
        result = run_simulation([car], road, DEFAULT_MTM_PARAMS,
                                t_max=30.0, dt=0.05, record_interval=1.0)
        traj = result.get_vehicle_trajectory(0)
        # 最終速度は v0=18 に近いはず (IDM free: a=1.0 → 30sで十分)
        assert traj["v"][-1] > 16.0
        # 単調増加
        for i in range(1, len(traj["v"])):
            assert traj["v"][i] >= traj["v"][i - 1] - 0.01  # 微小な数値誤差許容

    def test_stays_near_center(self):
        """横方向: 道路中央に留まる"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0, vehicle_id=0)
        road = Road()
        result = run_simulation([car], road, DEFAULT_MTM_PARAMS,
                                t_max=5.0, dt=0.05)
        traj = result.get_vehicle_trajectory(0)
        for y in traj["y"]:
            assert abs(y - 6.0) < 0.5

    def test_position_advances(self):
        """縦方向位置が進む"""
        car = Vehicle.create("Car", x=0.0, y=6.0, v=15.0, vehicle_id=0)
        road = Road()
        result = run_simulation([car], road, DEFAULT_MTM_PARAMS,
                                t_max=2.0, dt=0.05)
        traj = result.get_vehicle_trajectory(0)
        assert traj["x"][-1] > 25.0  # 2秒で少なくとも25m


# ---------------------------------------------------------------------------
# run_simulation: 2台インライン追従
# ---------------------------------------------------------------------------

class TestRunSimulationInlineFollowing:

    def test_follower_decelerates(self):
        """追従: follower は leader の速度に近づく"""
        follower = Vehicle.create("Car", x=0.0, y=6.0, v=18.0, vehicle_id=0)
        leader = Vehicle.create("Car", x=30.0, y=6.0, v=10.0, vehicle_id=1)
        road = Road()
        result = run_simulation(
            [follower, leader], road, DEFAULT_MTM_PARAMS,
            t_max=20.0, dt=0.05,
        )
        traj_f = result.get_vehicle_trajectory(0)
        traj_l = result.get_vehicle_trajectory(1)
        # follower の最終速度は leader の速度に近い
        assert abs(traj_f["v"][-1] - traj_l["v"][-1]) < 3.0

    def test_no_collision(self):
        """衝突なし: follower は leader を追い越さない"""
        follower = Vehicle.create("Car", x=0.0, y=6.0, v=18.0, vehicle_id=0)
        leader = Vehicle.create("Car", x=20.0, y=6.0, v=10.0, vehicle_id=1)
        road = Road()
        result = run_simulation(
            [follower, leader], road, DEFAULT_MTM_PARAMS,
            t_max=15.0, dt=0.05,
        )
        traj_f = result.get_vehicle_trajectory(0)
        traj_l = result.get_vehicle_trajectory(1)
        # follower の前端が leader の後端を超えない
        for xf, xl in zip(traj_f["x"], traj_l["x"]):
            gap = xl - xf - CAR.length
            assert gap > -0.5, f"Collision: gap={gap:.2f}"


# ---------------------------------------------------------------------------
# run_simulation: 横方向回避
# ---------------------------------------------------------------------------

class TestRunSimulationLateralAvoidance:

    def test_lateral_displacement_with_offset_leader(self):
        """横方向にオフセットした遅いリーダー → follower が横に避ける"""
        follower = Vehicle.create("Car", x=0.0, y=5.0, v=15.0, vehicle_id=0)
        leader = Vehicle.create("Car", x=40.0, y=6.5, v=5.0, vehicle_id=1)
        road = Road()
        result = run_simulation(
            [follower, leader], road, DEFAULT_MTM_PARAMS,
            t_max=15.0, dt=0.05,
        )
        traj_f = result.get_vehicle_trajectory(0)
        # follower の y が初期値 5.0 から変化したはず
        y_range = max(traj_f["y"]) - min(traj_f["y"])
        assert y_range > 0.1, f"No lateral displacement: y_range={y_range:.3f}"


# ---------------------------------------------------------------------------
# SimulationResult ヘルパー
# ---------------------------------------------------------------------------

class TestSimulationResult:

    def test_times(self):
        result = run_simulation(
            [Vehicle.create("Car", x=0.0, y=6.0, v=10.0)],
            Road(), DEFAULT_MTM_PARAMS,
            t_max=1.0, dt=0.05, record_interval=0.5,
        )
        times = result.times
        assert len(times) >= 3  # t=0, 0.5, 1.0
        assert times[0] == pytest.approx(0.0)

    def test_get_vehicle_trajectory(self):
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0, vehicle_id=42)
        result = run_simulation(
            [car], Road(), DEFAULT_MTM_PARAMS,
            t_max=0.5, dt=0.05, record_interval=0.1,
        )
        traj = result.get_vehicle_trajectory(42)
        assert len(traj["time"]) > 0
        assert all(key in traj for key in ["time", "x", "y", "v", "w"])

    def test_nonexistent_vehicle(self):
        car = Vehicle.create("Car", x=0.0, y=6.0, v=10.0, vehicle_id=0)
        result = run_simulation(
            [car], Road(), DEFAULT_MTM_PARAMS,
            t_max=0.5, dt=0.05,
        )
        traj = result.get_vehicle_trajectory(999)
        assert len(traj["time"]) == 0
