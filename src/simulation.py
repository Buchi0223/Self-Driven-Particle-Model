"""
シミュレーションループモジュール

全モジュール (longitudinal, lateral, vehicle, road, parameters) を結合し、
時間積分によって車両の軌跡を生成する。

積分法: Euler 法 (dt = 0.05 s)
制約:
    - v ≥ 0 (後退禁止)
    - |w/v| ≤ tan(θ) (heading 制約, 式(1)下の記述)

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.longitudinal import longitudinal_acceleration
from src.lateral import lateral_acceleration

if TYPE_CHECKING:
    from src.vehicle import Vehicle
    from src.road import Road
    from src.parameters import MTMParams


# ---------------------------------------------------------------------------
# シミュレーション履歴
# ---------------------------------------------------------------------------

@dataclass
class Snapshot:
    """ある時刻の全車両状態のスナップショット"""
    time: float
    states: list[VehicleState]


@dataclass
class VehicleState:
    """ある時刻の車両状態 (履歴保存用, 軽量コピー)"""
    vehicle_id: int
    vehicle_type: str
    x: float
    y: float
    v: float
    w: float
    f_long: float  # 縦方向加速度
    g_lat: float   # 横方向加速度


@dataclass
class SimulationResult:
    """シミュレーション結果"""
    snapshots: list[Snapshot] = field(default_factory=list)
    dt: float = 0.05
    params_info: str = ""

    @property
    def times(self) -> list[float]:
        return [s.time for s in self.snapshots]

    def get_vehicle_trajectory(self, vehicle_id: int) -> dict[str, list[float]]:
        """指定車両の時系列データを取得

        Returns:
            {"time": [...], "x": [...], "y": [...], "v": [...], "w": [...],
             "f_long": [...], "g_lat": [...]}
        """
        traj: dict[str, list[float]] = {
            "time": [], "x": [], "y": [], "v": [], "w": [],
            "f_long": [], "g_lat": [],
        }
        for snap in self.snapshots:
            for vs in snap.states:
                if vs.vehicle_id == vehicle_id:
                    traj["time"].append(snap.time)
                    traj["x"].append(vs.x)
                    traj["y"].append(vs.y)
                    traj["v"].append(vs.v)
                    traj["w"].append(vs.w)
                    traj["f_long"].append(vs.f_long)
                    traj["g_lat"].append(vs.g_lat)
                    break
        return traj


# ---------------------------------------------------------------------------
# 加速度計算
# ---------------------------------------------------------------------------

def compute_accelerations(
    vehicles: list[Vehicle],
    road: Road,
    mtm_params: MTMParams,
) -> list[tuple[float, float]]:
    """全車両の加速度 (f_long, g_lat) を計算

    Args:
        vehicles:   車両リスト
        road:       道路情報
        mtm_params: MTM パラメータ

    Returns:
        [(f_long_0, g_lat_0), (f_long_1, g_lat_1), ...]
    """
    accelerations = []
    for veh in vehicles:
        f_long = longitudinal_acceleration(veh, vehicles, road, mtm_params)
        g_lat = lateral_acceleration(veh, vehicles, road, mtm_params)
        accelerations.append((f_long, g_lat))
    return accelerations


# ---------------------------------------------------------------------------
# Euler 積分ステップ
# ---------------------------------------------------------------------------

def euler_step(
    vehicles: list[Vehicle],
    accelerations: list[tuple[float, float]],
    dt: float,
    theta: float,
) -> None:
    """Euler 法で1ステップ更新 (in-place)

    更新順序:
        1. 速度更新: v += f*dt, w += g*dt
        2. 速度制約: v ≥ 0, |w/v| ≤ tan(θ)
        3. 位置更新: x += v*dt, y += w*dt

    Args:
        vehicles:      車両リスト (in-place 更新)
        accelerations: 各車両の (f_long, g_lat)
        dt:            タイムステップ [s]
        theta:         最大道路軸角度 [rad]
    """
    tan_theta = math.tan(theta)

    for veh, (f_long, g_lat) in zip(vehicles, accelerations):
        # 速度更新
        v_new = veh.v + f_long * dt
        w_new = veh.w + g_lat * dt

        # 制約 1: 後退禁止
        v_new = max(v_new, 0.0)

        # 制約 2: heading 制約 |w/v| ≤ tan(θ)
        if v_new > 0.0:
            w_max = v_new * tan_theta
            w_new = max(-w_max, min(w_new, w_max))
        else:
            # v = 0 のとき横方向速度もゼロに
            w_new = 0.0

        # 位置更新 (更新後の速度を使用)
        veh.x += v_new * dt
        veh.y += w_new * dt

        # 速度を確定
        veh.v = v_new
        veh.w = w_new


# ---------------------------------------------------------------------------
# メインシミュレーションループ
# ---------------------------------------------------------------------------

def run_simulation(
    vehicles: list[Vehicle],
    road: Road,
    mtm_params: MTMParams,
    t_max: float,
    dt: float = 0.05,
    record_interval: float = 0.1,
) -> SimulationResult:
    """シミュレーションを実行し、軌跡履歴を返す

    Args:
        vehicles:        初期車両リスト (シミュレーション中に in-place 更新される)
        road:            道路情報
        mtm_params:      MTM パラメータ
        t_max:           シミュレーション終了時刻 [s]
        dt:              タイムステップ [s]  (デフォルト 0.05)
        record_interval: 履歴記録間隔 [s]   (デフォルト 0.1)

    Returns:
        SimulationResult
    """
    result = SimulationResult(dt=dt)
    t = 0.0
    next_record = 0.0

    while t <= t_max + dt / 2.0:
        # 履歴記録
        if t >= next_record - dt / 2.0:
            _record_snapshot(result, vehicles, t)
            next_record += record_interval

        # 加速度計算
        accelerations = compute_accelerations(vehicles, road, mtm_params)

        # 加速度をスナップショットに反映 (直前に記録した場合)
        if result.snapshots and abs(result.snapshots[-1].time - t) < dt / 2.0:
            for vs, (f, g) in zip(result.snapshots[-1].states, accelerations):
                vs.f_long = f
                vs.g_lat = g

        # Euler ステップ
        euler_step(vehicles, accelerations, dt, mtm_params.theta)

        t += dt

    return result


def _record_snapshot(
    result: SimulationResult,
    vehicles: list[Vehicle],
    t: float,
) -> None:
    """現在の車両状態をスナップショットとして記録"""
    states = [
        VehicleState(
            vehicle_id=veh.vehicle_id,
            vehicle_type=veh.vehicle_type,
            x=veh.x,
            y=veh.y,
            v=veh.v,
            w=veh.w,
            f_long=0.0,  # 後で更新
            g_lat=0.0,
        )
        for veh in vehicles
    ]
    result.snapshots.append(Snapshot(time=round(t, 6), states=states))
