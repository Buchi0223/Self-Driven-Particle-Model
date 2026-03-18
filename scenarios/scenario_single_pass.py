"""
Step 10: シナリオ — 単独追い越し

1台の遅いリーダーを追い越す最も基本的な動的テスト。

設定:
    Follower F: Car, v0=18, v_init=15, (x=0, y=5)
    Leader  L: Car, v0=9,  v_init=9,  (x=40, y=6.5) — やや左にオフセット
    Road: 幅12m (y=0~12)
    t_max: 30s

Reference:
    Kanagaraj & Treiber (2018), Section 3, Plausibility Test Case 1
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.parameters import CAR, DEFAULT_MTM_PARAMS, CFParams
from src.vehicle import Vehicle
from src.road import Road
from src.simulation import run_simulation


def main():
    road = Road(y_right=0.0, y_left=12.0, road_length=800.0)
    mtm = DEFAULT_MTM_PARAMS

    # リーダーは v0=9 の遅い車両
    slow_car = CFParams(
        vehicle_type="Car", length=4.2, width=1.7,
        v0=9.0, T=0.8, s0=2.0, a=1.0, b=1.0,
    )

    follower = Vehicle.from_cf_params(CAR, x=0.0, y=5.0, v=15.0, vehicle_id=0)
    leader = Vehicle.from_cf_params(slow_car, x=40.0, y=6.5, v=9.0, vehicle_id=1)

    result = run_simulation(
        [follower, leader], road, mtm,
        t_max=30.0, dt=0.05, record_interval=0.2,
    )

    traj_f = result.get_vehicle_trajectory(0)
    traj_l = result.get_vehicle_trajectory(1)

    # --- プロット ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (1) 相対縦位置
    axes[0].plot(traj_f["time"], [xf - xl for xf, xl in zip(traj_f["x"], traj_l["x"])],
                 "b-", label="F − L (x)")
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("X_F − X_L [m]")
    axes[0].legend()
    axes[0].set_title("Single Vehicle Passing")

    # (2) 横方向位置
    axes[1].plot(traj_f["time"], traj_f["y"], "b-", label="Follower F")
    axes[1].plot(traj_l["time"], traj_l["y"], "r--", label="Leader L")
    axes[1].axhline(road.y_right, color="gray", ls="-", alpha=0.3)
    axes[1].axhline(road.y_left, color="gray", ls="-", alpha=0.3)
    axes[1].set_ylabel("Lateral position y [m]")
    axes[1].legend()

    # (3) 速度
    axes[2].plot(traj_f["time"], traj_f["v"], "b-", label="Follower F")
    axes[2].plot(traj_l["time"], traj_l["v"], "r--", label="Leader L")
    axes[2].set_ylabel("Speed v [m/s]")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("output/step10_single_pass.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step10_single_pass.png")


if __name__ == "__main__":
    main()
