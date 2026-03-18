"""
Step 12: シナリオ — 2台を迂回 (Fig. 5b 再現)

横方向に近接した2台のリーダーを迂回して追い越す。

設定 (Fig. 5b に基づく):
    Follower F:  Car, v0=18, v_init=15, (x=0, y=3)    — 右寄りスタート
    Leader  L1:  Car, v0=9,  v_init=9,  (x=50, y=5.5) — 中央やや右
    Leader  L2:  Car, v0=6,  v_init=6,  (x=55, y=7.5) — 中央やや左、間を通れない
    Road: 幅12m
    t_max: 35s

Reference:
    Kanagaraj & Treiber (2018), Fig. 5(b)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.parameters import CAR, DEFAULT_MTM_PARAMS, CFParams
from src.vehicle import Vehicle
from src.road import Road
from src.simulation import run_simulation


def draw_vehicle_rect(ax, x, y, length, width, color, label=None):
    """車両の矩形を描画"""
    rect = patches.Rectangle(
        (x - length, y - width / 2), length, width,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x - length / 2, y, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color=color)


def main():
    road = Road(y_right=0.0, y_left=12.0, road_length=1000.0)
    mtm = DEFAULT_MTM_PARAMS

    slow_car_1 = CFParams(
        vehicle_type="Car", length=4.2, width=1.7,
        v0=9.0, T=0.8, s0=2.0, a=1.0, b=1.0,
    )
    slow_car_2 = CFParams(
        vehicle_type="Car", length=4.2, width=1.7,
        v0=6.0, T=0.8, s0=2.0, a=1.0, b=1.0,
    )

    # L1 と L2 が近接 (sy ≈ 0.3m) → 間を通れない → 右側から迂回
    follower = Vehicle.from_cf_params(CAR, x=0.0, y=3.0, v=15.0, vehicle_id=0)
    leader1 = Vehicle.from_cf_params(slow_car_1, x=50.0, y=5.5, v=9.0, vehicle_id=1)
    leader2 = Vehicle.from_cf_params(slow_car_2, x=55.0, y=7.5, v=6.0, vehicle_id=2)

    result = run_simulation(
        [follower, leader1, leader2], road, mtm,
        t_max=35.0, dt=0.05, record_interval=0.2,
    )

    traj_f = result.get_vehicle_trajectory(0)
    traj_l1 = result.get_vehicle_trajectory(1)
    traj_l2 = result.get_vehicle_trajectory(2)

    # --- プロット (Fig. 5b 形式: 3パネル) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # (1) 相対縦位置 (X - X_L1)
    axes[0].plot(traj_f["time"],
                 [xf - xl for xf, xl in zip(traj_f["x"], traj_l1["x"])],
                 "b-", label="Follower F")
    axes[0].plot(traj_l2["time"],
                 [x2 - x1 for x2, x1 in zip(traj_l2["x"], traj_l1["x"])],
                 "g--", label="Leader L2")
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("X − X_L1 [m]")
    axes[0].legend()
    axes[0].set_title("(b) Circumventing Two Leaders")

    # (2) 横方向位置
    axes[1].plot(traj_f["time"], traj_f["y"], "b-", label="Follower F")
    axes[1].plot(traj_l1["time"], traj_l1["y"], "r--", label="Leader L1")
    axes[1].plot(traj_l2["time"], traj_l2["y"], "g--", label="Leader L2")
    axes[1].axhline(road.y_right, color="gray", ls="-", alpha=0.3)
    axes[1].axhline(road.y_left, color="gray", ls="-", alpha=0.3)
    axes[1].set_ylabel("Lateral position y [m]")
    axes[1].set_ylim(-1, 13)
    axes[1].legend()

    # (3) スナップショット
    ax_snap = axes[2]
    snap_times = [0, 4, 8, 12, 16, 20]
    colors_f = plt.cm.Blues(np.linspace(0.3, 1.0, len(snap_times)))
    colors_l1 = plt.cm.Reds(np.linspace(0.3, 1.0, len(snap_times)))
    colors_l2 = plt.cm.Greens(np.linspace(0.3, 1.0, len(snap_times)))

    for idx, st in enumerate(snap_times):
        ti = min(range(len(traj_f["time"])),
                 key=lambda i: abs(traj_f["time"][i] - st))
        x_ref = traj_l1["x"][ti]

        draw_vehicle_rect(
            ax_snap, traj_f["x"][ti] - x_ref, traj_f["y"][ti],
            CAR.length, CAR.width, colors_f[idx],
            f"F({st}s)" if idx % 2 == 0 else None,
        )
        draw_vehicle_rect(
            ax_snap, traj_l1["x"][ti] - x_ref, traj_l1["y"][ti],
            CAR.length, CAR.width, colors_l1[idx],
            f"L1({st}s)" if idx % 2 == 0 else None,
        )
        draw_vehicle_rect(
            ax_snap, traj_l2["x"][ti] - x_ref, traj_l2["y"][ti],
            CAR.length, CAR.width, colors_l2[idx],
            f"L2({st}s)" if idx % 2 == 0 else None,
        )

    ax_snap.set_xlabel("X − X_L1 [m]")
    ax_snap.set_ylabel("y [m]")
    ax_snap.set_xlim(-60, 40)
    ax_snap.set_ylim(-1, 13)
    ax_snap.set_title("Snapshots")
    ax_snap.axhline(road.y_right, color="gray", ls="-", alpha=0.3)
    ax_snap.axhline(road.y_left, color="gray", ls="-", alpha=0.3)

    fig.subplots_adjust(hspace=0.3)
    plt.savefig("output/step12_circumvent.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step12_circumvent.png")


if __name__ == "__main__":
    main()
