"""
Step 11: シナリオ — 2台の間を通過 (Fig. 5a 再現)

横方向に十分離れた2台のリーダーの間を通過する。

設定 (Fig. 5a に基づく、論文グラフから読取):
    Follower F:  Car, v0=18, v_init=15, (x=0, y=5)    — L1-L2 の中間
    Leader  L1:  Car, v0=9,  v_init=9,  (x=90, y=4)   — 左寄り
    Leader  L2:  Car, v0=6,  v_init=6,  (x=100, y=7)  — 右寄り
    Road: 幅12m
    t_max: 35s

Reference:
    Kanagaraj & Treiber (2018), Fig. 5(a)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.parameters import CAR, DEFAULT_MTM_PARAMS, CFParams
from src.vehicle import Vehicle
from src.road import Road
from src.simulation import run_simulation


def draw_vehicle_rect(ax, x, y, length, width, color, label=None):
    """車両の矩形を描画 (rear-left corner 基準)"""
    rect = patches.Rectangle(
        (x - length, y - width / 2), length, width,
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.4,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x - length / 2, y, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color=color)


def main():
    road = Road(y_left=0.0, y_right=12.0, road_length=1000.0)
    mtm = DEFAULT_MTM_PARAMS

    slow_car_1 = CFParams(
        vehicle_type="Car", length=4.2, width=1.7,
        v0=9.0, T=0.8, s0=2.0, a=1.0, b=1.0,
    )
    slow_car_2 = CFParams(
        vehicle_type="Car", length=4.2, width=1.7,
        v0=6.0, T=0.8, s0=2.0, a=1.0, b=1.0,
    )

    follower = Vehicle.from_cf_params(CAR, x=0.0, y=5.0, v=15.0, vehicle_id=0)
    leader1 = Vehicle.from_cf_params(slow_car_1, x=90.0, y=4.0, v=9.0, vehicle_id=1)
    leader2 = Vehicle.from_cf_params(slow_car_2, x=100.0, y=7.0, v=6.0, vehicle_id=2)

    result = run_simulation(
        [follower, leader1, leader2], road, mtm,
        t_max=35.0, dt=0.05, record_interval=0.2,
    )

    traj_f = result.get_vehicle_trajectory(0)
    traj_l1 = result.get_vehicle_trajectory(1)
    traj_l2 = result.get_vehicle_trajectory(2)

    # --- プロット (Fig. 5a 形式: 3パネル, 横軸 Time [s]) ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (1) 相対縦位置 (X - X_L1)
    axes[0].plot(traj_f["time"],
                 [xf - xl for xf, xl in zip(traj_f["x"], traj_l1["x"])],
                 "b-", label="Follower F")
    axes[0].plot(traj_l2["time"],
                 [x2 - x1 for x2, x1 in zip(traj_l2["x"], traj_l1["x"])],
                 "g--", label="Leader L2")
    axes[0].axhline(0, color="gray", ls="--", alpha=0.5)
    axes[0].set_ylabel("X − X_L1 [m]")
    axes[0].set_xlim(0, 35)
    axes[0].legend()
    axes[0].set_title("(a) Passing Between Two Leaders")

    # (2) 横方向位置
    axes[1].plot(traj_f["time"], traj_f["y"], "b-", label="Follower F")
    axes[1].plot(traj_l1["time"], traj_l1["y"], "r--", label="Leader L1")
    axes[1].plot(traj_l2["time"], traj_l2["y"], "g--", label="Leader L2")
    axes[1].axhline(road.y_left, color="gray", ls="-", alpha=0.3)
    axes[1].axhline(road.y_right, color="gray", ls="-", alpha=0.3)
    axes[1].set_ylabel("Lateral position [m]")
    axes[1].set_ylim(0, 10)
    axes[1].set_xlim(0, 35)
    axes[1].legend()

    # (3) スナップショット (横軸 Time [s], 縦軸 y, 車両矩形を時間軸上に配置)
    ax_snap = axes[2]
    snap_times = [0, 5, 10, 15, 20, 25, 30, 35]
    colors_f = plt.cm.Blues(np.linspace(0.4, 1.0, len(snap_times)))
    colors_l1 = plt.cm.Reds(np.linspace(0.4, 1.0, len(snap_times)))
    colors_l2 = plt.cm.Greens(np.linspace(0.4, 1.0, len(snap_times)))

    for idx, st in enumerate(snap_times):
        ti = min(range(len(traj_f["time"])),
                 key=lambda i: abs(traj_f["time"][i] - st))

        # 横軸=time, 縦軸=y で矩形を配置 (幅=時間方向の太さ, 高さ=車幅)
        rect_w = 0.8  # 時間軸方向の矩形幅
        for traj, color, label_prefix in [
            (traj_f, colors_f[idx], "F"),
            (traj_l1, colors_l1[idx], "L1"),
            (traj_l2, colors_l2[idx], "L2"),
        ]:
            rect = patches.Rectangle(
                (st - rect_w / 2, traj["y"][ti] - CAR.width / 2),
                rect_w, CAR.width,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.5,
            )
            ax_snap.add_patch(rect)
            if idx == 0 or idx == len(snap_times) - 1:
                ax_snap.text(st, traj["y"][ti], label_prefix,
                             ha="center", va="center", fontsize=6,
                             fontweight="bold", color="black")

    ax_snap.set_xlabel("Time [s]")
    ax_snap.set_ylabel("Lateral position [m]")
    ax_snap.set_xlim(0, 35)
    ax_snap.set_ylim(0, 10)
    ax_snap.set_title("Snap shot")
    ax_snap.axhline(road.y_left, color="gray", ls="-", alpha=0.3)
    ax_snap.axhline(road.y_right, color="gray", ls="-", alpha=0.3)

    fig.subplots_adjust(hspace=0.25)
    plt.savefig("output/step11_pass_between.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step11_pass_between.png")


if __name__ == "__main__":
    main()
