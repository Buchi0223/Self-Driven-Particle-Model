"""
Step 12: シナリオ — 2台を迂回 (Fig. 5b 再現)

横方向に近接した2台のリーダーを迂回して追い越す。

設定 (Fig. 5b に基づく、論文グラフから読取):
    Follower F:  Car, v0=18, v_init=15, (x=0, y=5)    — L1 の左側
    Leader  L1:  Car, v0=9,  v_init=9,  (x=90, y=6)   — 右寄り
    Leader  L2:  Car, v0=6,  v_init=6,  (x=100, y=7)  — L1 と近接、間を通れない
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

    # 論文 Fig.5(b): F は 90m 後方、L2 は L1 の 10m 前方
    # L1,L2 を右寄りに配置 → F は左側から迂回
    follower = Vehicle.from_cf_params(CAR, x=0.0, y=5.0, v=15.0, vehicle_id=0)
    leader1 = Vehicle.from_cf_params(slow_car_1, x=90.0, y=6.0, v=9.0, vehicle_id=1)
    leader2 = Vehicle.from_cf_params(slow_car_2, x=100.0, y=7.0, v=6.0, vehicle_id=2)

    result = run_simulation(
        [follower, leader1, leader2], road, mtm,
        t_max=35.0, dt=0.05, record_interval=0.2,
    )

    traj_f = result.get_vehicle_trajectory(0)
    traj_l1 = result.get_vehicle_trajectory(1)
    traj_l2 = result.get_vehicle_trajectory(2)

    # --- プロット (Fig. 5b 形式: 3パネル, 横軸 Time [s]) ---
    # 色: 論文準拠 (L1=赤, L2=青, F=黒)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # (1) 相対縦位置 (X - X_L1)
    axes[0].plot(traj_l1["time"],
                 [0.0] * len(traj_l1["time"]),
                 "r-", label="Leader L1")
    axes[0].plot(traj_l2["time"],
                 [x2 - x1 for x2, x1 in zip(traj_l2["x"], traj_l1["x"])],
                 "b-", label="Leader L2")
    axes[0].plot(traj_f["time"],
                 [xf - xl for xf, xl in zip(traj_f["x"], traj_l1["x"])],
                 "k-", label="Follower F")
    axes[0].set_ylabel("X − X$_{L1}$ [m]")
    axes[0].set_xlim(0, 35)
    axes[0].legend(loc="upper right")
    axes[0].set_title("(b) Circumventing Two Leaders")

    # (2) 横方向位置
    axes[1].plot(traj_l1["time"], traj_l1["y"], "r-", label="Leader L1")
    axes[1].plot(traj_l2["time"], traj_l2["y"], "b-", label="Leader L2")
    axes[1].plot(traj_f["time"], traj_f["y"], "k-", label="Follower F")
    axes[1].set_ylabel("Lateral position [m]")
    axes[1].set_ylim(0, 10)
    axes[1].set_xlim(0, 35)
    axes[1].legend(loc="upper right")

    # (3) スナップショット (論文 Fig.5 準拠: 横軸=Time, 各時刻で車両の相対配置を表示)
    ax_snap = axes[2]
    snap_times = [5, 15, 25]

    for idx, st in enumerate(snap_times):
        ti = min(range(len(traj_f["time"])),
                 key=lambda i: abs(traj_f["time"][i] - st))

        # L1 基準の相対縦位置をスケーリングして時間軸上にオフセット
        x_ref = tl1_x = traj_l1["x"][ti]
        scale_x = 0.08  # 縦方向距離 → 時間軸方向のスケール

        for traj, ec, label_prefix, length in [
            (traj_l1, "red", "L1", CAR.length),
            (traj_l2, "blue", "L2", CAR.length),
            (traj_f, "black", "F", CAR.length),
        ]:
            dx = (traj["x"][ti] - x_ref) * scale_x
            rect_w = length * scale_x * 2
            rect = patches.Rectangle(
                (st + dx - rect_w / 2,
                 traj["y"][ti] - CAR.width / 2),
                rect_w, CAR.width,
                linewidth=1.5, edgecolor=ec, facecolor="none",
            )
            ax_snap.add_patch(rect)
            ax_snap.text(st + dx, traj["y"][ti] + 1.2,
                         label_prefix, ha="center", va="bottom", fontsize=7,
                         fontweight="bold", color=ec)

    ax_snap.set_xlabel("Time [s]")
    ax_snap.set_ylabel("Snap shot")
    ax_snap.set_xlim(0, 35)
    ax_snap.set_ylim(0, 10)

    fig.subplots_adjust(hspace=0.25)
    plt.savefig("output/step12_circumvent.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step12_circumvent.png")


if __name__ == "__main__":
    main()
