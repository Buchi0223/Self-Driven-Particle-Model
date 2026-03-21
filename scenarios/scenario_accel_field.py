"""
Step 9: 加速度ベクトル場の可視化 — 論文 Fig. 3 再現

2台のリーダーに対する follower の加速度場を (x, y) グリッドで計算し、
quiver (ベクトル場) + pcolormesh (縦方向加速度) で描画する。

(a) リーダー間通過可能 (Fig. 3a)
(b) リーダー間通過不可 → 迂回必要 (Fig. 3b)

設定 (Fig. 3 caption):
    follower: v=10 m/s, v0=18 m/s, Car パラメータ
    Leader 1: v=9 m/s, 近い方
    Leader 2: v=6 m/s, 遠い方

Reference:
    Kanagaraj & Treiber (2018), Fig. 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.parameters import CAR, DEFAULT_MTM_PARAMS
from src.vehicle import Vehicle
from src.longitudinal import (
    alpha, f_interaction_single, f_boundary_longitudinal,
)
from src.lateral import alpha_tilde, g_boundary_lateral
from src.cf_models import cf_free, cf_interaction
from src.geometry import sx_gap, delta_y, W_bar, sy_gap


def compute_field(
    x_grid, y_grid, leaders, v_follower, v0_follower, car_params, mtm_params,
):
    """(x, y) グリッド上で follower の加速度ベクトル (f, g) を計算"""
    ny, nx = x_grid.shape
    F = np.zeros((ny, nx))
    G = np.zeros((ny, nx))

    for iy in range(ny):
        for ix in range(nx):
            xi = x_grid[iy, ix]
            yi = y_grid[iy, ix]
            vi = v_follower
            wi = 0.0

            # --- 縦方向加速度 ---
            f_self = cf_free(vi, v0_follower, car_params.a, car_params.delta)

            # 最影響リーダー選択 (argmax |f_il|)
            best_f = 0.0
            for ld in leaders:
                f_il = f_interaction_single(
                    xi, yi, vi, car_params.width,
                    ld["x"], ld["y"], ld["v"], car_params.width, car_params.length,
                    v0_follower, car_params.T, car_params.s0,
                    car_params.a, car_params.b, car_params.delta, car_params.b_max,
                    mtm_params.sy0,
                )
                if abs(f_il) > abs(best_f):
                    best_f = f_il

            F[iy, ix] = f_self + best_f

            # --- 横方向加速度 ---
            # 全リーダーからの w0 合計
            w0_total = 0.0
            for ld in leaders:
                dy = delta_y(yi, ld["y"])
                w_bar = W_bar(car_params.width, car_params.width)
                at = alpha_tilde(dy, w_bar, mtm_params.sy0_tilde)
                if at == 0.0:
                    continue
                gap = sx_gap(xi, ld["x"], car_params.length)
                a_int = cf_interaction(
                    gap, vi, ld["v"], v0_follower,
                    car_params.T, car_params.s0,
                    car_params.a, car_params.b, car_params.delta, car_params.b_max,
                )
                sign_dy = 1.0 if dy > 0.0 else (-1.0 if dy < 0.0 else 0.0)
                dw_factor = 1.0 - mtm_params.lam_dw * (0.0 - wi) * sign_dy
                w0_total += mtm_params.lam * at * a_int * dw_factor

            g_int = (w0_total - wi) / mtm_params.tau
            G[iy, ix] = g_int

    return F, G


def plot_accel_field(ax, x_grid, y_grid, F, G, leaders, car_params,
                     panel_label, v_follower):
    """加速度場をプロット (論文 Fig.3 準拠: 横軸 y[m], 縦軸 x[m])"""
    # 縦方向加速度のカラーマップ (論文: jet 系, 範囲 -9~0)
    # jet_r: 0=赤, -9=青 (論文準拠)
    im = ax.pcolormesh(
        y_grid, x_grid, F,
        cmap="jet_r", vmin=-9, vmax=0, shading="auto",
    )

    # ベクトル場 (密に表示, 論文準拠)
    # 大きさをクリップして極端に長い矢印を抑制
    mag = np.sqrt(F**2 + G**2)
    max_mag = 3.0  # 矢印の最大長さに対応する加速度
    scale_factor = np.where(mag > max_mag, max_mag / np.clip(mag, 1e-6, None), 1.0)
    clipped_G = G * scale_factor
    clipped_F = F * scale_factor
    skip = 2
    ax.quiver(
        y_grid[::skip, ::skip], x_grid[::skip, ::skip],
        clipped_G[::skip, ::skip], clipped_F[::skip, ::skip],
        scale=40, width=0.003, headwidth=5, headlength=6,
        color="black", alpha=0.7,
    )

    # リーダーの矩形 (論文: 赤/茶系で塗りつぶし, 黒枠)
    for ld in leaders:
        rect = patches.Rectangle(
            (ld["y"] - car_params.width / 2, ld["x"] - car_params.length),
            car_params.width, car_params.length,
            linewidth=1.5, edgecolor="black", facecolor="darkred", alpha=0.7,
        )
        ax.add_patch(rect)
        # ラベルを矩形の外側に配置 (左のリーダーは左上、右は右上)
        x_offset = -1.2 if ld["y"] < 0 else 1.2
        ax.text(
            ld["y"] + x_offset,
            ld["x"] + 1.0,
            ld["label"],
            ha="center", va="bottom", fontsize=8, fontweight="bold",
            color="white",
        )

    # 速度情報テキストボックス (論文準拠)
    vl1 = next(ld["v"] for ld in leaders if ld["label"] == "Leader 1")
    vl2 = next(ld["v"] for ld in leaders if ld["label"] == "Leader 2")
    info_text = (f"v={v_follower:.1f} m/s, "
                 f"v$_{{lead1}}$={vl1:.1f} m/s, "
                 f"v$_{{lead2}}$={vl2:.1f} m/s")
    ax.text(0.5, 0.03, info_text, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="black", alpha=0.8))

    # パネルラベル (左上)
    ax.text(0.05, 0.97, panel_label, transform=ax.transAxes,
            ha="left", va="top", fontsize=14, fontweight="bold")

    ax.set_xlabel("y[m]", fontsize=11)
    ax.set_ylabel("x[m]", fontsize=11)
    ax.set_xlim(-6, 6)
    ax.set_xticks(range(-6, 8, 2))
    ax.set_ylim(-50, 5)
    # aspect を数値で指定: 縦55m / 横12m のデータを、縦長すぎないように横を伸ばす
    ax.set_aspect(0.5)
    return im


def main():
    mtm = DEFAULT_MTM_PARAMS
    car = CAR
    v_follower = 10.0
    v0_follower = 18.0

    # グリッド (論文 Fig.3: x=-50~0, y=-6~6, 高解像度)
    x_arr = np.linspace(-50, 5, 120)
    y_arr = np.linspace(-6, 6, 80)
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    # --- (a) リーダー間通過可能 ---
    leaders_a = [
        {"x": 0.0, "y": -3.0, "v": 9.0, "label": "Leader 1"},
        {"x": 0.0, "y": 3.0, "v": 6.0, "label": "Leader 2"},
    ]

    # --- (b) リーダー間通過不可 → 迂回 ---
    leaders_b = [
        {"x": 0.0, "y": -1.0, "v": 9.0, "label": "Leader 1"},
        {"x": 0.0, "y": 1.0, "v": 6.0, "label": "Leader 2"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle("accLong [m/s$^2$]", fontsize=13, y=0.92)

    for ax, leaders, panel_label in [
        (axes[0], leaders_a, "(a)"),
        (axes[1], leaders_b, "(b)"),
    ]:
        F, G = compute_field(
            x_grid, y_grid, leaders, v_follower, v0_follower, car, mtm,
        )
        im = plot_accel_field(
            ax, x_grid, y_grid, F, G, leaders, car,
            panel_label, v_follower,
        )
        # 個別カラーバー (論文: 各パネル右側)
        cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("accLong [m/s$^2$]", fontsize=10)

    plt.subplots_adjust(wspace=0.05)
    plt.savefig("output/step9_accel_field.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step9_accel_field.png")


if __name__ == "__main__":
    main()
