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


def plot_accel_field(ax, x_grid, y_grid, F, G, leaders, car_params, title):
    """加速度場をプロット (論文 Fig.3 準拠: 横軸 y[m], 縦軸 x[m])"""
    # 縦方向加速度のカラーマップ
    vmax = max(abs(F.min()), abs(F.max()))
    vmax = min(vmax, 9.0)
    # 横軸=y, 縦軸=x にするため y_grid, x_grid の順
    im = ax.pcolormesh(
        y_grid, x_grid, F,
        cmap="RdYlGn", vmin=-vmax, vmax=vmax, shading="auto",
    )

    # ベクトル場 (間引き): (G, F) = (横方向, 縦方向) に対応
    skip = 3
    scale = 80
    ax.quiver(
        y_grid[::skip, ::skip], x_grid[::skip, ::skip],
        G[::skip, ::skip], F[::skip, ::skip],
        scale=scale, width=0.003, color="black", alpha=0.6,
    )

    # リーダーの矩形 (横軸=y, 縦軸=x)
    for ld in leaders:
        rect = patches.Rectangle(
            (ld["y"] - car_params.width / 2, ld["x"] - car_params.length),
            car_params.width, car_params.length,
            linewidth=2, edgecolor="white", facecolor="white", alpha=0.8,
        )
        ax.add_patch(rect)
        ax.text(
            ld["y"],
            ld["x"] - car_params.length / 2,
            ld["label"],
            ha="center", va="center", fontsize=8, fontweight="bold",
        )

    ax.set_xlabel("y [m]")
    ax.set_ylabel("x [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    return im


def main():
    mtm = DEFAULT_MTM_PARAMS
    car = CAR
    v_follower = 10.0
    v0_follower = 18.0

    # グリッド (論文 Fig.3: x=-50~0, y=-6~6)
    x_arr = np.linspace(-50, 5, 80)
    y_arr = np.linspace(-6, 6, 60)
    x_grid, y_grid = np.meshgrid(x_arr, y_arr)

    # --- (a) リーダー間通過可能 ---
    leaders_a = [
        {"x": 0.0, "y": -3.0, "v": 9.0, "label": "L1"},
        {"x": 0.0, "y": 3.0, "v": 6.0, "label": "L2"},
    ]

    # --- (b) リーダー間通過不可 → 迂回 ---
    leaders_b = [
        {"x": 0.0, "y": -1.0, "v": 9.0, "label": "L1"},
        {"x": 0.0, "y": 1.0, "v": 6.0, "label": "L2"},
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, leaders, label in [
        (axes[0], leaders_a, "(a) Passing between leaders"),
        (axes[1], leaders_b, "(b) Circumventing leaders"),
    ]:
        F, G = compute_field(
            x_grid, y_grid, leaders, v_follower, v0_follower, car, mtm,
        )
        im = plot_accel_field(ax, x_grid, y_grid, F, G, leaders, car, label)

    fig.colorbar(im, ax=axes, label="Longitudinal acceleration [m/s²]", shrink=0.8)
    fig.suptitle(
        f"MTM Acceleration Vector Field (v={v_follower}, v0={v0_follower})",
        fontsize=14,
    )
    plt.savefig("output/step9_accel_field.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: output/step9_accel_field.png")


if __name__ == "__main__":
    main()
