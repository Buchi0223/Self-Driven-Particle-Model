"""
MTM 横方向ダイナミクスモジュール

車両間の横方向加速度を計算する。
MOBIL 車線変更モデルを連続横方向座標に一般化した OV-like モデル。

全横方向加速度 (式10):
    g_i = g_self + g_boundary + g_int

    g_self = 0  (本論文では戦術的成分を考慮しない)
    g_boundary — 式(11)
    g_int = (w0_i - w_i) / τ  — 式(12)
    w0_i = Σ_j w0_{ij}        — 式(13)

ペア相互作用 (式15):
    w0_{ij} = λ · α̃(Δy) · a_CF_int · [1 - λ_{Δw} · (w_j - w_i) · sign(Δy)]

横方向減衰因子 (式16):
    α̃(Δy) = sign(Δy) · {
        1 + s^y / W̄     if s^y < 0  (オーバーラップ)
        exp(-s^y / s̃^y_0) if s^y ≥ 0  (クリアランス)
    }

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2.2, Eq. (10)-(16)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.cf_models import cf_interaction
from src.geometry import delta_y, W_bar, sy_gap, sx_gap

if TYPE_CHECKING:
    from src.vehicle import Vehicle
    from src.road import Road
    from src.parameters import MTMParams


# ---------------------------------------------------------------------------
# 横方向減衰因子 α̃ — 式(16)
# ---------------------------------------------------------------------------

def alpha_tilde(dy: float, w_bar: float, sy0_tilde: float) -> float:
    """横方向減衰因子 α̃(Δy_{ij}) — 式(16)

    オーバーラップ領域 (s^y < 0) では線形、クリアランス領域では指数減衰。
    符号付き: リーダーが右 (Δy > 0) → 正、左 (Δy < 0) → 負。
    Δy = 0 のとき α̃ = 0 (インラインでは横方向力なし)。

    Args:
        dy:        Δy_{ij} = yj - yi [m]
        w_bar:     平均車幅 W̄_{ij} [m]
        sy0_tilde: 横方向減衰スケール(操舵用) s̃^y_0 [m]

    Returns:
        α̃ (符号付き)
    """
    if dy == 0.0:
        return 0.0

    sign_dy = 1.0 if dy > 0.0 else -1.0
    sy = abs(dy) - w_bar  # s^y_{ij}

    if sy < 0.0:
        # オーバーラップ: 線形 (式16 上段)
        # sign(Δy) · (1 + s^y / W̄)
        # s^y/W̄ は [-1, 0) の範囲 → 結果は (0, 1] の範囲
        return sign_dy * (1.0 + sy / w_bar)
    else:
        # クリアランス: 指数減衰 (式16 下段)
        return sign_dy * math.exp(-sy / sy0_tilde)


# ---------------------------------------------------------------------------
# ペア横方向 desired speed — 式(15)
# ---------------------------------------------------------------------------

def w0_ij_from_leader(
    xi: float, yi: float, vi: float, wi: float, Wi: float,
    xj: float, yj: float, vj: float, wj: float, Wj: float, Lj: float,
    v0: float, T: float, s0: float, a: float, b: float,
    delta: float, b_max: float,
    sy0_tilde: float, lam: float, lam_dw: float,
) -> float:
    """リーダー j から車両 i への横方向 desired speed 寄与 w^0_{ij} — 式(15)

    w^0_{ij} = λ · α̃(Δy) · a_CF_int · [1 - λ_{Δw} · (w_j - w_i) · sign(Δy)]

    Args:
        xi..Wi:  対象車両 i の状態・車幅
        xj..Lj:  リーダー j の状態・車幅・車長
        v0..b_max: CF パラメータ (対象車両 i)
        sy0_tilde: 横方向減衰スケール(操舵用) [m]
        lam:       横方向感度 λ [s]
        lam_dw:    横方向相対速度感度 λ_{Δw} [s/m]

    Returns:
        w^0_{ij} [m/s]
    """
    dy = delta_y(yi, yj)
    w_bar = W_bar(Wi, Wj)
    at = alpha_tilde(dy, w_bar, sy0_tilde)

    if at == 0.0:
        return 0.0

    gap = sx_gap(xi, xj, Lj)
    a_cf_int = cf_interaction(gap, vi, vj, v0, T, s0, a, b, delta, b_max)

    sign_dy = 1.0 if dy > 0.0 else (-1.0 if dy < 0.0 else 0.0)
    dw_factor = 1.0 - lam_dw * (wj - wi) * sign_dy

    return lam * at * a_cf_int * dw_factor


def w0_ij_from_follower(
    xi: float, yi: float, vi: float, wi: float, Wi: float, Li: float,
    xj: float, yj: float, vj: float, wj: float, Wj: float,
    v0_j: float, T_j: float, s0_j: float, a_j: float, b_j: float,
    delta_j: float, b_max_j: float,
    sy0_tilde: float, lam: float, lam_dw: float, p: float,
) -> float:
    """フォロワー j から車両 i への横方向 desired speed 寄与 — actio=reactio

    フォロワーを含める場合 (論文 Section 2.2, p.8):
    1. 役割を交換: j がフォロワー, i がリーダーとして CF 相互作用を計算
    2. 符号を反転
    3. politeness factor p で重み付け

    Args:
        xi..Li:    対象車両 i の状態・車幅・車長 (ここでは i がリーダー)
        xj..Wj:    フォロワー j の状態・車幅
        v0_j..b_max_j: フォロワー j の CF パラメータ
        sy0_tilde, lam, lam_dw: MTM パラメータ
        p:         politeness factor

    Returns:
        p · (-w^0_{ji}) [m/s]
    """
    if p == 0.0:
        return 0.0

    # j から見た i への相互作用を計算 (j がフォロワー, i がリーダー)
    w0_ji = w0_ij_from_leader(
        xi=xj, yi=yj, vi=vj, wi=wj, Wi=Wj,
        xj=xi, yj=yi, vj=vi, wj=wi, Wj=Wi, Lj=Li,
        v0=v0_j, T=T_j, s0=s0_j, a=a_j, b=b_j,
        delta=delta_j, b_max=b_max_j,
        sy0_tilde=sy0_tilde, lam=lam, lam_dw=lam_dw,
    )

    # actio=reactio: 符号反転, politeness で重み付け
    return -p * w0_ji


# ---------------------------------------------------------------------------
# 全近傍からの desired lateral speed — 式(13)
# ---------------------------------------------------------------------------

def w0_desired(
    vehicle_i: Vehicle,
    others: list[Vehicle],
    mtm_params: MTMParams,
) -> float:
    """車両 i の desired lateral speed w^0_i — 式(13)

    w^0_i = Σ_j w^0_{ij}

    前方車両 (リーダー) と後方車両 (フォロワー) の両方を含む。
    相互作用が閾値 a_thr 未満の車両は除外する。

    Args:
        vehicle_i: 対象車両
        others:    他の車両リスト
        mtm_params: MTM パラメータ

    Returns:
        w^0_i [m/s]
    """
    i = vehicle_i
    mp = mtm_params
    w0_total = 0.0

    for j in others:
        if j.vehicle_id == i.vehicle_id:
            continue

        if j.x > i.x:
            # j はリーダー: |a_CF_int_ij| > a_thr でフィルタ
            gap = sx_gap(i.x, j.x, j.length)
            a_int = cf_interaction(
                gap, i.v, j.v, i.v0, i.T, i.s0, i.a, i.b, i.delta, i.b_max
            )
            if abs(a_int) > mp.a_thr:
                w0_total += w0_ij_from_leader(
                    i.x, i.y, i.v, i.w, i.width,
                    j.x, j.y, j.v, j.w, j.width, j.length,
                    i.v0, i.T, i.s0, i.a, i.b, i.delta, i.b_max,
                    mp.sy0_tilde, mp.lam, mp.lam_dw,
                )
        else:
            # j はフォロワー: |a_CF_int_ji| > a_thr でフィルタ
            gap_ji = sx_gap(j.x, i.x, i.length)
            a_int_ji = cf_interaction(
                gap_ji, j.v, i.v, j.v0, j.T, j.s0, j.a, j.b, j.delta, j.b_max
            )
            if abs(a_int_ji) > mp.a_thr:
                w0_total += w0_ij_from_follower(
                    i.x, i.y, i.v, i.w, i.width, i.length,
                    j.x, j.y, j.v, j.w, j.width,
                    j.v0, j.T, j.s0, j.a, j.b, j.delta, j.b_max,
                    mp.sy0_tilde, mp.lam, mp.lam_dw, mp.p,
                )

    return w0_total


# ---------------------------------------------------------------------------
# 道路境界の横方向力 — 式(11)
# ---------------------------------------------------------------------------

def g_boundary_lateral(
    vi: float, yi: float, Wi: float, v0: float,
    y_left: float, y_right: float,
    bb_tilde: float, sy0b_tilde: float,
) -> float:
    """道路境界からの横方向力 — 式(11)

    g_{ib} = ±b̃_b · (v_i/v0) · exp(-s^y_{ib} / s̃^y_{0b})

    論文 p.6: "the '+' sign applies for the left boundary
              (y is increasing to the right)"

    座標系: y は右方向に増加 (論文準拠)。y_left < y_right。
        左境界 (y_left):  車両を右へ押す → + (y 増加方向)
        右境界 (y_right): 車両を左へ押す → - (y 減少方向)

    s^y_{ib} = ±(y_b - y_i) - W_i/2
        + for right boundary, - for left boundary  (論文 p.5)

    Args:
        vi:         対象車両の速度 [m/s]
        yi:         対象車両の横方向位置 [m]
        Wi:         対象車両の車幅 [m]
        v0:         希望速度 [m/s]
        y_left:     左側境界 y 座標 [m]  (小さい値)
        y_right:    右側境界 y 座標 [m]  (大きい値)
        bb_tilde:   境界横方向加速度 [m/s^2]
        sy0b_tilde: 境界減衰スケール(操舵用) [m]

    Returns:
        合計横方向境界力 [m/s^2]
    """
    if v0 <= 0.0 or vi <= 0.0:
        return 0.0

    speed_factor = vi / v0

    # 左側境界: 車両を右 (y増加方向) へ押す → +
    # s^y = -(y_left - yi) - Wi/2 = (yi - y_left) - Wi/2
    sy_left = (yi - y_left) - Wi / 2.0
    g_left = +bb_tilde * speed_factor * math.exp(-sy_left / sy0b_tilde)

    # 右側境界: 車両を左 (y減少方向) へ押す → -
    # s^y = +(y_right - yi) - Wi/2
    sy_right = (y_right - yi) - Wi / 2.0
    g_right = -bb_tilde * speed_factor * math.exp(-sy_right / sy0b_tilde)

    return g_left + g_right


# ---------------------------------------------------------------------------
# 全横方向加速度 — 式(10), (12)
# ---------------------------------------------------------------------------

def lateral_acceleration(
    vehicle_i: Vehicle,
    others: list[Vehicle],
    road: Road,
    mtm_params: MTMParams,
) -> float:
    """車両 i の全横方向加速度 — 式(10), (12)

    g_i = g_self + g_boundary + g_int
        = 0 + g_boundary + (w0_i - w_i) / τ

    Args:
        vehicle_i:  対象車両
        others:     他の車両リスト
        road:       道路情報
        mtm_params: MTM パラメータ

    Returns:
        横方向加速度 [m/s^2]
    """
    i = vehicle_i
    mp = mtm_params

    # g_self = 0 (戦術的成分なし)

    # 道路境界力 — 式(11)
    g_bnd = g_boundary_lateral(
        i.v, i.y, i.width, i.v0,
        road.y_left, road.y_right,
        mp.bb_tilde, mp.sy0b_tilde,
    )

    # 交通相互作用 — 式(12), (13)
    w0 = w0_desired(i, others, mp)
    g_int = (w0 - i.w) / mp.tau

    return g_bnd + g_int
