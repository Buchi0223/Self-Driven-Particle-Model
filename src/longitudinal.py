"""
MTM 縦方向ダイナミクスモジュール

車両間の縦方向加速度を計算する。
Car-Following モデルの加速度に横方向減衰 α を適用し、
最影響リーダー選択と道路境界力を組み合わせる。

全加速度 (式2):
    dv_i/dt = f_self + f_{il'} + Σ f_{ib}

    f_self  = a_CF_free(v_i)                        — 式(6)
    f_{il'} = α(Δy) · a_CF_int(Δx, v_i, v_l')      — 式(7)
    l'      = argmax_l |f_{il}|                      — 式(2)
    f_{ib}  = 境界力                                 — 式(9)

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2.1, Eq. (2), (6)-(9)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from src.cf_models import cf_free, cf_interaction
from src.geometry import delta_x, delta_y, W_bar, sy_gap, sx_gap

if TYPE_CHECKING:
    from src.vehicle import Vehicle
    from src.road import Road
    from src.parameters import MTMParams


# ---------------------------------------------------------------------------
# 横方向減衰因子 α — 式(8)
# ---------------------------------------------------------------------------

def alpha(dy: float, w_bar: float, sy0: float) -> float:
    """横方向減衰因子 α(Δy_{il}) — 式(8)

    横方向オーバーラップがある場合 (s^y < 0) は α = 1 (減衰なし)。
    横方向にクリアランスがある場合は指数減衰。

    Args:
        dy:   Δy_{il} = yl - yi [m]
        w_bar: 平均車幅 W̄_{il} [m]
        sy0:  横方向減衰スケール s^y_0 [m]

    Returns:
        α ∈ (0, 1]
    """
    sy = abs(dy) - w_bar  # 式(8): s^y_{il} = |Δy| - W̄
    if sy <= 0.0:
        return 1.0
    return math.exp(-sy / sy0)


# ---------------------------------------------------------------------------
# 単一リーダーからの縦方向相互作用力 — 式(7)
# ---------------------------------------------------------------------------

def f_interaction_single(
    xi: float, yi: float, vi: float, Wi: float,
    xl: float, yl: float, vl: float, Wl: float, Ll: float,
    v0: float, T: float, s0: float, a: float, b: float,
    delta: float, b_max: float, sy0: float,
) -> float:
    """単一リーダー l からの縦方向相互作用力 f_{il} — 式(7)

    f_{il} = α(Δy_{il}) · a_CF_int(gap, v_i, v_l)

    Args:
        xi, yi, vi, Wi: 対象車両 i の状態・車幅
        xl, yl, vl, Wl, Ll: リーダー l の状態・車幅・車長
        v0 ~ b_max: CF パラメータ (対象車両)
        sy0: 横方向減衰スケール [m]

    Returns:
        f_{il} [m/s^2]
    """
    dy = delta_y(yi, yl)
    w_bar = W_bar(Wi, Wl)
    att = alpha(dy, w_bar, sy0)

    gap = sx_gap(xi, xl, Ll)
    a_int = cf_interaction(gap, vi, vl, v0, T, s0, a, b, delta, b_max)

    return att * a_int


# ---------------------------------------------------------------------------
# 最影響リーダー選択 — 式(2)
# ---------------------------------------------------------------------------

def find_most_interacting_leader(
    vehicle_i: Vehicle,
    others: list[Vehicle],
    sy0: float,
) -> tuple[Vehicle | None, float]:
    """最影響リーダー l' を選択し、その相互作用力を返す — 式(2)

    l' = argmax_l |f_{il}|
    前方の車両 (xl > xi) のみを候補とする。

    Args:
        vehicle_i: 対象車両
        others:    他の車両リスト
        sy0:       横方向減衰スケール [m]

    Returns:
        (leader, f_il_value) — リーダーがいない場合は (None, 0.0)
    """
    best_leader = None
    best_f = 0.0
    best_abs_f = 0.0

    i = vehicle_i
    for other in others:
        if other.vehicle_id == i.vehicle_id:
            continue
        if other.x <= i.x:
            continue  # 前方の車両のみ

        f_il = f_interaction_single(
            i.x, i.y, i.v, i.width,
            other.x, other.y, other.v, other.width, other.length,
            i.v0, i.T, i.s0, i.a, i.b, i.delta, i.b_max, sy0,
        )

        if abs(f_il) > best_abs_f:
            best_abs_f = abs(f_il)
            best_f = f_il
            best_leader = other

    return best_leader, best_f


# ---------------------------------------------------------------------------
# 道路境界の縦方向力 — 式(9)
# ---------------------------------------------------------------------------

def f_boundary_longitudinal(
    vi: float, yi: float, Wi: float, v0: float,
    y_left: float, y_right: float,
    bb: float, sy0b: float,
) -> float:
    """道路境界からの縦方向減速力 — 式(9)

    f_{ib} = -b_b · (v_i / v0) · exp(-s^y_{ib} / s^y_{0b})

    s^y_{ib} = ±(y_b - y_i) - W_i/2
        + for right boundary, - for left boundary  (論文 p.5)

    両側の境界から力を受ける。v=0 のとき力は消失 (狭路低速通過を許容)。

    座標系: y は右方向に増加 (論文準拠)。y_left < y_right。

    Args:
        vi:      対象車両の速度 [m/s]
        yi:      対象車両の横方向位置 [m]
        Wi:      対象車両の車幅 [m]
        v0:      希望速度 [m/s]
        y_left:  左側境界 y 座標 [m]  (小さい値)
        y_right: 右側境界 y 座標 [m]  (大きい値)
        bb:      境界制動減速度 [m/s^2]
        sy0b:    境界減衰スケール [m]

    Returns:
        合計縦方向境界力 [m/s^2]  (常に ≤ 0)
    """
    if v0 <= 0.0 or vi <= 0.0:
        return 0.0

    speed_factor = vi / v0

    # 左側境界: s^y = -(y_left - yi) - Wi/2 = (yi - y_left) - Wi/2
    sy_left = (yi - y_left) - Wi / 2.0
    f_left = -bb * speed_factor * math.exp(-sy_left / sy0b)

    # 右側境界: s^y = +(y_right - yi) - Wi/2
    sy_right = (y_right - yi) - Wi / 2.0
    f_right = -bb * speed_factor * math.exp(-sy_right / sy0b)

    return f_left + f_right


# ---------------------------------------------------------------------------
# 全縦方向加速度 — 式(2)
# ---------------------------------------------------------------------------

def longitudinal_acceleration(
    vehicle_i: Vehicle,
    others: list[Vehicle],
    road: Road,
    mtm_params: MTMParams,
) -> float:
    """車両 i の全縦方向加速度 — 式(2)

    dv_i/dt = f_self + f_{il'} + Σ f_{ib}

    Args:
        vehicle_i:  対象車両
        others:     他の車両リスト
        road:       道路情報
        mtm_params: MTM パラメータ

    Returns:
        縦方向加速度 [m/s^2]
    """
    i = vehicle_i

    # 自己駆動力 — 式(6)
    f_self = cf_free(i.v, i.v0, i.a, i.delta)

    # 最影響リーダーとの相互作用力 — 式(7)
    _, f_leader = find_most_interacting_leader(i, others, mtm_params.sy0)

    # 道路境界力 — 式(9)
    f_bnd = f_boundary_longitudinal(
        i.v, i.y, i.width, i.v0,
        road.y_left, road.y_right,
        mtm_params.bb, mtm_params.sy0b,
    )

    return f_self + f_leader + f_bnd
