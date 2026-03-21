"""
Car-Following モデルモジュール

IDM (Intelligent Driver Model) と ACC (Adaptive Cruise Control) モデルを実装し、
MTM で必要な free/interaction 分解 (式3-5) を提供する。

IDM:
    a_CF(s, v, vl) = a * [1 - (v/v0)^δ - (s*(v, Δv) / s)^2]
    s*(v, Δv) = s0 + max(0, v*T + v*Δv / (2*sqrt(a*b)))

ACC (modified IDM with coolness):
    IDM を基本とし、CAH (Constant Acceleration Heuristic) と
    coolness factor c でブレンドすることで、カットイン反応を緩和する。
    Kesting et al. (2010), ref [11]

Free/Interaction 分解 (式3-5):
    a_CF_free(v)       = a_CF(∞, v, v)           — 式(4)
    a_CF_int(s, v, vl) = a_CF(s, v, vl) - a_CF_free(v)  — 式(5)

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2.1, Eq. (3)-(5)
    Treiber et al. (2000) — IDM
    Kesting et al. (2010) — ACC
"""

import math


# ---------------------------------------------------------------------------
# IDM (Intelligent Driver Model) — Treiber et al. (2000)
# ---------------------------------------------------------------------------

def idm_desired_gap(v: float, dv: float, s0: float, T: float,
                    a: float, b: float) -> float:
    """IDM の希望車間距離 s*(v, Δv)

    Args:
        v:  対象車両の速度 [m/s]
        dv: 速度差 v_i - v_l [m/s]  (正: 対象車両が速い)
        s0: 最小車間距離 [m]
        T:  安全車頭時間 [s]
        a:  最大加速度 [m/s^2]
        b:  快適制動減速度 [m/s^2]

    Returns:
        s* [m]
    """
    interaction_term = v * dv / (2.0 * math.sqrt(a * b))
    return s0 + max(0.0, v * T + interaction_term)


def idm_acceleration(s: float, v: float, vl: float,
                     v0: float, T: float, s0: float,
                     a: float, b: float, delta: float = 4.0,
                     b_max: float = 9.0) -> float:
    """IDM の全加速度 a_CF(s, v, vl)

    Args:
        s:     縦方向距離 Δx_{il} = xl - xi [m]  (front-center 間距離, 車両長含む)
        v:     対象車両の速度 [m/s]
        vl:    リーダーの速度 [m/s]
        v0:    希望速度 [m/s]
        T:     安全車頭時間 [s]
        s0:    最小車間距離 [m]
        a:     最大加速度 [m/s^2]
        b:     快適制動減速度 [m/s^2]
        delta: 加速指数 [-]  (デフォルト 4)
        b_max: 最大制動減速度 [m/s^2]

    Returns:
        加速度 [m/s^2]

    Note:
        s はリーダーの車両長を含む front-center 間距離。
        gap (バンパー間) = s - Ll だが、IDM は内部で s* と s を比較するため、
        s にはリーダー長 Ll を引いた net gap を渡す必要がある。
        呼び出し側で gap = sx_gap(xi, xl, Ll) を計算して渡すこと。

        gap <= 0 の場合は -b_max を返す (論文 式(8) 下の記述)。
    """
    # エッジケース: 縦方向オーバーラップ → 緊急ブレーキ
    if s <= 0.0:
        return -b_max

    dv = v - vl
    s_star = idm_desired_gap(v, dv, s0, T, a, b)

    # IDM 加速度: a * [1 - (v/v0)^δ - (s*/s)^2]
    free_term = (v / v0) ** delta if v0 > 0.0 else 0.0
    interaction_term = (s_star / s) ** 2

    # クランプ: 正の gap でも -b_max を下回らない
    # (gap→0+ で (s*/s)^2 が発散するため)
    return max(a * (1.0 - free_term - interaction_term), -b_max)


# ---------------------------------------------------------------------------
# ACC (Adaptive Cruise Control) — Kesting et al. (2010)
# ---------------------------------------------------------------------------

def _cah_acceleration(s: float, v: float, vl: float,
                      al: float = 0.0) -> float:
    """CAH (Constant Acceleration Heuristic) — ACC の補助関数

    リーダーが一定加速度 al で走行すると仮定した場合の
    衝突回避に必要な最小加速度。

    Args:
        s:  net gap (バンパー間距離) [m]
        v:  対象車両の速度 [m/s]
        vl: リーダーの速度 [m/s]
        al: リーダーの加速度 [m/s^2]  (デフォルト 0)

    Returns:
        a_CAH [m/s^2]
    """
    if s <= 0.0:
        return -1e6  # 大きな負の値 → IDM 側が採用される

    dv = v - vl

    # vl * dv が十分大きいケースと小さいケースで場合分け
    # Kesting et al. (2010) Eq. (11.44) in Treiber & Kesting (2013)
    if vl * dv <= -2.0 * s * al:
        # リーダーが十分減速中で衝突リスク低
        a_cah = (v * v * al) / (vl * vl - 2.0 * s * al) if (vl * vl - 2.0 * s * al) != 0.0 else 0.0
    else:
        a_cah = al - dv * dv / (2.0 * s) if s > 0.0 else -1e6

    return a_cah


def acc_acceleration(s: float, v: float, vl: float,
                     v0: float, T: float, s0: float,
                     a: float, b: float, delta: float = 4.0,
                     b_max: float = 9.0, coolness: float = 0.99,
                     al: float = 0.0) -> float:
    """ACC モデルの全加速度

    IDM をベースに、CAH と coolness factor c でブレンドすることで
    カットインへの過剰反応を抑制する。

    a_ACC = a_IDM                                       if a_IDM >= a_CAH
          = (1-c)*a_IDM + c*(a_CAH + b*tanh((a_IDM - a_CAH)/b))  otherwise

    Args:
        s ~ b_max: IDM と同じ
        coolness: ACC coolness factor [0, 1]
        al:       リーダーの加速度 [m/s^2]

    Returns:
        加速度 [m/s^2]
    """
    a_idm = idm_acceleration(s, v, vl, v0, T, s0, a, b, delta, b_max)

    if coolness == 0.0:
        return a_idm

    a_cah = _cah_acceleration(s, v, vl, al)

    if a_idm >= a_cah:
        return a_idm

    return (1.0 - coolness) * a_idm + coolness * (
        a_cah + b * math.tanh((a_idm - a_cah) / b)
    )


# ---------------------------------------------------------------------------
# Free/Interaction 分解 — 式(3)-(5)
# ---------------------------------------------------------------------------

def cf_free(v: float, v0: float, a: float, delta: float = 4.0) -> float:
    """自由加速度 a_CF_free(v) — 式(4)

    リーダーが無限遠にいる (または存在しない) 場合の加速度。
    a_CF(∞, v, v) = a * [1 - (v/v0)^δ]

    Args:
        v:     対象車両の速度 [m/s]
        v0:    希望速度 [m/s]
        a:     最大加速度 [m/s^2]
        delta: 加速指数 [-]

    Returns:
        自由加速度 [m/s^2]
    """
    if v0 <= 0.0:
        return 0.0
    return a * (1.0 - (v / v0) ** delta)


def cf_interaction(s: float, v: float, vl: float,
                   v0: float, T: float, s0: float,
                   a: float, b: float, delta: float = 4.0,
                   b_max: float = 9.0,
                   coolness: float = 0.0) -> float:
    """相互作用加速度 a_CF_int(s, v, vl) — 式(5)

    a_CF_int = a_CF(s, v, vl) - a_CF_free(v)

    リーダーとの相互作用による加速度成分。
    gap が大きいとき → 0 に収束 (式(5) 下の記述)。

    coolness > 0 の場合、基盤 CF モデルとして ACC を使用する。
    coolness = 0 の場合は IDM を使用する (後方互換)。

    Args:
        s:  net gap (バンパー間距離) [m]
        v ~ b_max: CF モデルパラメータ
        coolness: ACC coolness factor (0=IDM, >0=ACC)

    Returns:
        相互作用加速度 [m/s^2]  (通常は負: 減速方向)
    """
    if coolness > 0.0:
        a_total = acc_acceleration(s, v, vl, v0, T, s0, a, b, delta, b_max, coolness)
    else:
        a_total = idm_acceleration(s, v, vl, v0, T, s0, a, b, delta, b_max)
    a_free = cf_free(v, v0, a, delta)
    return a_total - a_free
