"""
パラメータ定義モジュール

論文 Table 1 (車種別 Car-Following パラメータ) と
Table 2 (MTM 固有パラメータ) を定義する。

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 4, Tables 1-2
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Car-Following (CF) モデルパラメータ — 論文 Table 1
# ---------------------------------------------------------------------------
# 基盤モデル: ACC (modified IDM), Kesting et al. (2010)
# 全車種共通の最大制動減速度
B_MAX = 9.0  # [m/s^2]  — 論文 Table 1 caption


@dataclass(frozen=True)
class CFParams:
    """車種別の Car-Following モデルパラメータ (Table 1)

    Attributes:
        vehicle_type: 車種名
        length:       車両長 [m]
        width:        車両幅 [m]
        v0:           希望速度 [m/s]  (Table 1 の range から代表値を選択)
        T:            安全車頭時間 [s]
        s0:           最小車間距離 [m]
        a:            最大加速度 [m/s^2]
        b:            快適制動減速度 [m/s^2]
        b_max:        最大制動減速度 [m/s^2]
        delta:        加速指数 [-]  (IDM 標準値 = 4)
        coolness:     ACC coolness factor [-]  (Kesting et al. 2010)
    """
    vehicle_type: str
    length: float
    width: float
    v0: float
    T: float
    s0: float
    a: float
    b: float
    b_max: float = B_MAX
    delta: float = 4.0
    coolness: float = 0.99


# --- 車種別パラメータインスタンス (Table 1) ---
# v0: Table 1 の "range" 列から代表値を採用
#   Motorcycle:     25-18 → 25 m/s (free flow 上限)
#   Car:            18-12 → 18 m/s
#   Bus:            14-10 → 14 m/s
#   Auto-Rickshaw:   5-6  →  6 m/s

MOTORCYCLE = CFParams(
    vehicle_type="Motorcycle",
    length=1.8,
    width=0.6,
    v0=25.0,
    T=0.3,
    s0=0.5,
    a=2.0,
    b=2.0,
)

CAR = CFParams(
    vehicle_type="Car",
    length=4.2,
    width=1.7,
    v0=18.0,
    T=0.8,
    s0=2.0,
    a=1.0,
    b=1.0,
)

BUS = CFParams(
    vehicle_type="Bus",
    length=10.3,
    width=2.1,
    v0=14.0,
    T=1.0,
    s0=2.0,
    a=1.0,
    b=1.0,
)

AUTO_RICKSHAW = CFParams(
    vehicle_type="Auto-Rickshaw",
    length=2.6,
    width=0.9,
    v0=6.0,
    T=1.0,
    s0=2.0,
    a=1.0,
    b=1.0,
)

# 全車種をまとめた辞書 (車種名 → パラメータ)
CF_PARAMS_BY_TYPE = {
    "Motorcycle": MOTORCYCLE,
    "Car": CAR,
    "Bus": BUS,
    "Auto-Rickshaw": AUTO_RICKSHAW,
}


# ---------------------------------------------------------------------------
# MTM (Mixed Traffic Flow Model) 固有パラメータ — 論文 Table 2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MTMParams:
    """MTM モデル固有パラメータ (Table 2)

    Attributes:
        theta:      最大道路軸角度 [rad]          — 車両進行方向の許容偏角
        sy0:        横方向減衰スケール(制動用) [m] — 式(8) の s^y_0
        sy0_tilde:  横方向減衰スケール(操舵用) [m] — 式(16) の s̃^y_0
        sy0b:       境界減衰スケール(制動用) [m]   — 式(9) の s^y_{0b}
        sy0b_tilde: 境界減衰スケール(操舵用) [m]   — 式(11) の s̃^y_{0b}
        lam:        横方向感度 [s]                 — 式(15) の λ
        tau:        横方向時定数 [s]               — 式(12) の τ
        lam_dw:     横方向相対速度感度 [s/m]       — 式(15) の λ_{Δw}
        p:          politeness factor [-]          — follower 考慮の重み
        bb:         境界制動減速度 [m/s^2]         — 式(9) の b_b  (論文未明記, 推定値)
        bb_tilde:   境界横方向加速度 [m/s^2]       — 式(11) の b̃_b (論文未明記, 推定値)
        a_thr:      相互作用閾値 [m/s^2]           — 近傍車両フィルタリング用
    """
    theta: float = 0.2
    sy0: float = 0.15
    sy0_tilde: float = 0.30
    sy0b: float = 0.15
    sy0b_tilde: float = 0.25
    lam: float = 0.4
    tau: float = 1.0
    lam_dw: float = 0.7
    p: float = 0.2
    bb: float = 3.0       # 論文未明記 — 推定値 (要チューニング)
    bb_tilde: float = 5.0  # 論文未明記 — 推定値 (要チューニング)
    a_thr: float = 0.01   # プレ検証用: ほぼ全近傍を含む


# デフォルトの MTM パラメータインスタンス
DEFAULT_MTM_PARAMS = MTMParams()
