"""
幾何計算ヘルパーモジュール

論文 Fig. 2 に定義される車両間の幾何量を計算する純粋関数群。
全関数はステートレスで副作用なし。

座標系:
    x: 縦方向 (道路進行方向, 正が前方)
    y: 横方向 (正が左方向)
    位置は車両の front-center を基準

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2, Fig. 2, Eq. (8)
"""


def delta_x(xi: float, xl: float) -> float:
    """縦方向距離 (front-center 間)

    Args:
        xi: 対象車両 i の縦方向位置 [m]
        xl: リーダー l の縦方向位置 [m]

    Returns:
        Δx_{il} = xl - xi [m]  (正: リーダーが前方)
    """
    return xl - xi


def delta_y(yi: float, yl: float) -> float:
    """横方向変位 (符号付き)

    Args:
        yi: 対象車両 i の横方向位置 [m]
        yl: リーダー l の横方向位置 [m]

    Returns:
        Δy_{il} = yl - yi [m]  (正: リーダーが左側)
    """
    return yl - yi


def W_bar(Wi: float, Wl: float) -> float:
    """平均車幅 — 式(8) の W̄_{il}

    Args:
        Wi: 対象車両の車幅 [m]
        Wl: リーダーの車幅 [m]

    Returns:
        (Wi + Wl) / 2 [m]
    """
    return (Wi + Wl) / 2.0


def sy_gap(dy: float, w_bar: float) -> float:
    """横方向ギャップ — 式(8) の s^y_{il}

    横方向の車体間クリアランス。
    負値 = 横方向オーバーラップ (車体が重なる投影あり)
    正値 = 横方向にクリアランスあり

    Args:
        dy:    Δy_{il} = yl - yi [m]  (delta_y() の戻り値)
        w_bar: 平均車幅 W̄_{il} [m]    (W_bar() の戻り値)

    Returns:
        s^y_{il} = |Δy_{il}| - W̄_{il} [m]
    """
    return abs(dy) - w_bar


def sx_gap(xi: float, xl: float, Ll: float) -> float:
    """縦方向ギャップ (バンパー間距離)

    Args:
        xi: 対象車両 i の縦方向位置 (front-center) [m]
        xl: リーダー l の縦方向位置 (front-center) [m]
        Ll: リーダー l の車両長 [m]

    Returns:
        s_{il} = xl - xi - Ll [m]  (負: 車体が縦方向で重なっている)

    Note:
        論文では Δx_{il} = xl - xi を距離、s_{il} = Δx_{il} - Ll をギャップと
        定義している。CF モデルへの入力は Δx_{il} (= xl - xi) であり、
        CF モデル内部で Ll を引いてギャップを得る。
        この関数は検証・デバッグ用に直接ギャップを返す。
    """
    return xl - xi - Ll
