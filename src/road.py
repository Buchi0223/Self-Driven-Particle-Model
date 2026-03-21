"""
道路定義モジュール

道路境界の幾何情報を保持する。

座標系 (論文準拠):
    x: 縦方向 (道路進行方向)
    y: 横方向 (正が右方向, "y is increasing to the right" — 論文 p.6)
    y_left (左端, 小さい値) ~ y_right (右端, 大きい値)
    デフォルト: y_left=0, y_right=12 (チェンナイ観測に対応, 論文 Section 4)

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 4, Fig. 2
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Road:
    """道路の幾何情報

    Attributes:
        y_left:      左側境界の y 座標 [m]  (小さい値)
        y_right:     右側境界の y 座標 [m]  (大きい値, y_right > y_left)
        road_length: シミュレーション領域の縦方向長さ [m]
    """
    y_left: float = 0.0
    y_right: float = 12.0
    road_length: float = 500.0

    @property
    def width(self) -> float:
        """道路幅 [m]"""
        return self.y_right - self.y_left

    @property
    def center(self) -> float:
        """道路中心の y 座標 [m]"""
        return (self.y_left + self.y_right) / 2.0
