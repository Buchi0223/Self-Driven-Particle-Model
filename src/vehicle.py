"""
車両状態モジュール

各車両の動的状態（位置・速度）と物理属性（寸法）、
Car-Following パラメータを保持するクラスを定義する。

座標系 (論文 Fig. 2):
    x: 縦方向 (道路進行方向, 正が前方)
    y: 横方向 (正が左方向)
    位置は車両の front-center を基準とする

Reference:
    Kanagaraj & Treiber (2018), arXiv:1805.05076
    Section 2, Fig. 2
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.parameters import CFParams, CF_PARAMS_BY_TYPE, CAR


@dataclass
class Vehicle:
    """車両の状態と属性

    動的状態 (シミュレーション中に更新される):
        x:  縦方向位置 [m]  — front-center
        y:  横方向位置 [m]  — front-center
        v:  縦方向速度 [m/s]
        w:  横方向速度 [m/s]

    物理属性 (不変):
        length: 車両長 [m]
        width:  車両幅 [m]

    CF パラメータ (不変, CFParams から展開):
        v0:    希望速度 [m/s]
        T:     安全車頭時間 [s]
        s0:    最小車間距離 [m]
        a:     最大加速度 [m/s^2]
        b:     快適制動減速度 [m/s^2]
        b_max: 最大制動減速度 [m/s^2]
        delta: 加速指数 [-]

    メタ情報:
        vehicle_type: 車種名 (可視化・デバッグ用)
        vehicle_id:   車両識別子
    """

    # --- 動的状態 ---
    x: float = 0.0
    y: float = 0.0
    v: float = 0.0
    w: float = 0.0

    # --- 物理属性 ---
    length: float = 4.2
    width: float = 1.7

    # --- CF パラメータ ---
    v0: float = 18.0
    T: float = 0.8
    s0: float = 2.0
    a: float = 1.0
    b: float = 1.0
    b_max: float = 9.0
    delta: float = 4.0

    # --- メタ情報 ---
    vehicle_type: str = "Car"
    vehicle_id: int = 0

    @classmethod
    def from_cf_params(
        cls,
        cf_params: CFParams,
        x: float = 0.0,
        y: float = 0.0,
        v: float = 0.0,
        w: float = 0.0,
        vehicle_id: int = 0,
    ) -> Vehicle:
        """CFParams から車両を生成するファクトリメソッド

        Args:
            cf_params:  車種別パラメータ (parameters.py で定義)
            x, y:       初期位置 [m]
            v, w:       初期速度 [m/s]
            vehicle_id: 車両識別子

        Returns:
            Vehicle インスタンス
        """
        return cls(
            x=x,
            y=y,
            v=v,
            w=w,
            length=cf_params.length,
            width=cf_params.width,
            v0=cf_params.v0,
            T=cf_params.T,
            s0=cf_params.s0,
            a=cf_params.a,
            b=cf_params.b,
            b_max=cf_params.b_max,
            delta=cf_params.delta,
            vehicle_type=cf_params.vehicle_type,
            vehicle_id=vehicle_id,
        )

    @classmethod
    def create(
        cls,
        vehicle_type: str = "Car",
        x: float = 0.0,
        y: float = 0.0,
        v: float = 0.0,
        w: float = 0.0,
        vehicle_id: int = 0,
    ) -> Vehicle:
        """車種名を指定して車両を生成するファクトリメソッド

        Args:
            vehicle_type: "Motorcycle", "Car", "Bus", "Auto-Rickshaw"
            x, y:         初期位置 [m]
            v, w:         初期速度 [m/s]
            vehicle_id:   車両識別子

        Returns:
            Vehicle インスタンス

        Raises:
            KeyError: 未知の車種名が指定された場合
        """
        cf_params = CF_PARAMS_BY_TYPE[vehicle_type]
        return cls.from_cf_params(
            cf_params, x=x, y=y, v=v, w=w, vehicle_id=vehicle_id
        )
