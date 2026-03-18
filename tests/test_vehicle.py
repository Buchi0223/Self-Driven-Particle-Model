"""
vehicle モジュールのテスト

Vehicle クラスの生成・属性アクセス・ファクトリメソッドを検証する。
"""

import pytest
from src.parameters import CAR, MOTORCYCLE, BUS, AUTO_RICKSHAW
from src.vehicle import Vehicle


# ---------------------------------------------------------------------------
# 直接コンストラクタ
# ---------------------------------------------------------------------------

class TestVehicleConstructor:
    """Vehicle の直接生成テスト"""

    def test_default_values(self):
        """デフォルトは Car 相当"""
        v = Vehicle()
        assert v.x == 0.0
        assert v.y == 0.0
        assert v.v == 0.0
        assert v.w == 0.0
        assert v.vehicle_type == "Car"

    def test_custom_state(self):
        v = Vehicle(x=100.0, y=3.5, v=15.0, w=0.5)
        assert v.x == 100.0
        assert v.y == 3.5
        assert v.v == 15.0
        assert v.w == 0.5

    def test_state_is_mutable(self):
        """動的状態はシミュレーション中に更新されるため mutable"""
        v = Vehicle()
        v.x = 50.0
        v.v = 10.0
        v.w = -0.3
        assert v.x == 50.0
        assert v.v == 10.0
        assert v.w == -0.3


# ---------------------------------------------------------------------------
# from_cf_params ファクトリ
# ---------------------------------------------------------------------------

class TestFromCFParams:
    """CFParams からの Vehicle 生成テスト"""

    def test_car_from_params(self):
        v = Vehicle.from_cf_params(CAR, x=10.0, y=5.0, v=12.0)
        assert v.length == CAR.length
        assert v.width == CAR.width
        assert v.v0 == CAR.v0
        assert v.T == CAR.T
        assert v.s0 == CAR.s0
        assert v.a == CAR.a
        assert v.b == CAR.b
        assert v.b_max == CAR.b_max
        assert v.delta == CAR.delta
        assert v.vehicle_type == "Car"
        assert v.x == 10.0
        assert v.y == 5.0
        assert v.v == 12.0
        assert v.w == 0.0  # デフォルト

    def test_motorcycle_from_params(self):
        v = Vehicle.from_cf_params(MOTORCYCLE, vehicle_id=42)
        assert v.length == 1.8
        assert v.width == 0.6
        assert v.v0 == 25.0
        assert v.vehicle_type == "Motorcycle"
        assert v.vehicle_id == 42

    def test_bus_from_params(self):
        v = Vehicle.from_cf_params(BUS)
        assert v.length == 10.3
        assert v.width == 2.1
        assert v.v0 == 14.0

    def test_auto_rickshaw_from_params(self):
        v = Vehicle.from_cf_params(AUTO_RICKSHAW)
        assert v.length == 2.6
        assert v.width == 0.9
        assert v.v0 == 6.0


# ---------------------------------------------------------------------------
# create ファクトリ (車種名指定)
# ---------------------------------------------------------------------------

class TestCreate:
    """車種名から Vehicle を生成するテスト"""

    def test_create_car(self):
        v = Vehicle.create("Car", x=50.0, v=18.0)
        assert v.vehicle_type == "Car"
        assert v.length == CAR.length
        assert v.x == 50.0
        assert v.v == 18.0

    def test_create_motorcycle(self):
        v = Vehicle.create("Motorcycle", y=2.0, vehicle_id=7)
        assert v.vehicle_type == "Motorcycle"
        assert v.y == 2.0
        assert v.vehicle_id == 7

    def test_create_bus(self):
        v = Vehicle.create("Bus")
        assert v.vehicle_type == "Bus"
        assert v.length == BUS.length

    def test_create_auto_rickshaw(self):
        v = Vehicle.create("Auto-Rickshaw")
        assert v.vehicle_type == "Auto-Rickshaw"

    def test_create_unknown_type_raises(self):
        """未知の車種名で KeyError"""
        with pytest.raises(KeyError):
            Vehicle.create("Truck")

    def test_create_default_is_car(self):
        v = Vehicle.create()
        assert v.vehicle_type == "Car"


# ---------------------------------------------------------------------------
# 車種間の属性差異
# ---------------------------------------------------------------------------

class TestVehicleTypeDifferences:
    """各車種で物理属性・CF パラメータが異なることを確認"""

    def test_dimensions_differ(self):
        moto = Vehicle.create("Motorcycle")
        car = Vehicle.create("Car")
        bus = Vehicle.create("Bus")
        # Motorcycle < Car < Bus (長さ)
        assert moto.length < car.length < bus.length
        # Motorcycle < Auto-Rickshaw < Car < Bus (幅)
        assert moto.width < car.width < bus.width

    def test_desired_speed_differs(self):
        moto = Vehicle.create("Motorcycle")
        car = Vehicle.create("Car")
        rickshaw = Vehicle.create("Auto-Rickshaw")
        # Motorcycle が最も速い, Auto-Rickshaw が最も遅い
        assert moto.v0 > car.v0 > rickshaw.v0

    def test_all_share_b_max(self):
        """全車種で b_max = 9.0"""
        for vtype in ["Motorcycle", "Car", "Bus", "Auto-Rickshaw"]:
            v = Vehicle.create(vtype)
            assert v.b_max == 9.0
