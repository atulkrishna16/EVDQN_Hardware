"""Unit tests for EnergyDispatcher."""

import pytest
from live.dispatcher import EnergyDispatcher


@pytest.fixture
def dispatcher():
    config = {
        "timing": {"dt_hours": 1.0},
        "grid_specs": {
            "grid_limit_kw": 100.0,
            "battery": {
                "capacity_kwh": 100.0,
                "initial_soc": 50.0,
                "min_soc": 20.0,
                "max_soc": 95.0,
                "max_charge_kw": 25.0,
                "max_discharge_kw": 25.0,
                "efficiency": 0.95,
            },
            "evs": [
                {"capacity_kwh": 50.0, "initial_soc": 30.0, "target_soc": 80.0, "max_charge_kw": 7.0, "efficiency": 0.95},
                {"capacity_kwh": 50.0, "initial_soc": 40.0, "target_soc": 90.0, "max_charge_kw": 7.0, "efficiency": 0.95},
                {"capacity_kwh": 50.0, "initial_soc": 25.0, "target_soc": 80.0, "max_charge_kw": 7.0, "efficiency": 0.95},
            ],
        },
    }
    return EnergyDispatcher(config)


def test_action_decoding(dispatcher):
    # Action 0: battery strong discharge (-25 kW), no EVs charging (bits 0)
    pwr, evs = dispatcher.decode_action(0)
    assert pwr == -25.0
    assert evs == [False, False, False]

    # Action 7: battery strong discharge (-25 kW), all 3 EVs charging (bits 111 = 7)
    pwr, evs = dispatcher.decode_action(7)
    assert pwr == -25.0
    assert evs == [True, True, True]

    # Action 16: battery idle (0.0 kW), no EVs (index 2*8=16)
    pwr, evs = dispatcher.decode_action(16)
    assert pwr == 0.0
    assert evs == [False, False, False]

    # Action 39: battery strong charge (+25 kW), all 3 EVs (index 4*8 + 7 = 39)
    pwr, evs = dispatcher.decode_action(39)
    assert pwr == 25.0
    assert evs == [True, True, True]

    # Invalid action
    with pytest.raises(ValueError):
        dispatcher.decode_action(40)


def test_power_balance_and_dispatch(dispatcher):
    # Action 16: idle battery, no EVs
    res = dispatcher.dispatch(
        action=16,
        load_kw=40.0,
        solar_kw=15.0,
        tariff_rate=6.0,
        battery_soc=50.0,
        ev_socs=[50.0, 50.0, 50.0],
    )

    # Solar used should be 15 kW, grid import should be 40 - 15 = 25 kW
    assert res["solar_used_kw"] == 15.0
    assert res["solar_curtailed_kw"] == 0.0
    assert res["grid_import_kw"] == 25.0
    assert res["battery_discharge_kw"] == 0.0
    assert res["battery_charge_kw"] == 0.0
    assert res["overload_kw"] == 0.0
    # Cost = 25 kW * 6.0 = 150.0
    assert res["grid_cost_inr"] == 150.0

    # Supply must equal demand
    assert pytest.approx(res["total_demand_kw"]) == 40.0
    assert pytest.approx(res["total_supply_kw"]) == 40.0


def test_battery_discharging_supply(dispatcher):
    # Action 0: strong discharge (-25 kW requested)
    res = dispatcher.dispatch(
        action=0,
        load_kw=50.0,
        solar_kw=10.0,
        tariff_rate=9.0,
        battery_soc=60.0,  # Plenty of energy
    )

    # Max discharge is 25 kW
    assert res["battery_discharge_kw"] == 25.0
    assert res["solar_used_kw"] == 10.0
    # Net grid import: 50 - 10 - 25 = 15 kW
    assert res["grid_import_kw"] == 15.0
    assert res["battery_mode"] == "DISCHARGE"


def test_grid_overload_cap(dispatcher):
    # Action 39: strong charge (25 kW) + 3 EVs (3 * 7 = 21 kW) + load 80 kW = 126 kW
    res = dispatcher.dispatch(
        action=39,
        load_kw=80.0,
        solar_kw=0.0,
        tariff_rate=10.0,
        battery_soc=30.0,
        ev_socs=[30.0, 30.0, 30.0],
    )

    # Total grid import exceeds 100 kW limit
    assert res["grid_import_kw"] > 100.0
    assert res["overload_kw"] == pytest.approx(res["grid_import_kw"] - 100.0)
