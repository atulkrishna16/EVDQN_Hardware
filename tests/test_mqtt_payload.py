"""Unit tests for MQTT payload validation and thread-safe store."""

import math
import time
import pytest
from live.mqtt_client import SensorStateStore


def test_payload_validation_bounds():
    # Valid bounds
    ok, val = SensorStateStore.validate_field("load_kw", 45.2)
    assert ok and val == 45.2

    ok, val = SensorStateStore.validate_field("battery_soc", 88.5)
    assert ok and val == 88.5

    # Negative load rejected
    ok, val = SensorStateStore.validate_field("load_kw", -5.0)
    assert not ok

    # Unreasonable load (>500 kW) rejected
    ok, val = SensorStateStore.validate_field("load_kw", 1200.0)
    assert not ok

    # Invalid SOC (< 0 or > 100) rejected
    ok, val = SensorStateStore.validate_field("battery_soc", 105.0)
    assert not ok
    ok, val = SensorStateStore.validate_field("ev0_soc", -1.0)
    assert not ok

    # Non-numeric or NaN / Inf rejected
    ok, val = SensorStateStore.validate_field("load_kw", float("nan"))
    assert not ok
    ok, val = SensorStateStore.validate_field("load_kw", float("inf"))
    assert not ok
    ok, val = SensorStateStore.validate_field("load_kw", "abc")
    assert not ok


def test_store_staleness():
    store = SensorStateStore(stale_timeout_seconds=0.5)

    # Initial state is empty and not fresh
    snap, is_fresh, age = store.get_snapshot()
    assert not is_fresh

    # Update with fresh data
    store.update_from_dict({"load_kw": 30.0, "battery_soc": 60.0})
    snap, is_fresh, age = store.get_snapshot()
    assert is_fresh
    assert snap["load_kw"] == 30.0
    assert snap["battery_soc"] == 60.0

    # Sleep past timeout
    time.sleep(0.6)
    snap, is_fresh, age = store.get_snapshot()
    assert not is_fresh
    assert age >= 0.5
