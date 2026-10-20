"""Unit tests for LCD text formatting and screen rotation."""

import pytest
from live.lcd import LCDRenderer


@pytest.fixture
def mock_lcd_20x4():
    config = {
        "lcd": {
            "mock": True,
            "cols": 20,
            "rows": 4,
            "rotation_interval_seconds": 1.0,
        }
    }
    return LCDRenderer(config, mock=True)


@pytest.fixture
def mock_lcd_16x2():
    config = {
        "lcd": {
            "mock": True,
            "cols": 16,
            "rows": 2,
            "rotation_interval_seconds": 1.0,
        }
    }
    return LCDRenderer(config, mock=True)


def test_20x4_power_split_screen(mock_lcd_20x4):
    sample_dispatch = {
        "grid_import_kw": 42.0,
        "grid_pct": 60.0,
        "solar_used_kw": 18.0,
        "solar_pct": 25.0,
        "battery_mode": "DISCHARGE",
        "battery_discharge_kw": 10.5,
        "battery_pct": 15.0,
        "battery_soc_pct": 63.0,
        "tariff_rate": 9.0,
    }

    lines = mock_lcd_20x4._generate_power_split_screen(sample_dispatch)
    assert len(lines) == 4
    for line in lines:
        assert len(line) == 20

    assert "GRID   42.0kW  60%" in lines[0]
    assert "SOLAR  18.0kW  25%" in lines[1]
    assert "BATT   10.5kW  15%" in lines[2]
    assert "T:PEAK  SOC: 63%" in lines[3]


def test_20x4_status_screen(mock_lcd_20x4):
    sample_dispatch = {
        "tariff_rate": 6.5,
        "battery_soc_pct": 63.0,
        "battery_mode": "DISCHARGE",
        "ev_socs_pct": [45.0, 60.0, 30.0],
    }

    lines = mock_lcd_20x4._generate_status_screen(sample_dispatch, is_connected=True, age_seconds=0.8)
    assert len(lines) == 4
    for line in lines:
        assert len(line) == 20

    assert "TARIFF: Rs6.5 (MED)" in lines[0]
    assert "BAT:  63% (DISCHARG)" in lines[1]
    assert "EV: 45%/60%/30%" in lines[2]
    assert "MQTT:OK  AGE: 0.8s" in lines[3]


def test_16x2_screens_fit_grid(mock_lcd_16x2):
    sample_dispatch = {
        "grid_import_kw": 25.0,
        "solar_used_kw": 15.0,
        "battery_mode": "IDLE",
        "battery_discharge_kw": 0.0,
        "battery_soc_pct": 50.0,
        "tariff_rate": 5.0,
        "ev_socs_pct": [30.0, 40.0, 25.0],
    }

    lines1 = mock_lcd_16x2._generate_power_split_screen(sample_dispatch)
    assert len(lines1) == 2
    for line in lines1:
        assert len(line) == 16

    lines2 = mock_lcd_16x2._generate_status_screen(sample_dispatch, is_connected=True, age_seconds=1.2)
    assert len(lines2) == 2
    for line in lines2:
        assert len(line) == 16


def test_no_data_alert_screen(mock_lcd_20x4):
    lines = mock_lcd_20x4._generate_no_data_screen(is_connected=False, age_seconds=20.0)
    assert len(lines) == 4
    for line in lines:
        assert len(line) == 20
    assert "*** SYSTEM ALERT ***" in lines[0]
    assert "SAFE GRID-ONLY MODE" in lines[2]
