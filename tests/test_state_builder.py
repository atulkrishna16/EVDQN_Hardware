"""Unit tests for StateBuilder."""

from datetime import datetime
import numpy as np
import pytest
from live.state_builder import StateBuilder, get_tariff


@pytest.fixture
def sample_config():
    return {
        "timing": {
            "timezone": "Asia/Kolkata",
            "episode_length_steps": 24,
        },
        "grid_specs": {
            "battery": {"initial_soc": 50.0},
            "evs": [
                {"initial_soc": 30.0},
                {"initial_soc": 40.0},
                {"initial_soc": 25.0},
            ],
        },
    }


def test_tariff_schedule():
    # 00:00 - 05:59: 5.0
    assert get_tariff(0) == 5.0
    assert get_tariff(5) == 5.0
    # 06:00 - 08:59: 7.0
    assert get_tariff(6) == 7.0
    assert get_tariff(8) == 7.0
    # 09:00 - 11:59: 9.0
    assert get_tariff(9) == 9.0
    assert get_tariff(11) == 9.0
    # 12:00 - 17:59: 6.0
    assert get_tariff(12) == 6.0
    assert get_tariff(17) == 6.0
    # 18:00 - 21:59: 10.0
    assert get_tariff(18) == 10.0
    assert get_tariff(21) == 10.0
    # 22:00 - 23:59: 6.0
    assert get_tariff(22) == 6.0
    assert get_tariff(23) == 6.0


def test_state_vector_shape_and_types(sample_config):
    builder = StateBuilder(sample_config)
    snapshot = {
        "load_kw": 50.0,
        "next_load_kw": 55.0,
        "solar_kw": 30.0,
        "battery_soc": 75.0,
        "ev0_soc": 50.0,
        "ev1_soc": 60.0,
        "ev2_soc": 40.0,
    }
    fixed_time = datetime(2026, 3, 2, 10, 30, 0)  # Monday (weekday), 10 AM
    state, meta = builder.build_state(snapshot, current_time=fixed_time, step=5)

    assert isinstance(state, np.ndarray)
    assert state.shape == (12,)
    assert state.dtype == np.float32

    # Check components
    expected_sin = np.sin(2 * np.pi * 10 / 24.0)
    expected_cos = np.cos(2 * np.pi * 10 / 24.0)
    assert np.isclose(state[0], expected_sin, atol=1e-5)
    assert np.isclose(state[1], expected_cos, atol=1e-5)
    assert np.isclose(state[2], 50.0 / 100.0)  # load_kw / 100
    assert np.isclose(state[3], 55.0 / 100.0)  # next_load_kw / 100
    assert np.isclose(state[4], 30.0 / 60.0)   # solar_kw / 60
    assert np.isclose(state[5], 9.0 / 10.0)    # tariff / 10
    assert np.isclose(state[6], 75.0 / 100.0)  # bat SOC / 100
    assert np.isclose(state[7], 50.0 / 100.0)  # EV0 SOC / 100
    assert np.isclose(state[8], 60.0 / 100.0)  # EV1 SOC / 100
    assert np.isclose(state[9], 40.0 / 100.0)  # EV2 SOC / 100
    assert state[10] == 0.0                    # Monday = weekday (0.0)
    assert np.isclose(state[11], 5.0 / 23.0)   # progress


def test_fallback_on_missing_values(sample_config):
    builder = StateBuilder(sample_config)
    empty_snapshot = {}
    state, meta = builder.build_state(empty_snapshot, current_time=datetime(2026, 3, 7, 14, 0, 0))  # Saturday

    assert state.shape == (12,)
    assert state[10] == 1.0  # Saturday = weekend (1.0)
    # Default battery SOC was 50.0% -> 0.5
    assert np.isclose(state[6], 0.5)
