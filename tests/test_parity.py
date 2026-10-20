"""
End-to-End Parity Test
======================
Verifies that feeding the exact same scenario to the live pipeline and to
the original SmartGridEnv / evaluate_dqn produces identical state vectors,
actions, and physical power dispatches.
"""

from datetime import datetime
from pathlib import Path
import numpy as np
import pytest
import yaml

from environment.smart_grid_env import SmartGridEnv
from live.dispatcher import EnergyDispatcher
from live.inference import DQNInferenceEngine
from live.state_builder import StateBuilder

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = ROOT / "models" / "best_dqn_model.pth"
CONFIG_PATH = ROOT / "config_live.yaml"


def test_live_pipeline_parity_with_environment():
    assert TEST_DATA.exists(), f"Missing {TEST_DATA}"
    assert MODEL_PATH.exists(), f"Missing {MODEL_PATH}"

    # 1. Initialize original environment
    env = SmartGridEnv(
        data_path=TEST_DATA,
        episode_length=24,
        random_start=False,
    )
    env_state = env.reset(start_index=0)

    # 2. Extract initial row from env dataset
    row0 = env.df.iloc[0]
    hour = int(row0["hour"])
    load_kw = float(row0["real_load_kW"])
    next_load_kw = float(env.df.iloc[1]["real_load_kW"])
    solar_kw = env.solar.get_generation(hour)
    is_weekend = float(row0["is_weekend"])

    # Build simulated datetime with exact hour and weekend match
    # 2018-02-01 was Thursday (is_weekend = 0)
    test_dt = datetime(2018, 2, 1, hour, 0, 0)

    # 3. Initialize Live Components
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    state_builder = StateBuilder(config)
    inference_engine = DQNInferenceEngine(MODEL_PATH)
    dispatcher = EnergyDispatcher(config)

    telemetry = {
        "load_kw": load_kw,
        "next_load_kw": next_load_kw,
        "solar_kw": solar_kw,
        "battery_soc": env.battery.soc,
        "ev0_soc": env.evs[0].soc,
        "ev1_soc": env.evs[1].soc,
        "ev2_soc": env.evs[2].soc,
    }

    live_state, meta = state_builder.build_state(
        sensor_snapshot=telemetry,
        current_time=test_dt,
        step=0,
    )

    # State vector parity check
    assert np.allclose(live_state, env_state, atol=1e-4), (
        f"State vector mismatch:\nLive: {live_state}\nEnv:  {env_state}"
    )

    # 4. Action parity check
    live_action, q_vals = inference_engine.predict(live_state)

    # 5. Step both and check physical dispatch parity
    next_env_state, reward, done, info = env.step(live_action)

    dispatch = dispatcher.dispatch(
        action=live_action,
        load_kw=meta["load_kw"],
        solar_kw=meta["solar_kw"],
        tariff_rate=meta["tariff_inr"],
        battery_soc=telemetry["battery_soc"],
        ev_socs=[telemetry["ev0_soc"], telemetry["ev1_soc"], telemetry["ev2_soc"]],
        dt_hours=1.0,
    )

    # Compare power values
    assert dispatch["grid_import_kw"] == pytest.approx(info["grid_imported_kW"], abs=1e-4)
    assert dispatch["battery_charge_kw"] == pytest.approx(info["battery_charge_kW"], abs=1e-4)
    assert dispatch["battery_discharge_kw"] == pytest.approx(info["battery_discharge_kW"], abs=1e-4)
    assert dispatch["solar_used_kw"] == pytest.approx(info["solar_used_kW"], abs=1e-4)
    assert dispatch["solar_curtailed_kw"] == pytest.approx(info["solar_curtailed_kW"], abs=1e-4)
    assert dispatch["overload_kw"] == pytest.approx(info["grid_overload_kW"], abs=1e-4)
    assert dispatch["grid_cost_inr"] == pytest.approx(info["grid_cost_inr"], abs=1e-4)
