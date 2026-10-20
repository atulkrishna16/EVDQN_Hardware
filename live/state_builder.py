"""
State Builder for Live Smart Grid Dispatch Controller
=====================================================
Transforms real-time sensor telemetry and Pi system clock into the exact
12-dimensional observation vector expected by the trained DQN model.
"""

from datetime import datetime
import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Python < 3.9 compatibility
    from backports.zoneinfo import ZoneInfo  # type: ignore


def get_tariff(hour: int) -> float:
    """
    Time-of-use tariff matching SmartGridEnv.get_tariff(hour).
    Rate in ₹/kWh.
    """
    hour = int(hour) % 24
    if 0 <= hour < 6:
        return 5.0
    if 6 <= hour < 9:
        return 7.0
    if 9 <= hour < 12:
        return 9.0
    if 12 <= hour < 18:
        return 6.0
    if 18 <= hour < 22:
        return 10.0
    return 6.0


class StateBuilder:
    """
    Constructs the 12-dimensional normalized state vector:
    [0]  hour_sin: np.sin(2 * pi * hour / 24.0)
    [1]  hour_cos: np.cos(2 * pi * hour / 24.0)
    [2]  load_kw / 100.0
    [3]  next_load_kw / 100.0 (persisted or forecasted)
    [4]  solar_kw / 60.0
    [5]  tariff / 10.0
    [6]  battery_soc / 100.0
    [7]  ev0_soc / 100.0
    [8]  ev1_soc / 100.0
    [9]  ev2_soc / 100.0
    [10] is_weekend (float: 1.0 or 0.0)
    [11] progress (step / 23.0, capped at 1.0)
    """

    STATE_SIZE = 12

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tz_name = config.get("timing", {}).get("timezone", "Asia/Kolkata")
        try:
            self.tz = ZoneInfo(self.tz_name)
        except Exception:
            self.tz = None

        # Episode progression parameters
        self.episode_length = config.get("timing", {}).get("episode_length_steps", 24)
        self.step_counter = 0

        # Safe default values matching config.py / SmartGridEnv
        grid_specs = config.get("grid_specs", {})
        self.default_battery_soc = grid_specs.get("battery", {}).get("initial_soc", 50.0)
        evs = grid_specs.get("evs", [])
        self.default_ev_soc = [
            evs[0].get("initial_soc", 30.0) if len(evs) > 0 else 30.0,
            evs[1].get("initial_soc", 40.0) if len(evs) > 1 else 40.0,
            evs[2].get("initial_soc", 25.0) if len(evs) > 2 else 25.0,
        ]

        # Last known valid readings for fallback
        self._last_known = {
            "load_kw": 25.0,
            "next_load_kw": 25.0,
            "solar_kw": 0.0,
            "battery_soc": self.default_battery_soc,
            "ev0_soc": self.default_ev_soc[0],
            "ev1_soc": self.default_ev_soc[1],
            "ev2_soc": self.default_ev_soc[2],
        }

    def reset_step_counter(self):
        """Reset episode progression counter."""
        self.step_counter = 0

    def increment_step(self):
        """Advance step counter within episode length."""
        self.step_counter = (self.step_counter + 1) % self.episode_length

    def build_state(
        self,
        sensor_snapshot: Dict[str, Optional[float]],
        current_time: Optional[datetime] = None,
        step: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Builds the 12-element numpy float32 vector and metadata dictionary.
        """
        # Resolve time
        if current_time is None:
            if self.tz is not None:
                current_time = datetime.now(self.tz)
            else:
                current_time = datetime.now()

        hour = current_time.hour
        # Monday is 0 and Sunday is 6; weekend is Saturday (5) or Sunday (6)
        is_weekend = 1.0 if current_time.weekday() >= 5 else 0.0

        # Step and progress
        current_step = step if step is not None else self.step_counter
        max_steps = max(1, self.episode_length - 1)
        progress = min(1.0, float(current_step) / float(max_steps))

        # Sine/Cosine of hour
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)

        # Tariff
        tariff_inr = get_tariff(hour)
        tariff_scaled = tariff_inr / 10.0

        # Extract sensor values with fallback to last known
        def resolve_val(key: str, default: float) -> float:
            val = sensor_snapshot.get(key)
            if val is not None and not math.isnan(val):
                self._last_known[key] = float(val)
                return float(val)
            return self._last_known.get(key, default)

        load_kw = resolve_val("load_kw", 25.0)
        # If next_load_kw is not provided, use current load_kw as persistence forecast
        next_load_kw = sensor_snapshot.get("next_load_kw")
        if next_load_kw is None or math.isnan(next_load_kw):
            next_load_kw = load_kw
        else:
            self._last_known["next_load_kw"] = float(next_load_kw)

        solar_kw = resolve_val("solar_kw", 0.0)
        battery_soc = resolve_val("battery_soc", self.default_battery_soc)
        ev0_soc = resolve_val("ev0_soc", self.default_ev_soc[0])
        ev1_soc = resolve_val("ev1_soc", self.default_ev_soc[1])
        ev2_soc = resolve_val("ev2_soc", self.default_ev_soc[2])

        # Normalizations matching SmartGridEnv._get_state
        state_vec = np.array(
            [
                hour_sin,                    # [0] Time sin
                hour_cos,                    # [1] Time cos
                load_kw / 100.0,             # [2] Real load (scaled by 100 kW)
                next_load_kw / 100.0,        # [3] Next load (scaled by 100 kW)
                solar_kw / 60.0,             # [4] Solar generation (scaled by 60 kW)
                tariff_scaled,               # [5] Tariff (scaled by 10.0)
                battery_soc / 100.0,         # [6] Battery SOC (scaled to [0, 1])
                ev0_soc / 100.0,             # [7] EV0 SOC (scaled to [0, 1])
                ev1_soc / 100.0,             # [8] EV1 SOC (scaled to [0, 1])
                ev2_soc / 100.0,             # [9] EV2 SOC (scaled to [0, 1])
                float(is_weekend),           # [10] Weekend indicator
                progress,                    # [11] Episode progress
            ],
            dtype=np.float32,
        )

        metadata = {
            "hour": hour,
            "is_weekend": bool(is_weekend),
            "step": current_step,
            "progress": progress,
            "tariff_inr": tariff_inr,
            "load_kw": load_kw,
            "next_load_kw": next_load_kw,
            "solar_kw": solar_kw,
            "battery_soc": battery_soc,
            "ev0_soc": ev0_soc,
            "ev1_soc": ev1_soc,
            "ev2_soc": ev2_soc,
        }

        return state_vec, metadata
