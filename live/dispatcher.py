"""
Energy Dispatcher for Live Smart Grid Dispatch Controller
=========================================================
Converts DQN action (0..39) and incoming sensor readings into exact power flows.
Reuses the environment's Battery, EV, and Grid models and constraints.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from environment.battery import Battery
from environment.ev import EV
from environment.grid import Grid
from environment.smart_grid_env import SmartGridEnv

logger = logging.getLogger("smart_grid.dispatcher")


class EnergyDispatcher:
    """
    Computes power dispatch per source using the physical models from environment/.
    """

    BATTERY_ACTIONS = SmartGridEnv.BATTERY_ACTIONS  # [-25.0, -12.5, 0.0, 12.5, 25.0]

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        grid_specs = config.get("grid_specs", {})

        # 1. Battery Model
        bat_cfg = grid_specs.get("battery", {})
        self.battery = Battery(
            capacity_kwh=bat_cfg.get("capacity_kwh", 100.0),
            initial_soc=bat_cfg.get("initial_soc", 50.0),
            min_soc=bat_cfg.get("min_soc", 20.0),
            max_soc=bat_cfg.get("max_soc", 95.0),
            max_charge_kw=bat_cfg.get("max_charge_kw", 25.0),
            max_discharge_kw=bat_cfg.get("max_discharge_kw", 25.0),
            efficiency=bat_cfg.get("efficiency", 0.95),
        )

        # 2. EV Models
        ev_cfgs = grid_specs.get("evs", [])
        self.evs: List[EV] = []
        for i, ev_c in enumerate(ev_cfgs):
            self.evs.append(
                EV(
                    capacity_kwh=ev_c.get("capacity_kwh", 50.0),
                    initial_soc=ev_c.get("initial_soc", 30.0),
                    target_soc=ev_c.get("target_soc", 80.0),
                    max_charge_kw=ev_c.get("max_charge_kw", 7.0),
                    efficiency=ev_c.get("efficiency", 0.95),
                )
            )

        # Fallback if config had no EVs
        if not self.evs:
            self.evs = [
                EV(capacity_kwh=50.0, initial_soc=30.0, target_soc=80.0, max_charge_kw=7.0, efficiency=0.95),
                EV(capacity_kwh=50.0, initial_soc=40.0, target_soc=90.0, max_charge_kw=7.0, efficiency=0.95),
                EV(capacity_kwh=50.0, initial_soc=25.0, target_soc=80.0, max_charge_kw=7.0, efficiency=0.95),
            ]

        # 3. Grid Model
        grid_limit = grid_specs.get("grid_limit_kw", 100.0)
        self.grid = Grid(limit_kw=grid_limit)

        # Simulation dt
        self.dt_hours = config.get("timing", {}).get("dt_hours", 1.0)

    def decode_action(self, action: int) -> Tuple[float, List[bool]]:
        """
        Action decoding matching SmartGridEnv.decode_action.
        action = battery_index * 8 + ev_bits
        """
        action = int(action)
        if not (0 <= action < 40):
            raise ValueError(f"Invalid action: {action}. Must be in [0, 39].")

        battery_index = action // 8
        ev_bits = action % 8

        battery_power = self.BATTERY_ACTIONS[battery_index]
        ev_enabled = [
            bool(ev_bits & 1),
            bool(ev_bits & 2),
            bool(ev_bits & 4),
        ]
        return battery_power, ev_enabled

    def sync_telemetry_soc(
        self,
        battery_soc: Optional[float] = None,
        ev_socs: Optional[List[float]] = None,
    ):
        """Synchronizes physical model states with live incoming telemetry."""
        if battery_soc is not None:
            self.battery.soc = max(self.battery.min_soc, min(self.battery.max_soc, float(battery_soc)))

        if ev_socs is not None:
            for ev, s in zip(self.evs, ev_socs):
                if s is not None:
                    ev.soc = max(0.0, min(100.0, float(s)))

    def dispatch(
        self,
        action: int,
        load_kw: float,
        solar_kw: float,
        tariff_rate: float,
        battery_soc: Optional[float] = None,
        ev_socs: Optional[List[float]] = None,
        dt_hours: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes physical dispatch calculation and power balance for the chosen action.
        """
        dt = dt_hours if dt_hours is not None else self.dt_hours

        # Sync telemetry SOCs
        self.sync_telemetry_soc(battery_soc, ev_socs)

        # Decode action
        battery_request, ev_enabled = self.decode_action(action)

        # 1. EV Charging step
        ev_powers = []
        for ev, enabled in zip(self.evs, ev_enabled):
            p = ev.charge(enabled=enabled, dt_hours=dt)
            ev_powers.append(p)
        total_ev_power = sum(ev_powers)

        # 2. Battery step
        actual_charge_kw, actual_discharge_kw = self.battery.step(
            battery_request, dt_hours=dt
        )

        # 3. Grid calculate
        grid_result = self.grid.calculate(
            load_kw=load_kw,
            ev_power_kw=total_ev_power,
            solar_kw=solar_kw,
            battery_charge_kw=actual_charge_kw,
            battery_discharge_kw=actual_discharge_kw,
            tariff_rate=tariff_rate,
        )

        grid_import_kw = grid_result["grid_import_kw"]
        grid_cost_inr = grid_result["grid_cost_inr"]
        overload_kw = grid_result["overload_kw"]
        solar_used_kw = grid_result["solar_used_kw"]
        solar_curtailed_kw = grid_result["solar_curtailed_kw"]

        # Total energy consumed this timestep
        # Consumed by: building load + EVs + battery charging (if charging)
        total_demand_kw = load_kw + total_ev_power + actual_charge_kw

        # Energy supplied by: Solar used + Battery discharge + Grid import
        total_supply_kw = solar_used_kw + actual_discharge_kw + grid_import_kw
        denominator = max(total_supply_kw, 1e-5)

        grid_pct = (grid_import_kw / denominator) * 100.0
        solar_pct = (solar_used_kw / denominator) * 100.0
        batt_discharge_pct = (actual_discharge_kw / denominator) * 100.0

        if actual_discharge_kw > 0.05:
            battery_mode = "DISCHARGE"
            battery_display_kw = actual_discharge_kw
        elif actual_charge_kw > 0.05:
            battery_mode = "CHARGE"
            battery_display_kw = -actual_charge_kw
        else:
            battery_mode = "IDLE"
            battery_display_kw = 0.0

        return {
            "action": action,
            "battery_request_kw": battery_request,
            "ev_enabled": ev_enabled,
            # Sources
            "grid_import_kw": grid_import_kw,
            "grid_pct": grid_pct,
            "solar_generated_kw": solar_kw,
            "solar_used_kw": solar_used_kw,
            "solar_pct": solar_pct,
            "solar_curtailed_kw": solar_curtailed_kw,
            "battery_charge_kw": actual_charge_kw,
            "battery_discharge_kw": actual_discharge_kw,
            "battery_display_kw": battery_display_kw,
            "battery_pct": batt_discharge_pct,
            "battery_mode": battery_mode,
            # Loads
            "base_load_kw": load_kw,
            "ev_powers_kw": ev_powers,
            "total_ev_power_kw": total_ev_power,
            "total_demand_kw": total_demand_kw,
            "total_supply_kw": total_supply_kw,
            # State
            "battery_soc_pct": self.battery.soc,
            "ev_socs_pct": [ev.soc for ev in self.evs],
            "tariff_rate": tariff_rate,
            "grid_cost_inr": grid_cost_inr,
            "overload_kw": overload_kw,
        }
