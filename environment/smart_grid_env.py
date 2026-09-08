from pathlib import Path

import numpy as np
import pandas as pd

from .battery import Battery
from .solar import SolarModel
from .ev import EV
from .grid import Grid


class SmartGridEnv:
    """
    Smart Energy Grid Reinforcement Learning Environment.

    Real-world demand:
        ETT dataset

    Simulated components:
        Solar
        Battery
        3 EVs
        Time-of-use tariff

    Action space:
        40 discrete actions

        5 battery actions
        ×
        2 EV0 states
        ×
        2 EV1 states
        ×
        2 EV2 states

        = 40 actions

    State:
        12 features
    """

    BATTERY_ACTIONS = [
        -25.0,   # strong discharge
        -12.5,   # moderate discharge
         0.0,    # idle
         12.5,   # moderate charge
         25.0,   # strong charge
    ]

    def __init__(
        self,
        data_path,
        episode_length=24,
        random_start=True,
        seed=42,
    ):

        self.data_path = Path(data_path)

        self.df = pd.read_csv(
            self.data_path,
            parse_dates=["date"]
        )

        required_columns = [
            "date",
            "real_load_kW",
            "hour",
            "is_weekend",
        ]

        missing = [
            c
            for c in required_columns
            if c not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        self.df = (
            self.df
            .sort_values("date")
            .reset_index(drop=True)
        )

        self.episode_length = min(
            episode_length,
            len(self.df)
        )

        self.random_start = random_start

        self.rng = np.random.default_rng(
            seed
        )

        # ====================================================
        # BATTERY
        # ====================================================

        self.battery = Battery(
            capacity_kwh=100.0,
            initial_soc=50.0,
            min_soc=20.0,
            max_soc=95.0,
            max_charge_kw=25.0,
            max_discharge_kw=25.0,
            efficiency=0.95,
        )

        # ====================================================
        # SOLAR
        # ====================================================

        self.solar = SolarModel(
            peak_kw=60.0
        )

        # ====================================================
        # EVs
        # ====================================================

        self.evs = [

            EV(
                capacity_kwh=50.0,
                initial_soc=30.0,
                target_soc=80.0,
                max_charge_kw=7.0,
                efficiency=0.95,
            ),

            EV(
                capacity_kwh=50.0,
                initial_soc=40.0,
                target_soc=90.0,
                max_charge_kw=7.0,
                efficiency=0.95,
            ),

            EV(
                capacity_kwh=50.0,
                initial_soc=25.0,
                target_soc=80.0,
                max_charge_kw=7.0,
                efficiency=0.95,
            ),
        ]

        # ====================================================
        # GRID
        # ====================================================

        self.grid = Grid(
            limit_kw=100.0
        )

        self.current_index = 0
        self.start_index = 0
        self.steps = 0

        self.last_info = {}

        self.state_size = 12
        self.action_size = 40

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, start_index=None):

        max_start = (
            len(self.df)
            - self.episode_length
        )

        if max_start < 0:
            raise ValueError(
                "Dataset is shorter than episode length."
            )

        if start_index is not None:

            self.start_index = int(
                start_index
            )

        elif self.random_start:

            if max_start == 0:

                self.start_index = 0

            else:

                self.start_index = int(
                    self.rng.integers(
                        0,
                        max_start + 1
                    )
                )

        else:

            self.start_index = 0

        self.current_index = (
            self.start_index
        )

        self.steps = 0

        self.battery.reset()

        for ev in self.evs:
            ev.reset()

        self.last_info = {}

        return self._get_state()

    # ========================================================
    # ACTION DECODING
    # ========================================================

    def decode_action(self, action):

        action = int(action)

        if not (
            0 <= action < self.action_size
        ):
            raise ValueError(
                f"Invalid action: {action}"
            )

        battery_index = action // 8

        ev_bits = action % 8

        battery_power = (
            self.BATTERY_ACTIONS[
                battery_index
            ]
        )

        ev0_on = bool(
            ev_bits & 1
        )

        ev1_on = bool(
            ev_bits & 2
        )

        ev2_on = bool(
            ev_bits & 4
        )

        return (
            battery_power,
            [
                ev0_on,
                ev1_on,
                ev2_on,
            ]
        )

    # ========================================================
    # TARIFF
    # ========================================================

    def get_tariff(self, hour):

        hour = int(hour)

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

    # ========================================================
    # STATE
    # ========================================================

    def _get_state(self):

        row = self.df.iloc[
            self.current_index
        ]

        hour = int(
            row["hour"]
        )

        load_kw = float(
            row["real_load_kW"]
        )

        next_index = min(
            self.current_index + 1,
            len(self.df) - 1
        )

        next_load_kw = float(
            self.df.iloc[next_index][
                "real_load_kW"
            ]
        )

        solar_kw = (
            self.solar.get_generation(
                hour
            )
        )

        tariff = self.get_tariff(
            hour
        )

        hour_sin = np.sin(
            2 * np.pi * hour / 24.0
        )

        hour_cos = np.cos(
            2 * np.pi * hour / 24.0
        )

        # ----------------------------------------------------
        # Progress is capped at 1.0
        # ----------------------------------------------------

        progress = min(
            1.0,
            self.steps
            / max(
                1,
                self.episode_length - 1
            )
        )

        state = np.array(
            [

                # Time
                hour_sin,
                hour_cos,

                # Real load
                load_kw / 100.0,
                next_load_kw / 100.0,

                # Solar
                solar_kw / 60.0,

                # Tariff
                tariff / 10.0,

                # Battery
                self.battery.soc / 100.0,

                # EVs
                self.evs[0].soc / 100.0,
                self.evs[1].soc / 100.0,
                self.evs[2].soc / 100.0,

                # Weekend
                float(
                    row["is_weekend"]
                ),

                # Episode progress
                progress,
            ],
            dtype=np.float32
        )

        return state

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):

        battery_request, ev_enabled = (
            self.decode_action(
                action
            )
        )

        row = self.df.iloc[
            self.current_index
        ]

        hour = int(
            row["hour"]
        )

        load_kw = float(
            row["real_load_kW"]
        )

        solar_kw = (
            self.solar.get_generation(
                hour
            )
        )

        tariff = self.get_tariff(
            hour
        )

        # ====================================================
        # EV CHARGING
        # ====================================================

        ev_power = []

        for ev, enabled in zip(
            self.evs,
            ev_enabled
        ):

            power = ev.charge(
                enabled=enabled,
                dt_hours=1.0
            )

            ev_power.append(
                power
            )

        total_ev_power = sum(
            ev_power
        )

        # ====================================================
        # BATTERY
        # ====================================================

        (
            actual_charge_kw,
            actual_discharge_kw,
        ) = self.battery.step(
            battery_request,
            dt_hours=1.0
        )

        # ====================================================
        # GRID
        # ====================================================

        grid_result = self.grid.calculate(
            load_kw=load_kw,
            ev_power_kw=total_ev_power,
            solar_kw=solar_kw,
            battery_charge_kw=actual_charge_kw,
            battery_discharge_kw=actual_discharge_kw,
            tariff_rate=tariff,
        )

        grid_import = grid_result[
            "grid_import_kw"
        ]

        grid_cost = grid_result[
            "grid_cost_inr"
        ]

        overload_kw = grid_result[
            "overload_kw"
        ]

        solar_used = grid_result[
            "solar_used_kw"
        ]

        solar_curtailed = grid_result[
            "solar_curtailed_kw"
        ]

        # ====================================================
        # SOLAR UTILIZATION
        # ====================================================

        if solar_kw > 0:

            solar_utilization = (
                solar_used
                / solar_kw
            )

        else:

            solar_utilization = 1.0

        solar_utilization = min(
            1.0,
            max(
                0.0,
                solar_utilization
            )
        )

        # ====================================================
        # SELF SUFFICIENCY
        # ====================================================

        total_demand = (
            load_kw
            + total_ev_power
        )

        renewable_and_battery = (
            solar_used
            + actual_discharge_kw
        )

        self_sufficiency = (
            renewable_and_battery
            / max(
                total_demand,
                1e-6
            )
        )

        self_sufficiency = min(
            1.0,
            max(
                0.0,
                self_sufficiency
            )
        )

        # ====================================================
        # EV DEFICIT
        # ====================================================

        total_ev_deficit = sum(
            ev.deficit
            for ev in self.evs
        )

        # ====================================================
        # BATTERY CYCLING
        # ====================================================

        battery_cycle = (
            actual_charge_kw
            + actual_discharge_kw
        )

        # ====================================================
        # REWARD
        # ====================================================

        reward = 0.0

        # ----------------------------------------------------
        # 1. Electricity cost
        # ----------------------------------------------------

        reward -= (
            0.08
            * grid_cost
        )

        # ----------------------------------------------------
        # 2. Grid overload
        # ----------------------------------------------------

        if overload_kw > 0:

            reward -= (
                30.0
                * overload_kw
            )

            reward -= (
                2.0
                * overload_kw ** 2
            )

        # ----------------------------------------------------
        # 3. Peak grid penalty
        # ----------------------------------------------------

        if grid_import > 80.0:

            reward -= (
                0.015
                * (
                    grid_import
                    - 80.0
                )
            )

        # ----------------------------------------------------
        # 4. Solar utilization
        # ----------------------------------------------------

        reward += (
            1.5
            * solar_utilization
        )

        # ----------------------------------------------------
        # 5. EV charging
        # ----------------------------------------------------

        reward += (
            0.10
            * total_ev_power
        )

        # ----------------------------------------------------
        # 6. EV target deficit
        # ----------------------------------------------------

        reward -= (
            0.04
            * total_ev_deficit
        )

        # ----------------------------------------------------
        # 7. Battery health
        # ----------------------------------------------------

        if self.battery.soc < 25.0:

            reward -= 2.0

        elif self.battery.soc > 90.0:

            reward -= 0.5

        # ----------------------------------------------------
        # 8. NEW: Battery degradation/cycling penalty
        # ----------------------------------------------------
        #
        # Prevents the DQN from unnecessarily moving the
        # battery between extreme SOC values.
        #

        reward -= (
            0.02
            * battery_cycle
        )

        # ----------------------------------------------------
        # 9. Penalize grid-powered battery charging
        # ----------------------------------------------------

        if (
            actual_charge_kw > 0
            and solar_kw < actual_charge_kw
        ):

            grid_charging_power = (
                actual_charge_kw
                - solar_kw
            )

            grid_charging_power = max(
                0.0,
                grid_charging_power
            )

            reward -= (
                0.05
                * grid_charging_power
            )

        # ====================================================
        # ADVANCE
        # ====================================================

        self.steps += 1

        next_index = (
            self.current_index + 1
        )

        # Prevent index overflow
        if next_index >= len(
            self.df
        ):

            next_index = (
                len(self.df) - 1
            )

        self.current_index = next_index

        done = (
            self.steps
            >= self.episode_length
        )

        next_state = (
            self._get_state()
        )

        # ====================================================
        # INFORMATION
        # ====================================================

        info = {

            "date":
                str(row["date"]),

            "hour":
                hour,

            "load_kW":
                load_kw,

            "solar_generated_kW":
                solar_kw,

            "solar_used_kW":
                solar_used,

            "solar_curtailed_kW":
                solar_curtailed,

            "battery_charge_kW":
                actual_charge_kw,

            "battery_discharge_kW":
                actual_discharge_kw,

            "battery_cycle_kW":
                battery_cycle,

            "battery_soc_pct":
                self.battery.soc,

            "ev0_power_kW":
                ev_power[0],

            "ev1_power_kW":
                ev_power[1],

            "ev2_power_kW":
                ev_power[2],

            "ev0_soc_pct":
                self.evs[0].soc,

            "ev1_soc_pct":
                self.evs[1].soc,

            "ev2_soc_pct":
                self.evs[2].soc,

            "total_ev_power_kW":
                total_ev_power,

            "grid_imported_kW":
                grid_import,

            "grid_cost_inr":
                grid_cost,

            "grid_overload_kW":
                overload_kw,

            "tariff_rate":
                tariff,

            "renewable_utilization_pct":
                solar_utilization * 100.0,

            "self_sufficiency_pct":
                self_sufficiency * 100.0,

            "reward":
                reward,
        }

        self.last_info = info

        return (
            next_state,
            float(reward),
            done,
            info,
        )