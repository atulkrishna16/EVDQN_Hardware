"""
Main Live Controller Entry Point
=================================
Runs on Raspberry Pi. Connects to MQTT broker, receives ESP telemetry,
executes DQN inference, dispatches power per source, and drives the LCD.

Usage:
    python live/main_live.py [--config config_live.yaml] [--mock-lcd] [--fast-forward]
"""

import argparse
from datetime import datetime, timedelta
import logging
from logging.handlers import RotatingFileHandler
import os
import signal
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import time
from typing import Any, Dict, Optional
import yaml

from live.dispatcher import EnergyDispatcher
from live.inference import DQNInferenceEngine
from live.lcd import LCDRenderer
from live.mqtt_client import SensorStateStore, SmartGridMQTTSubscriber
from live.state_builder import StateBuilder, get_tariff

logger = logging.getLogger("smart_grid")


def setup_logging(config: Dict[str, Any]):
    log_cfg = config.get("logging", {})
    level_name = log_cfg.get("level", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    # Optional Rotating File handler
    if log_cfg.get("log_to_file", False):
        log_file = log_cfg.get("file_path", "live_controller.log")
        max_bytes = int(log_cfg.get("max_bytes", 5242880))
        backup_count = int(log_cfg.get("backup_count", 3))
        fh = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        fh.setLevel(level)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


class LiveGridController:
    """Orchestrates MQTT ingestion, DQN inference, energy dispatch, and LCD display."""

    def __init__(self, config_path: str, mock_lcd: bool = False, fast_forward: bool = False):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        setup_logging(self.config)
        logger.info("Initializing Live Smart Grid Controller...")

        self.fast_forward = fast_forward or self.config.get("timing", {}).get("fast_forward", False)
        self.simulated_clock = datetime(2026, 1, 1, 0, 0, 0) if self.fast_forward else None

        # Timing
        timing_cfg = self.config.get("timing", {})
        self.decision_interval = float(timing_cfg.get("decision_interval_seconds", 5.0))
        self.stale_timeout = float(timing_cfg.get("data_stale_timeout_seconds", 15.0))
        self.dt_hours = float(timing_cfg.get("dt_hours", 1.0))

        # 1. State Store and MQTT Subscriber
        self.store = SensorStateStore(stale_timeout_seconds=self.stale_timeout)
        self.mqtt_sub = SmartGridMQTTSubscriber(self.config.get("mqtt", {}), self.store)

        # 2. State Builder
        self.state_builder = StateBuilder(self.config)

        # 3. DQN Inference Engine
        model_cfg = self.config.get("model", {})
        model_file = model_cfg.get("model_path", "models/best_dqn_model.pth")
        engine_type = model_cfg.get("engine", "torch")
        self.inference_engine = DQNInferenceEngine(model_file, engine_type=engine_type)

        # 4. Energy Dispatcher
        self.dispatcher = EnergyDispatcher(self.config)

        # 5. LCD Renderer
        self.lcd = LCDRenderer(self.config, mock=mock_lcd)

        # Control loop flags
        self.running = False
        self.last_safe_dispatch: Optional[Dict[str, Any]] = None

    def start(self):
        self.running = True
        logger.info("Starting MQTT subscriber...")
        self.mqtt_sub.start()

        logger.info(
            f"Controller running. Decision interval: {self.decision_interval}s, Stale timeout: {self.stale_timeout}s"
        )
        if self.fast_forward:
            logger.info(">>> FAST-FORWARD DEMO MODE ENABLED: Clock advances 1 simulated hour every tick.")

        try:
            self.loop()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
        finally:
            self.stop()

    def loop(self):
        while self.running:
            start_ts = time.time()

            # 1. Retrieve telemetry snapshot
            snapshot, is_fresh, age = self.store.get_snapshot()
            is_connected = self.mqtt_sub.store.is_connected

            current_time = self.simulated_clock if self.fast_forward else None

            # 2. Safety Fallback if data is missing or stale
            if not is_fresh:
                logger.warning(
                    f"Sensor data is STALE or MISSING (age: {age:.1f}s > {self.stale_timeout}s, connected: {is_connected}). "
                    "Applying SAFE GRID-ONLY fallback."
                )
                # In safe mode, battery remains idle (0 kW), EVs not charged, grid covers base load
                safe_load = snapshot.get("load_kw") or 25.0
                hour = current_time.hour if current_time else datetime.now().hour
                safe_tariff = get_tariff(hour)
                fallback_dispatch = self.dispatcher.dispatch(
                    action=16,  # Action 16: battery idle (index 2*8=16), all EVs off (bits 0)
                    load_kw=safe_load,
                    solar_kw=0.0,
                    tariff_rate=safe_tariff,
                    battery_soc=snapshot.get("battery_soc"),
                    ev_socs=[snapshot.get("ev0_soc"), snapshot.get("ev1_soc"), snapshot.get("ev2_soc")],
                    dt_hours=self.dt_hours,
                )
                self.lcd.render(
                    dispatch_info=fallback_dispatch,
                    is_fresh=False,
                    age_seconds=age,
                    is_connected=is_connected,
                )
            else:
                # 3. Fresh data: Build state vector
                state_vec, metadata = self.state_builder.build_state(
                    sensor_snapshot=snapshot,
                    current_time=current_time,
                )

                # 4. Greedy DQN Inference
                action, q_values = self.inference_engine.predict(state_vec)

                # 5. Energy Dispatch calculation
                dispatch = self.dispatcher.dispatch(
                    action=action,
                    load_kw=metadata["load_kw"],
                    solar_kw=metadata["solar_kw"],
                    tariff_rate=metadata["tariff_inr"],
                    battery_soc=metadata["battery_soc"],
                    ev_socs=[metadata["ev0_soc"], metadata["ev1_soc"], metadata["ev2_soc"]],
                    dt_hours=self.dt_hours,
                )
                self.last_safe_dispatch = dispatch

                logger.info(
                    f"[TICK] Act: {action:2d} | Grid: {dispatch['grid_import_kw']:5.1f} kW ({dispatch['grid_pct']:2.0f}%) | "
                    f"Solar: {dispatch['solar_used_kw']:4.1f} kW | Batt: {dispatch['battery_display_kw']:+5.1f} kW ({dispatch['battery_mode']}) | "
                    f"Load: {dispatch['total_demand_kw']:5.1f} kW | Cost: Rs{dispatch['grid_cost_inr']:6.1f}"
                )

                # =====================================================================
                # [ACTUATION HOOK - FUTURE EXPANSION]
                # If physical relays / inverter controls are added in future:
                # Example:
                #   gpio_relay_battery.set_power(dispatch['battery_request_kw'])
                #   gpio_relay_evs.set_chargers(dispatch['ev_enabled'])
                # Current scope: Passive real-time computation & LCD display only.
                # =====================================================================

                # 6. Update LCD screen
                self.lcd.render(
                    dispatch_info=dispatch,
                    is_fresh=True,
                    age_seconds=age,
                    is_connected=is_connected,
                )

                # Advance simulated clock if fast-forward is enabled
                if self.fast_forward and self.simulated_clock:
                    self.simulated_clock += timedelta(hours=1)
                    self.state_builder.increment_step()

            # Sleep remaining interval
            elapsed = time.time() - start_ts
            sleep_duration = max(0.1, self.decision_interval - elapsed)
            time.sleep(sleep_duration)

    def stop(self):
        logger.info("Shutting down controller...")
        self.running = False
        try:
            self.mqtt_sub.stop()
        except Exception as e:
            logger.error(f"Error stopping MQTT client: {e}")
        try:
            self.lcd.clear()
        except Exception as e:
            logger.error(f"Error clearing LCD: {e}")
        logger.info("Shutdown complete.")


def handle_signals(controller: LiveGridController):
    def _signal_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received signal {sig_name}. Requesting graceful shutdown...")
        controller.running = False

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Smart Energy Grid Live Dispatch Controller")
    parser.add_argument(
        "--config",
        type=str,
        default="config_live.yaml",
        help="Path to YAML configuration file (default: config_live.yaml)",
    )
    parser.add_argument(
        "--mock-lcd",
        action="store_true",
        help="Use terminal ASCII mock LCD instead of physical I2C LCD",
    )
    parser.add_argument(
        "--fast-forward",
        action="store_true",
        help="Fast-forward demo mode: advance simulated clock 1 hour every tick",
    )
    args = parser.parse_args()

    controller = LiveGridController(
        config_path=args.config,
        mock_lcd=args.mock_lcd,
        fast_forward=args.fast_forward,
    )
    handle_signals(controller)
    controller.start()


if __name__ == "__main__":
    main()
