"""
Fake ESP32/ESP8266 Simulator for Smart Energy Grid
==================================================
Simulates an ESP microcontroller publishing sensor telemetry over MQTT.
Replays realistic data from data/processed/test.csv or generates synthetic profiles.

Usage:
    python simulator/fake_esp.py [--broker localhost] [--port 1883] [--interval 2.0] [--topic grid/esp1/telemetry]
"""

import argparse
import json
import logging
from pathlib import Path
import random
import time
from typing import Optional

import pandas as pd
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [FAKE-ESP] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fake_esp")

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA_PATH = ROOT / "data" / "processed" / "test.csv"


class FakeESP:

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        topic: str = "grid/esp1/telemetry",
        interval: float = 2.0,
        csv_path: Optional[Path] = None,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic = topic
        self.interval = interval
        self.csv_path = csv_path or TEST_DATA_PATH

        self.df = None
        if self.csv_path.exists():
            try:
                self.df = pd.read_csv(self.csv_path)
                logger.info(f"Loaded {len(self.df)} records from {self.csv_path.name}")
            except Exception as e:
                logger.warning(f"Failed to read CSV: {e}. Using synthetic data.")

        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id="fake_esp32_sensor",
        )
        self.client.on_connect = self._on_connect

        # Simulated state
        self.battery_soc = 50.0
        self.ev0_soc = 30.0
        self.ev1_soc = 40.0
        self.ev2_soc = 25.0
        self.row_idx = 0

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        logger.info(f"Connected to MQTT broker at {self.broker_host}:{self.broker_port} (code: {reason_code})")

    def start(self):
        logger.info(f"Connecting to broker {self.broker_host}:{self.broker_port}...")
        self.client.connect(self.broker_host, self.broker_port, keepalive=60)
        self.client.loop_start()

        try:
            self.run_loop()
        except KeyboardInterrupt:
            logger.info("Stopping Fake ESP...")
        finally:
            self.client.loop_stop()
            self.client.disconnect()

    def run_loop(self):
        while True:
            payload = self.generate_payload()
            payload_str = json.dumps(payload)
            res = self.client.publish(self.topic, payload_str, qos=1)
            logger.info(
                f"Published -> Load: {payload['load_kw']:5.1f} kW | Solar: {payload['solar_kw']:4.1f} kW | "
                f"BatSOC: {payload['battery_soc']:4.1f}% | EV_SOCs: [{payload['ev0_soc']:.0f}%, {payload['ev1_soc']:.0f}%, {payload['ev2_soc']:.0f}%]"
            )
            time.sleep(self.interval)

    def generate_payload(self) -> dict:
        now = time.time()

        if self.df is not None and len(self.df) > 0:
            row = self.df.iloc[self.row_idx]
            self.row_idx = (self.row_idx + 1) % len(self.df)
            load_kw = float(row["real_load_kW"])
            next_idx = (self.row_idx) % len(self.df)
            next_load_kw = float(self.df.iloc[next_idx]["real_load_kW"])
            hour = int(row["hour"])
        else:
            # Synthetic 24h cycle
            hour = int((now // 3600) % 24)
            load_kw = 35.0 + 15.0 * (random.random() - 0.5)
            next_load_kw = load_kw + random.uniform(-2.0, 2.0)

        # Solar generation model (bell curve between 6h and 18h)
        if 6 <= hour < 18:
            import math
            angle = math.pi * (hour - 6.0) / 12.0
            solar_kw = max(0.0, 50.0 * math.sin(angle) + random.uniform(-3.0, 3.0))
        else:
            solar_kw = 0.0

        # Simulate gradual battery/EV drift for realism
        self.battery_soc = max(20.0, min(95.0, self.battery_soc + random.uniform(-0.5, 0.5)))
        self.ev0_soc = min(90.0, self.ev0_soc + random.uniform(0.0, 0.3))
        self.ev1_soc = min(90.0, self.ev1_soc + random.uniform(0.0, 0.2))
        self.ev2_soc = min(90.0, self.ev2_soc + random.uniform(0.0, 0.3))

        return {
            "timestamp": now,
            "load_kw": round(load_kw, 2),
            "next_load_kw": round(next_load_kw, 2),
            "solar_kw": round(solar_kw, 2),
            "battery_soc": round(self.battery_soc, 2),
            "ev0_soc": round(self.ev0_soc, 2),
            "ev1_soc": round(self.ev1_soc, 2),
            "ev2_soc": round(self.ev2_soc, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Simulate ESP sensor telemetry publishing to MQTT")
    parser.add_argument("--broker", type=str, default="localhost", help="MQTT broker host")
    parser.add_argument("--port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument("--topic", type=str, default="grid/esp1/telemetry", help="MQTT topic")
    parser.add_argument("--interval", type=float, default=2.0, help="Publish interval in seconds")
    args = parser.parse_args()

    esp = FakeESP(
        broker_host=args.broker,
        broker_port=args.port,
        topic=args.topic,
        interval=args.interval,
    )
    esp.start()


if __name__ == "__main__":
    main()
