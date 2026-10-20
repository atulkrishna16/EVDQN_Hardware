"""
MQTT Client for Live Smart Grid Dispatch Controller
===================================================
Subscribes to ESP32/ESP8266 telemetry topics using Paho-MQTT v2 API.
Provides thread-safe state storage, automatic reconnection, and payload validation.
"""

import json
import logging
import math
import threading
import time
from typing import Any, Dict, Optional, Tuple

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

logger = logging.getLogger("smart_grid.mqtt")


class SensorStateStore:
    """Thread-safe store for latest sensor telemetry."""

    def __init__(self, stale_timeout_seconds: float = 15.0):
        self._lock = threading.Lock()
        self.stale_timeout_seconds = stale_timeout_seconds

        # Sensor readings
        self._state: Dict[str, Optional[float]] = {
            "load_kw": None,
            "next_load_kw": None,
            "solar_kw": None,
            "battery_soc": None,
            "ev0_soc": None,
            "ev1_soc": None,
            "ev2_soc": None,
        }
        self._last_update_ts: Optional[float] = None
        self._msg_count = 0
        self._is_connected = False

    def update_field(self, key: str, value: float, ts: Optional[float] = None) -> bool:
        """Update a single telemetry field with validation."""
        valid, sanitized = self.validate_field(key, value)
        if not valid:
            logger.warning(f"Rejected invalid sensor reading: {key}={value}")
            return False

        with self._lock:
            self._state[key] = sanitized
            self._last_update_ts = ts if ts is not None else time.time()
            self._msg_count += 1
        return True

    def update_from_dict(self, data: Dict[str, Any], ts: Optional[float] = None) -> int:
        """Batch update fields from parsed JSON dictionary."""
        valid_updates = 0
        now = ts if ts is not None else time.time()
        with self._lock:
            for k, v in data.items():
                if k in self._state:
                    valid, sanitized = self.validate_field(k, v)
                    if valid:
                        self._state[k] = sanitized
                        valid_updates += 1
                    else:
                        logger.warning(f"Rejected invalid reading in batch: {k}={v}")
            if valid_updates > 0:
                self._last_update_ts = now
                self._msg_count += 1
        return valid_updates

    def get_snapshot(self) -> Tuple[Dict[str, Optional[float]], bool, float]:
        """
        Returns:
            (state_dict_copy, is_fresh, age_in_seconds)
        """
        now = time.time()
        with self._lock:
            snapshot = dict(self._state)
            last_ts = self._last_update_ts

        if last_ts is None:
            return snapshot, False, float("inf")

        age = max(0.0, now - last_ts)
        is_fresh = age <= self.stale_timeout_seconds
        return snapshot, is_fresh, age

    def set_connected(self, connected: bool):
        with self._lock:
            self._is_connected = connected

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._is_connected

    @staticmethod
    def validate_field(key: str, value: Any) -> Tuple[bool, Optional[float]]:
        """Validate numeric bounds and finite value."""
        try:
            val = float(value)
        except (ValueError, TypeError):
            return False, None

        if math.isnan(val) or math.isinf(val):
            return False, None

        # Domain bounds
        if key in ("load_kw", "next_load_kw", "solar_kw"):
            if val < 0.0 or val > 500.0:  # Reasonable electrical limits
                return False, None
        elif key in ("battery_soc", "ev0_soc", "ev1_soc", "ev2_soc"):
            if val < 0.0 or val > 100.0:  # SOC percentage range
                return False, None

        return True, val


class SmartGridMQTTSubscriber:
    """Paho-MQTT subscriber with auto-reconnect and callbacks."""

    def __init__(self, config: Dict[str, Any], store: SensorStateStore):
        self.config = config
        self.store = store
        self.telemetry_topic = config.get("telemetry_topic", "grid/esp1/telemetry")
        self.subtopics = config.get("subtopics", {})

        # Paho v2 API Client
        client_id = config.get("client_id", "pi_energy_controller")
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=client_id,
        )

        # Credentials
        username = config.get("username", "")
        password = config.get("password", "")
        if username:
            self.client.username_pw_set(username, password)

        # Reconnect parameters
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)

        # Callbacks
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            logger.error(f"MQTT connect failed: {reason_code}")
            self.store.set_connected(False)
            return

        logger.info(f"Connected to MQTT broker (reason: {reason_code})")
        self.store.set_connected(True)

        # Subscribe to telemetry topic
        client.subscribe(self.telemetry_topic, qos=1)
        logger.info(f"Subscribed to primary telemetry: {self.telemetry_topic}")

        # Subscribe to granular subtopics
        for name, topic in self.subtopics.items():
            if topic:
                client.subscribe(topic, qos=1)
                logger.debug(f"Subscribed to subtopic: {topic} ({name})")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        logger.warning(f"MQTT disconnected (reason: {reason_code}). Auto-reconnecting...")
        self.store.set_connected(False)

    def _on_message(self, client, userdata, message):
        topic = message.topic
        payload_bytes = message.payload

        try:
            payload_str = payload_bytes.decode("utf-8").strip()
        except UnicodeDecodeError:
            logger.warning(f"Received non-UTF8 payload on {topic}")
            return

        # Check if topic is the combined JSON telemetry
        if topic == self.telemetry_topic:
            try:
                data = json.loads(payload_str)
            except json.JSONDecodeError:
                logger.warning(f"Malformed JSON on {topic}: {payload_str}")
                return

            if isinstance(data, dict):
                ts = data.get("timestamp")
                if ts is not None:
                    try:
                        ts = float(ts)
                    except (ValueError, TypeError):
                        ts = time.time()
                else:
                    ts = time.time()
                updated = self.store.update_from_dict(data, ts=ts)
                logger.debug(f"Updated {updated} fields from {topic}")
            return

        # Check if message is on a granular subtopic
        for key, subtopic in self.subtopics.items():
            if topic == subtopic:
                # Could be raw number or small JSON {"value": 12.3}
                val = None
                try:
                    parsed = json.loads(payload_str)
                    if isinstance(parsed, dict) and "value" in parsed:
                        val = parsed["value"]
                    elif isinstance(parsed, (int, float)):
                        val = parsed
                except json.JSONDecodeError:
                    try:
                        val = float(payload_str)
                    except ValueError:
                        pass

                if val is not None:
                    self.store.update_field(key, val)
                    logger.debug(f"Updated {key}={val} from {topic}")
                else:
                    logger.warning(f"Could not parse reading on {topic}: {payload_str}")
                return

    def start(self):
        host = self.config.get("broker_host", "localhost")
        port = int(self.config.get("broker_port", 1883))
        keepalive = int(self.config.get("keepalive", 60))
        logger.info(f"Connecting to MQTT broker at {host}:{port}...")
        try:
            self.client.connect(host, port, keepalive=keepalive)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"Failed to initiate MQTT connection: {e}")
            raise

    def stop(self):
        logger.info("Stopping MQTT subscriber...")
        self.client.loop_stop()
        self.client.disconnect()
        self.store.set_connected(False)
