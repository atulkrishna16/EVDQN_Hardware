"""
LCD Driver and Screen Renderer for Live Smart Grid Dispatch Controller
======================================================================
Supports:
  - Physical HD44780 20x4 / 16x2 via I2C PCF8574 backpack using RPLCD
  - Terminal Mock LCD (--mock-lcd) for development & testing without hardware
  - Rotating views (Screen 1: Power Split, Screen 2: System Status)
  - Flicker-free differential updating and I2C retry logic
  - Safe 'NO DATA' fallback screen
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("smart_grid.lcd")

try:
    from RPLCD.i2c import CharLCD
    HAS_RPLCD = True
except ImportError:
    HAS_RPLCD = False


class LCDRenderer:
    """
    Manages screen rendering, line buffering, and hardware/mock driver abstraction.
    """

    def __init__(self, config: Dict[str, Any], mock: bool = False):
        self.config = config.get("lcd", {})
        self.cols = int(self.config.get("cols", 20))
        self.rows = int(self.config.get("rows", 4))
        self.i2c_address = self.config.get("i2c_address", 0x27)
        if isinstance(self.i2c_address, str):
            self.i2c_address = int(self.i2c_address, 16)
        self.i2c_bus = int(self.config.get("i2c_bus", 1))
        self.i2c_expander = self.config.get("i2c_expander", "PCF8574")
        self.rotation_interval = float(self.config.get("rotation_interval_seconds", 3.0))

        # Check mock setting
        self.mock = mock or self.config.get("mock", False)

        # State buffers
        self._buffer: List[str] = [" " * self.cols for _ in range(self.rows)]
        self._current_screen = 0  # 0 = Power Split, 1 = System Status
        self._last_rotation_ts = time.time()
        self._hardware_lcd = None

        if not self.mock:
            self._init_hardware()
        else:
            logger.info(f"Initialized Terminal Mock LCD ({self.cols}x{self.rows})")

    def _init_hardware(self):
        if not HAS_RPLCD:
            logger.warning("RPLCD library not installed. Falling back to Mock LCD.")
            self.mock = True
            return

        try:
            self._hardware_lcd = CharLCD(
                i2c_expander=self.i2c_expander,
                address=self.i2c_address,
                port=self.i2c_bus,
                cols=self.cols,
                rows=self.rows,
                dotsize=8,
                charmap="A02",
                auto_linebreaks=False,
                backlight_enabled=True,
            )
            self._hardware_lcd.clear()
            logger.info(
                f"Initialized physical LCD ({self.cols}x{self.rows}) at I2C address 0x{self.i2c_address:02x} on bus {self.i2c_bus}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize physical LCD: {e}. Falling back to Mock LCD.")
            self.mock = True
            self._hardware_lcd = None

    def render(
        self,
        dispatch_info: Optional[Dict[str, Any]],
        is_fresh: bool,
        age_seconds: float,
        is_connected: bool,
        force_screen: Optional[int] = None,
    ):
        """
        Renders the active screen based on rotation schedule or alert status.
        """
        now = time.time()
        if force_screen is not None:
            self._current_screen = force_screen
        elif now - self._last_rotation_ts >= self.rotation_interval:
            self._current_screen = (self._current_screen + 1) % 2
            self._last_rotation_ts = now

        # Stale or No Data alert screen
        if not is_fresh or dispatch_info is None:
            lines = self._generate_no_data_screen(is_connected, age_seconds)
        elif self._current_screen == 0:
            lines = self._generate_power_split_screen(dispatch_info)
        else:
            lines = self._generate_status_screen(dispatch_info, is_connected, age_seconds)

        self._write_lines(lines)

    def _generate_power_split_screen(self, d: Dict[str, Any]) -> List[str]:
        lines: List[str] = []

        g_kw = d.get("grid_import_kw", 0.0)
        g_pct = d.get("grid_pct", 0.0)
        s_kw = d.get("solar_used_kw", 0.0)
        s_pct = d.get("solar_pct", 0.0)

        b_mode = d.get("battery_mode", "IDLE")
        b_chg = d.get("battery_charge_kw", 0.0)
        b_dis = d.get("battery_discharge_kw", 0.0)
        b_pct = d.get("battery_pct", 0.0)
        soc = d.get("battery_soc_pct", 50.0)
        tariff = d.get("tariff_rate", 6.0)

        # Tariff label
        t_label = "LOW" if tariff <= 5.0 else ("PEAK" if tariff >= 9.0 else "MED")

        if self.rows == 4:
            # 20x4 Screen 1
            lines.append(f"GRID  {g_kw:5.1f}kW {g_pct:3.0f}%")
            lines.append(f"SOLAR {s_kw:5.1f}kW {s_pct:3.0f}%")
            if b_mode == "CHARGE":
                lines.append(f"BATT  CHG:{b_chg:4.1f}kW  ")
            elif b_mode == "DISCHARGE":
                lines.append(f"BATT  {b_dis:5.1f}kW {b_pct:3.0f}%")
            else:
                lines.append("BATT  IDLE        0%")
            lines.append(f"T:{t_label:<4}  SOC:{soc:3.0f}%")
        else:
            # 16x2 Screen 1
            lines.append(f"G:{g_kw:4.1f}k S:{s_kw:4.1f}k")
            if b_mode == "CHARGE":
                lines.append(f"CHG:{b_chg:3.0f}k T:{t_label[:2]} {soc:2.0f}%")
            elif b_mode == "DISCHARGE":
                lines.append(f"B:{b_dis:4.1f}k T:{t_label[:2]} {soc:2.0f}%")
            else:
                lines.append(f"B:IDLE T:{t_label[:2]} {soc:2.0f}%")

        return [line.ljust(self.cols)[: self.cols] for line in lines]

    def _generate_status_screen(
        self, d: Dict[str, Any], is_connected: bool, age_seconds: float
    ) -> List[str]:
        lines: List[str] = []

        soc = d.get("battery_soc_pct", 50.0)
        b_mode = d.get("battery_mode", "IDLE")
        ev_socs = d.get("ev_socs_pct", [0.0, 0.0, 0.0])
        tariff = d.get("tariff_rate", 6.0)
        t_label = "LOW" if tariff <= 5.0 else ("PEAK" if tariff >= 9.0 else "MED")
        mqtt_status = "OK" if is_connected else "ERR"

        age_str = f"{min(99.9, age_seconds):.1f}s" if age_seconds < 100.0 else ">99s"

        if self.rows == 4:
            # 20x4 Screen 2
            lines.append(f"TARIFF: Rs{tariff:.1f} ({t_label})")
            lines.append(f"BAT: {soc:3.0f}% ({b_mode[:8]})")
            lines.append(f"EV: {ev_socs[0]:2.0f}%/{ev_socs[1]:2.0f}%/{ev_socs[2]:2.0f}%")
            lines.append(f"MQTT:{mqtt_status:<3} AGE:{age_str:>5}")
        else:
            # 16x2 Screen 2
            lines.append(f"BAT:{soc:2.0f}% EV:{ev_socs[0]:2.0f}/{ev_socs[1]:2.0f}")
            lines.append(f"T:{tariff:.1f} M:{mqtt_status} {age_str}")

        return [line.ljust(self.cols)[: self.cols] for line in lines]

    def _generate_no_data_screen(self, is_connected: bool, age_seconds: float) -> List[str]:
        lines: List[str] = []
        conn = "CONNECTED" if is_connected else "NO BROKER"
        if self.rows == 4:
            lines.append("*** SYSTEM ALERT ***")
            lines.append("NO SENSOR DATA      ")
            lines.append("SAFE GRID-ONLY MODE ")
            lines.append(f"MQTT: {conn[:12]}")
        else:
            lines.append("*** NO DATA *** ")
            lines.append("SAFE GRID-ONLY  ")
        return [line.ljust(self.cols)[: self.cols] for line in lines]

    def _write_lines(self, lines: List[str]):
        """Writes lines to hardware with differential caching or terminal mock."""
        if self.mock or self._hardware_lcd is None:
            self._render_mock(lines)
            return

        # Physical LCD with differential update and retry
        for row_idx, new_line in enumerate(lines[: self.rows]):
            old_line = self._buffer[row_idx]
            if new_line != old_line:
                self._safe_write_row(row_idx, new_line)
                self._buffer[row_idx] = new_line

    def _safe_write_row(self, row_idx: int, text: str, retries: int = 2):
        for attempt in range(retries + 1):
            try:
                self._hardware_lcd.cursor_pos = (row_idx, 0)
                self._hardware_lcd.write_string(text)
                return
            except Exception as e:
                if attempt == retries:
                    logger.error(f"I2C write error on row {row_idx}: {e}")
                time.sleep(0.01)

    def _render_mock(self, lines: List[str]):
        """Renders ASCII box display to stdout."""
        border = "+" + "-" * self.cols + "+"
        output = [border]
        for line in lines[: self.rows]:
            output.append(f"|{line}|")
        output.append(border)

        display_text = "\n".join(output)
        # Print with timestamp header
        header = f"--- [MOCK LCD {self.cols}x{self.rows} - Screen {self._current_screen + 1}] ---"
        print(f"\n{header}\n{display_text}")

    def clear(self):
        """Cleans up the screen on shutdown."""
        if not self.mock and self._hardware_lcd is not None:
            try:
                self._hardware_lcd.clear()
                self._hardware_lcd.backlight_enabled = False
            except Exception:
                pass
        print("\n[LCD] Screen cleared.")
