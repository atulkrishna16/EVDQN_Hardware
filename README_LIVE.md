# Live Smart Grid Dispatch Controller (Raspberry Pi Edge Deployment)

This directory contains the real-time production deployment package `live/` for running the trained DQN Smart Energy Grid Agent on a Raspberry Pi. The controller receives sensor telemetry from an ESP32/ESP8266 over MQTT, evaluates the optimal energy dispatch policy, calculates power flows across all generation/storage sources, and displays real-time metrics on an I2C LCD character display.

---

## 1. System Architecture

```text
+-------------------+              +-------------------------------------------------------------+
|   ESP32 / ESP8266 |              |                      Raspberry Pi                           |
|   Current Sensors | --(MQTT)-->  |  [MQTT Subscriber]                                          |
|   Solar / SOCs    |              |         |                                                   |
+-------------------+              |         v                                                   |
                                   |  [State Builder]  <-- Pi Clock / Tariff Schedule            |
                                   |         |                                                   |
                                   |         v                                                   |
                                   |  [DQN Inference Engine] (Torch or Pure-NumPy)               |
                                   |         |                                                   |
                                   |         v                                                   |
                                   |  [Energy Dispatcher] (Battery, EV, and Grid Physics)        |
                                   |         |                                                   |
                                   |         v                                                   |
                                   |  [LCD Renderer]  --> 20x4 / 16x2 I2C Display (or Mock CLI)  |
                                   +-------------------------------------------------------------+
```

---

## 2. Hardware Wiring (Raspberry Pi to I2C LCD)

Target display: **20x4 or 16x2 HD44780 LCD** with **PCF8574 I2C Backpack**.

```text
  Raspberry Pi 40-Pin Header                   PCF8574 I2C Backpack
  ==========================                   ====================
  Pin 02 / 04 [5.0V Power]  -----------------> VCC
  Pin 06      [Ground]      -----------------> GND
  Pin 03      [GPIO 2 / SDA] ----------------> SDA
  Pin 05      [GPIO 3 / SCL] ----------------> SCL
```

> **Note on LCD Contrast**: If the display lights up but shows no characters, turn the blue potentiometer on the back of the PCF8574 backpack with a small screwdriver until text is clearly visible.

---

## 3. MQTT Specification & Payload Schema

### Primary Telemetry Topic
`grid/esp1/telemetry`

### JSON Payload Schema
The ESP publishes one JSON telemetry frame per sampling interval (e.g. every 2 to 5 seconds):

```json
{
  "timestamp": 1729415400.0,
  "load_kw": 42.50,
  "next_load_kw": 45.00,
  "solar_kw": 18.20,
  "battery_soc": 63.50,
  "ev0_soc": 45.00,
  "ev1_soc": 60.00,
  "ev2_soc": 30.00
}
```

| Field | Type | Unit | Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | Float | Unix Epoch (s) | `> 0` | Timestamp when sample was taken |
| `load_kw` | Float | kW | `0.0 – 500.0` | Measured total base electrical load |
| `next_load_kw` | Float | kW | `0.0 – 500.0` | Forecasted next-hour load (optional; defaults to `load_kw`) |
| `solar_kw` | Float | kW | `0.0 – 500.0` | Measured solar PV power output |
| `battery_soc` | Float | % | `0.0 – 100.0` | Battery state of charge |
| `ev0_soc` | Float | % | `0.0 – 100.0` | Electric Vehicle 1 state of charge |
| `ev1_soc` | Float | % | `0.0 – 100.0` | Electric Vehicle 2 state of charge |
| `ev2_soc` | Float | % | `0.0 – 100.0` | Electric Vehicle 3 state of charge |

### Manual Test Command (`mosquitto_pub`)
```bash
mosquitto_pub -h localhost -p 1883 -t "grid/esp1/telemetry" -m '{
  "timestamp": 1729415400,
  "load_kw": 42.0,
  "solar_kw": 18.0,
  "battery_soc": 63.0,
  "ev0_soc": 45.0,
  "ev1_soc": 60.0,
  "ev2_soc": 30.0
}'
```

---

## 4. ESP32 / ESP8266 Arduino C++ Firmware Snippet

Install `PubSubClient` and `ArduinoJson` via Arduino Library Manager or PlatformIO:

```cpp
#include <WiFi.h>           // Use <ESP8266WiFi.h> on ESP8266
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* mqtt_broker = "192.168.1.50"; // IP address of Raspberry Pi
const int mqtt_port = 1883;
const char* telemetry_topic = "grid/esp1/telemetry";

WiFiClient espClient;
PubSubClient client(espClient);

void setup_wifi() {
    delay(10);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }
}

void reconnect() {
    while (!client.connected()) {
        if (client.connect("ESP32_Grid_Sensor")) {
            // Connected
        } else {
            delay(2000);
        }
    }
}

void setup() {
    Serial.begin(115200);
    setup_wifi();
    client.setServer(mqtt_broker, mqtt_port);
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();

    // Read real sensors (ADC / CT sensors / Inverter RS485 Modbus)
    float load_kw = analogRead(34) * (100.0 / 4095.0);
    float solar_kw = analogRead(35) * (60.0 / 4095.0);
    float battery_soc = 65.0; // From BMS
    float ev0_soc = 50.0;
    float ev1_soc = 70.0;
    float ev2_soc = 40.0;

    StaticJsonDocument<256> doc;
    doc["timestamp"] = (double)millis() / 1000.0;
    doc["load_kw"] = load_kw;
    doc["solar_kw"] = solar_kw;
    doc["battery_soc"] = battery_soc;
    doc["ev0_soc"] = ev0_soc;
    doc["ev1_soc"] = ev1_soc;
    doc["ev2_soc"] = ev2_soc;

    char buffer[256];
    serializeJson(doc, buffer);
    client.publish(telemetry_topic, buffer);

    delay(2000); // Sample every 2 seconds
}
```

---

## 5. Raspberry Pi Setup Instructions (Debian Bookworm 64-bit)

### Step 1: Enable I2C Interface
```bash
sudo raspi-config
# Navigate to: 3 Interface Options -> I5 I2C -> Enable -> Yes
sudo reboot
```
Verify the I2C bus detects the LCD backpack (usually address `0x27` or `0x3F`):
```bash
sudo apt update && sudo apt install -y i2c-tools mosquitto mosquitto-clients
sudo i2cdetect -y 1
```

### Step 2: Configure Mosquitto Broker on the Pi
Edit `/etc/mosquitto/conf.d/smart_grid.conf`:
```text
listener 1883
allow_anonymous true
```
Restart and enable the broker:
```bash
sudo systemctl restart mosquitto
sudo systemctl enable mosquitto
```

### Step 3: Set up Python Environment
```bash
cd /home/pi/smart-energy-grid
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements_pi.txt
```

### Step 4: Run Tests
```bash
pytest -v tests/
```

---

## 6. Running the Controller

### A. Terminal Mock Mode (No Hardware Required)
Test locally on any laptop or Pi without physical LCD or ESP connected:
```bash
python live/main_live.py --mock-lcd
```

### B. End-to-End Simulation with Fake ESP
In Terminal 1 (Run Controller):
```bash
python live/main_live.py --mock-lcd
```
In Terminal 2 (Run Sensor Publisher Simulator):
```bash
python simulator/fake_esp.py --interval 2.0
```

### C. Fast-Forward Demo Mode
Advances the simulated clock by 1 hour on every tick so you can observe an entire 24-hour tariff cycle and solar peak in under 2 minutes:
```bash
python live/main_live.py --mock-lcd --fast-forward
```

### D. Production Deployment with Systemd Service
```bash
sudo cp smart-grid-live.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable smart-grid-live
sudo systemctl start smart-grid-live
```
Check status and live logs:
```bash
sudo systemctl status smart-grid-live
journalctl -u smart-grid-live -f
```

---

## 7. Rotating LCD Displays

### Screen 1: Real-Time Power Split
Displays instantaneous power drawn from every source and its percentage share of total supply:
```text
+--------------------+
|GRID   42.0kW  60%  |
|SOLAR  18.0kW  25%  |
|BATT   10.5kW  15%  |
|T:PEAK  SOC: 63%    |
+--------------------+
```
*(When the battery is charging, line 3 shows `BATT  CHG: 12.5kW`)*

### Screen 2: System & Network Status
Displays tariff rates, battery/EV charging state, MQTT health, and telemetry freshness:
```text
+--------------------+
|TARIFF: Rs9.0 (PEAK)|
|BAT:  63% (DISCHARG)|
|EV: 45%/60%/30%     |
|MQTT:OK  AGE:  1.2s |
+--------------------+
```

### Safety Alert: Missing Telemetry ("NO DATA")
If telemetry stops arriving for more than `data_stale_timeout_seconds` (default 15s), the controller switches to a safe hold dispatch (battery idle, EVs held) and triggers the alert:
```text
+--------------------+
|*** SYSTEM ALERT ***|
|NO SENSOR DATA      |
|SAFE GRID-ONLY MODE |
|MQTT: CONNECTED     |
+--------------------+
```

---

## 8. Decision Timing and Demo vs. Real Deployment

| Parameter | Demo Setting | Real-World Deployment | Rationale |
| :--- | :--- | :--- | :--- |
| `decision_interval_seconds` | `2.0 s` | `5.0 – 10.0 s` | Frequent polling in demo allows fast interactive feedback when sensor readings change. |
| `dt_hours` | `1.0` | `1.0` | The DQN agent was trained on 1-hour time horizons. Energy balance calculations use `dt_hours=1.0` to match state transition dynamics. |
| `fast_forward` | `true` | `false` | Real deployment uses the Pi's local RTC system clock for actual time-of-day tariffs. |
