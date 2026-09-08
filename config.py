# config.py

# ============================================================
# SIMULATION CONFIGURATION
# ============================================================

TIME_STEP_HOURS = 1.0


# ============================================================
# BATTERY
# ============================================================

BATTERY_CAPACITY_KWH = 100.0

BATTERY_INITIAL_SOC = 50.0
BATTERY_MIN_SOC = 20.0
BATTERY_MAX_SOC = 95.0

BATTERY_MAX_CHARGE_KW = 25.0
BATTERY_MAX_DISCHARGE_KW = 25.0

BATTERY_CHARGE_EFFICIENCY = 0.95
BATTERY_DISCHARGE_EFFICIENCY = 0.95


# ============================================================
# EV FLEET
# ============================================================

EV_CAPACITY_KWH = [
    50.0,
    50.0,
    50.0
]

EV_INITIAL_SOC = [
    25.0,
    40.0,
    20.0
]

EV_TARGET_SOC = [
    80.0,
    90.0,
    80.0
]

EV_MAX_CHARGE_KW = [
    7.0,
    7.0,
    7.0
]

EV_ARRIVAL_HOUR = [
    0,
    0,
    0
]

EV_DEPARTURE_HOUR = [
    20,
    22,
    22
]


# ============================================================
# GRID
# ============================================================

GRID_MAX_POWER_KW = 100.0


# ============================================================
# TARIFF
# ============================================================

LOW_TARIFF = 4.0
MEDIUM_TARIFF = 6.5
HIGH_TARIFF = 9.0


# ============================================================
# CONTROLLER
# ============================================================

# Minimum battery SOC reserve
BATTERY_RESERVE_SOC = 25.0

# SOC at which we consider battery sufficiently full
BATTERY_HIGH_SOC = 80.0

# Number of EVs
NUM_EVS = 3