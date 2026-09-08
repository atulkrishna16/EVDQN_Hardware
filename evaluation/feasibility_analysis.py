# ============================================================
# evaluation/feasibility_analysis.py
#
# Determines the physically achievable minimum grid import
# for every hour using the current battery and EV limits.
# ============================================================

import os
import pandas as pd


# ============================================================
# PATH
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    ROOT,
    "data",
    "energy_data.csv"
)

OUTPUT_PATH = os.path.join(
    ROOT,
    "data",
    "feasibility_analysis.csv"
)


# ============================================================
# SYSTEM PARAMETERS
# ============================================================

GRID_LIMIT = 100.0

BATTERY_MAX_DISCHARGE = 25.0

# Three EVs, each capable of 7 kW.
EV0_MAX = 7.0
EV1_MAX = 7.0
EV2_MAX = 7.0

TOTAL_EV_MAX = (
    EV0_MAX
    + EV1_MAX
    + EV2_MAX
)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    DATA_PATH
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required = [
    "hour",
    "solar_kW",
    "house_kW",
    "factory_kW",
    "hospital_kW",
    "ev_base_kW"
]

missing = [
    column
    for column in required
    if column not in data.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# ANALYSIS
# ============================================================

results = []


for _, row in data.iterrows():

    hour = float(
        row["hour"]
    )

    solar = float(
        row["solar_kW"]
    )

    house = float(
        row["house_kW"]
    )

    factory = float(
        row["factory_kW"]
    )

    hospital = float(
        row["hospital_kW"]
    )

    ev = float(
        row["ev_base_kW"]
    )


    # --------------------------------------------------------
    # Base demand
    # --------------------------------------------------------

    base_load = (

        house
        + factory
        + hospital
    )


    normal_total_load = (
        base_load
        + ev
    )


    # --------------------------------------------------------
    # Grid import without battery
    # --------------------------------------------------------

    grid_without_battery = max(

        0.0,

        normal_total_load
        - solar
    )


    # --------------------------------------------------------
    # Maximum battery support
    # --------------------------------------------------------

    battery_support = min(

        BATTERY_MAX_DISCHARGE,

        grid_without_battery
    )


    # --------------------------------------------------------
    # Grid after maximum battery discharge
    # --------------------------------------------------------

    grid_with_battery = max(

        0.0,

        grid_without_battery
        - battery_support
    )


    # --------------------------------------------------------
    # Theoretical minimum if all EV charging can be stopped
    # --------------------------------------------------------

    flexible_ev_reduction = min(

        ev,

        TOTAL_EV_MAX
    )


    theoretical_min_grid = max(

        0.0,

        grid_with_battery
        - flexible_ev_reduction
    )


    # --------------------------------------------------------
    # Required support to reach 100 kW
    # --------------------------------------------------------

    required_reduction = max(

        0.0,

        grid_without_battery
        - GRID_LIMIT
    )


    # --------------------------------------------------------
    # Maximum available flexibility
    # --------------------------------------------------------

    maximum_flexibility = (

        battery_support
        + flexible_ev_reduction
    )


    # --------------------------------------------------------
    # Is 100 kW physically achievable?
    # --------------------------------------------------------

    can_reach_grid_limit = (

        theoretical_min_grid
        <= GRID_LIMIT
    )


    # --------------------------------------------------------
    # Additional reduction needed
    # --------------------------------------------------------

    remaining_reduction = max(

        0.0,

        theoretical_min_grid
        - GRID_LIMIT
    )


    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if theoretical_min_grid <= GRID_LIMIT:

        status = "FEASIBLE"

    else:

        status = "PHYSICALLY IMPOSSIBLE"


    # --------------------------------------------------------
    # Save row
    # --------------------------------------------------------

    results.append({

        "hour":
            hour,

        "solar_kW":
            solar,

        "base_load_kW":
            base_load,

        "ev_load_kW":
            ev,

        "total_load_kW":
            normal_total_load,

        "grid_without_battery_kW":
            grid_without_battery,

        "max_battery_support_kW":
            battery_support,

        "grid_after_battery_kW":
            grid_with_battery,

        "max_ev_reduction_kW":
            flexible_ev_reduction,

        "theoretical_min_grid_kW":
            theoretical_min_grid,

        "required_reduction_to_100_kW":
            required_reduction,

        "remaining_reduction_kW":
            remaining_reduction,

        "grid_limit_feasible":
            can_reach_grid_limit,

        "status":
            status
    })


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(

    OUTPUT_PATH,

    index=False
)


# ============================================================
# DISPLAY
# ============================================================

print("\n")

print("=" * 90)

print(
    "SMART GRID PHYSICAL FEASIBILITY ANALYSIS"
)

print("=" * 90)


print(
    f"\nGrid limit: "
    f"{GRID_LIMIT:.2f} kW"
)

print(
    f"Maximum battery discharge: "
    f"{BATTERY_MAX_DISCHARGE:.2f} kW"
)

print(
    f"Maximum total EV flexibility: "
    f"{TOTAL_EV_MAX:.2f} kW"
)


print("\n")

print(
    results_df[
        [
            "hour",
            "solar_kW",
            "total_load_kW",
            "grid_without_battery_kW",
            "max_battery_support_kW",
            "max_ev_reduction_kW",
            "theoretical_min_grid_kW",
            "status"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# SUMMARY
# ============================================================

impossible_hours = results_df[
    results_df["status"]
    == "PHYSICALLY IMPOSSIBLE"
]


feasible_hours = results_df[
    results_df["status"]
    == "FEASIBLE"
]


print("\n")

print("=" * 90)

print(
    f"Hours where 100 kW is feasible: "
    f"{len(feasible_hours)}"
)

print(
    f"Hours where 100 kW is physically impossible: "
    f"{len(impossible_hours)}"
)


print(
    f"\nTheoretical minimum daily peak: "
    f"{results_df['theoretical_min_grid_kW'].max():.2f} kW"
)


print(
    f"Current uncontrolled peak: "
    f"{results_df['grid_without_battery_kW'].max():.2f} kW"
)


if len(impossible_hours) > 0:

    print("\n")

    print(
        "PHYSICALLY IMPOSSIBLE HOURS:"
    )

    print(
        impossible_hours[
            [
                "hour",
                "total_load_kW",
                "solar_kW",
                "theoretical_min_grid_kW"
            ]
        ].to_string(
            index=False
        )
    )


print("\n")

print(
    f"Saved analysis to:\n{OUTPUT_PATH}"
)

print("=" * 90)