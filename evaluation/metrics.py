# evaluation/metrics.py

import pandas as pd


def calculate_metrics(df):

    # ========================================================
    # GRID
    # ========================================================

    total_grid_energy = (
        df["grid_imported_kW"]
        .sum()
    )


    peak_grid_import = (
        df["grid_imported_kW"]
        .max()
    )


    total_grid_cost = (
        df["grid_cost_inr"]
        .sum()
    )


    overload_events = int(
        df["grid_overload"]
        .sum()
    )


    # ========================================================
    # SOLAR
    # ========================================================

    solar_generated = (
        df["solar_generated_kW"]
        .sum()
    )


    solar_used = (
        df["solar_used_kW"]
        .sum()
    )


    solar_curtailed = (
        df["solar_curtailed_kW"]
        .sum()
    )


    if solar_generated > 0:

        solar_utilization = (
            solar_used
            / solar_generated
            * 100.0
        )

    else:

        solar_utilization = 0.0


    # ========================================================
    # SELF SUFFICIENCY
    # ========================================================

    average_self_sufficiency = (
        df["self_sufficiency_pct"]
        .mean()
    )


    # ========================================================
    # BATTERY
    # ========================================================

    minimum_battery_soc = (
        df["battery_soc_pct"]
        .min()
    )


    maximum_battery_soc = (
        df["battery_soc_pct"]
        .max()
    )


    final_battery_soc = (
        df["battery_soc_pct"]
        .iloc[-1]
    )


    battery_charge_energy = (
        df["battery_charged_kW"]
        .sum()
    )


    battery_discharge_energy = (
        df["battery_discharged_kW"]
        .sum()
    )


    # ========================================================
    # EV
    # ========================================================

    final_ev0_soc = (
        df["ev0_soc_pct"]
        .iloc[-1]
    )


    final_ev1_soc = (
        df["ev1_soc_pct"]
        .iloc[-1]
    )


    final_ev2_soc = (
        df["ev2_soc_pct"]
        .iloc[-1]
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "Total Grid Energy (kWh)":
            total_grid_energy,

        "Peak Grid Import (kW)":
            peak_grid_import,

        "Total Grid Cost (INR)":
            total_grid_cost,

        "Grid Overload Events":
            overload_events,

        "Solar Generated (kWh)":
            solar_generated,

        "Solar Used (kWh)":
            solar_used,

        "Solar Curtailed (kWh)":
            solar_curtailed,

        "Solar Utilization (%)":
            solar_utilization,

        "Average Self-Sufficiency (%)":
            average_self_sufficiency,

        "Minimum Battery SOC (%)":
            minimum_battery_soc,

        "Maximum Battery SOC (%)":
            maximum_battery_soc,

        "Final Battery SOC (%)":
            final_battery_soc,

        "Battery Charge Energy (kWh)":
            battery_charge_energy,

        "Battery Discharge Energy (kWh)":
            battery_discharge_energy,

        "Final EV0 SOC (%)":
            final_ev0_soc,

        "Final EV1 SOC (%)":
            final_ev1_soc,

        "Final EV2 SOC (%)":
            final_ev2_soc
    }


def calculate_from_file(path):

    df = pd.read_csv(path)

    return calculate_metrics(df)