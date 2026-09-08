# plot_results.py

import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

DATA = os.path.join(
    ROOT,
    "data"
)


# ============================================================
# LOAD RESULTS
# ============================================================

uncontrolled_path = os.path.join(
    DATA,
    "uncontrolled_results.csv"
)

rule_based_path = os.path.join(
    DATA,
    "baseline_results.csv"
)


uncontrolled = pd.read_csv(
    uncontrolled_path
)

rule_based = pd.read_csv(
    rule_based_path
)


hours = uncontrolled["hour"]


# ============================================================
# CHECK SAME INPUT PROFILE
# ============================================================

if not (
    uncontrolled["solar_generated_kW"]
    .equals(
        rule_based["solar_generated_kW"]
    )
):

    print(
        "WARNING: Solar profiles differ!"
    )


if not (
    uncontrolled["house_load_kW"]
    .equals(
        rule_based["house_load_kW"]
    )
):

    print(
        "WARNING: House loads differ!"
    )


if not (
    uncontrolled["factory_load_kW"]
    .equals(
        rule_based["factory_load_kW"]
    )
):

    print(
        "WARNING: Factory loads differ!"
    )


if not (
    uncontrolled["hospital_load_kW"]
    .equals(
        rule_based["hospital_load_kW"]
    )
):

    print(
        "WARNING: Hospital loads differ!"
    )


# ============================================================
# 1. GRID IMPORT COMPARISON
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    hours,
    uncontrolled["grid_imported_kW"],
    label="Uncontrolled",
    linewidth=2
)


plt.plot(
    hours,
    rule_based["grid_imported_kW"],
    label="Rule-Based",
    linewidth=2
)


plt.axhline(
    100,
    linestyle="--",
    label="Grid Limit"
)


plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Grid Import (kW)"
)


plt.title(
    "Uncontrolled vs Rule-Based — Grid Import"
)


plt.xticks(
    range(0, 24, 2)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        DATA,
        "plot_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. BATTERY SOC
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    hours,
    uncontrolled["battery_soc_pct"],
    label="Uncontrolled",
    linewidth=2
)


plt.plot(
    hours,
    rule_based["battery_soc_pct"],
    label="Rule-Based",
    linewidth=2
)


plt.axhline(
    20,
    linestyle="--",
    label="Minimum SOC"
)


plt.axhline(
    95,
    linestyle="--",
    label="Maximum SOC"
)


plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Battery SOC (%)"
)


plt.title(
    "Battery State-of-Charge Comparison"
)


plt.xticks(
    range(0, 24, 2)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        DATA,
        "plot_battery_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. EV SOC
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    hours,
    uncontrolled["ev0_soc_pct"],
    label="Uncontrolled EV-0",
    linestyle="--"
)

plt.plot(
    hours,
    uncontrolled["ev1_soc_pct"],
    label="Uncontrolled EV-1",
    linestyle="--"
)

plt.plot(
    hours,
    uncontrolled["ev2_soc_pct"],
    label="Uncontrolled EV-2",
    linestyle="--"
)


plt.plot(
    hours,
    rule_based["ev0_soc_pct"],
    label="Rule-Based EV-0",
    linewidth=2
)

plt.plot(
    hours,
    rule_based["ev1_soc_pct"],
    label="Rule-Based EV-1",
    linewidth=2
)

plt.plot(
    hours,
    rule_based["ev2_soc_pct"],
    label="Rule-Based EV-2",
    linewidth=2
)


plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "SOC (%)"
)


plt.title(
    "EV Fleet State-of-Charge Comparison"
)


plt.xticks(
    range(0, 24, 2)
)

plt.legend(
    fontsize=8
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        DATA,
        "plot_ev_comparison.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 4. SOLAR
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    hours,
    rule_based["solar_generated_kW"],
    label="Solar Generated",
    linewidth=2
)


plt.plot(
    hours,
    rule_based["solar_used_kW"],
    label="Solar Used",
    linewidth=2
)


plt.plot(
    hours,
    rule_based["solar_curtailed_kW"],
    label="Solar Curtailed",
    linewidth=2
)


plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Power (kW)"
)


plt.title(
    "Solar Generation, Utilization and Curtailment"
)


plt.xticks(
    range(0, 24, 2)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        DATA,
        "plot_solar.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 5. GRID COST
# ============================================================

plt.figure(
    figsize=(12, 6)
)


plt.plot(
    hours,
    uncontrolled["grid_cost_inr"],
    label="Uncontrolled",
    linewidth=2
)


plt.plot(
    hours,
    rule_based["grid_cost_inr"],
    label="Rule-Based",
    linewidth=2
)


plt.xlabel(
    "Hour of Day"
)

plt.ylabel(
    "Grid Cost (INR/hour)"
)


plt.title(
    "Hourly Electricity Cost Comparison"
)


plt.xticks(
    range(0, 24, 2)
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        DATA,
        "plot_cost_comparison.png"
    ),
    dpi=300
)

plt.close()


print("\nPlots regenerated successfully.")

print(
    "\nGenerated files:"
)

print(
    "  data/plot_comparison.png"
)

print(
    "  data/plot_battery_comparison.png"
)

print(
    "  data/plot_ev_comparison.png"
)

print(
    "  data/plot_solar.png"
)

print(
    "  data/plot_cost_comparison.png"
)