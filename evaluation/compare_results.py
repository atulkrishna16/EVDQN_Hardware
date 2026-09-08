from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent

BASELINE_DIR = (
    ROOT
    / "results"
    / "baselines"
)

DQN_DIR = (
    ROOT
    / "results"
    / "dqn"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FILES = {

    "UNCONTROLLED":
        BASELINE_DIR
        / "uncontrolled_results.csv",

    "RULE_BASED":
        BASELINE_DIR
        / "rule_based_results.csv",

    "DQN":
        DQN_DIR
        / "dqn_test_results.csv",
}


def calculate_metrics(
    name,
    df,
):

    peak_grid = (
        df["grid_imported_kW"]
        .max()
    )

    total_grid_energy = (
        df["grid_imported_kW"]
        .sum()
    )

    total_cost = (
        df["grid_cost_inr"]
        .sum()
    )

    overload_events = (
        df["grid_overload_kW"]
        > 0
    ).sum()

    total_overload = (
        df["grid_overload_kW"]
        .sum()
    )

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

        solar_utilization = 100.0

    average_self_sufficiency = (
        df[
            "self_sufficiency_pct"
        ].mean()
    )

    minimum_battery_soc = (
        df[
            "battery_soc_pct"
        ].min()
    )

    final_battery_soc = (
        df.iloc[-1][
            "battery_soc_pct"
        ]
    )

    final_ev0_soc = (
        df.iloc[-1][
            "ev0_soc_pct"
        ]
    )

    final_ev1_soc = (
        df.iloc[-1][
            "ev1_soc_pct"
        ]
    )

    final_ev2_soc = (
        df.iloc[-1][
            "ev2_soc_pct"
        ]
    )

    total_battery_throughput = (
        df[
            "battery_cycle_kW"
        ].sum()
    )

    return {

        "controller":
            name,

        "peak_grid_kW":
            peak_grid,

        "total_grid_energy_kWh":
            total_grid_energy,

        "total_grid_cost_INR":
            total_cost,

        "overload_events":
            overload_events,

        "total_overload_kWh":
            total_overload,

        "solar_generated_kWh":
            solar_generated,

        "solar_used_kWh":
            solar_used,

        "solar_curtailed_kWh":
            solar_curtailed,

        "solar_utilization_pct":
            solar_utilization,

        "average_self_sufficiency_pct":
            average_self_sufficiency,

        "minimum_battery_soc_pct":
            minimum_battery_soc,

        "final_battery_soc_pct":
            final_battery_soc,

        "final_ev0_soc_pct":
            final_ev0_soc,

        "final_ev1_soc_pct":
            final_ev1_soc,

        "final_ev2_soc_pct":
            final_ev2_soc,

        "battery_throughput_kWh":
            total_battery_throughput,
    }


def main():

    print("=" * 70)
    print("FINAL CONTROLLER COMPARISON")
    print("=" * 70)

    metrics = []

    for name, path in FILES.items():

        if not path.exists():

            print(
                f"WARNING: Missing {path}"
            )

            continue

        df = pd.read_csv(
            path
        )

        print(
            f"{name}: {len(df):,} rows"
        )

        metrics.append(
            calculate_metrics(
                name,
                df
            )
        )

    if not metrics:

        raise FileNotFoundError(
            "No result files found."
        )

    metrics_df = pd.DataFrame(
        metrics
    )

    output_path = (
        OUTPUT_DIR
        / "final_comparison_metrics.csv"
    )

    metrics_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\n"
        + metrics_df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output_path}"
    )

    # ========================================================
    # IMPROVEMENT
    # ========================================================

    uncontrolled = metrics_df[
        metrics_df["controller"]
        == "UNCONTROLLED"
    ]

    if len(uncontrolled) == 0:
        return

    base = uncontrolled.iloc[0]

    print(
        "\n"
        + "=" * 70
    )

    print(
        "IMPROVEMENT OVER UNCONTROLLED"
    )

    print(
        "=" * 70
    )

    for _, row in metrics_df.iterrows():

        if (
            row["controller"]
            == "UNCONTROLLED"
        ):
            continue

        peak_reduction = (
            (
                base["peak_grid_kW"]
                - row["peak_grid_kW"]
            )
            / base["peak_grid_kW"]
            * 100.0
        )

        cost_reduction = (
            (
                base[
                    "total_grid_cost_INR"
                ]
                - row[
                    "total_grid_cost_INR"
                ]
            )
            / base[
                "total_grid_cost_INR"
            ]
            * 100.0
        )

        overload_event_reduction = (
            (
                base[
                    "overload_events"
                ]
                - row[
                    "overload_events"
                ]
            )
            / max(
                1,
                base[
                    "overload_events"
                ]
            )
            * 100.0
        )

        overload_energy_reduction = (
            (
                base[
                    "total_overload_kWh"
                ]
                - row[
                    "total_overload_kWh"
                ]
            )
            / max(
                1e-6,
                base[
                    "total_overload_kWh"
                ]
            )
            * 100.0
        )

        solar_improvement = (
            row[
                "solar_utilization_pct"
            ]
            - base[
                "solar_utilization_pct"
            ]
        )

        self_sufficiency_improvement = (
            row[
                "average_self_sufficiency_pct"
            ]
            - base[
                "average_self_sufficiency_pct"
            ]
        )

        print(
            f"\n{row['controller']}"
        )

        print(
            f"Peak reduction: "
            f"{peak_reduction:.2f}%"
        )

        print(
            f"Cost reduction: "
            f"{cost_reduction:.2f}%"
        )

        print(
            f"Overload event reduction: "
            f"{overload_event_reduction:.2f}%"
        )

        print(
            f"Overload energy reduction: "
            f"{overload_energy_reduction:.2f}%"
        )

        print(
            f"Solar utilization improvement: "
            f"{solar_improvement:.2f} percentage points"
        )

        print(
            f"Self-sufficiency improvement: "
            f"{self_sufficiency_improvement:.2f} percentage points"
        )


if __name__ == "__main__":
    main()