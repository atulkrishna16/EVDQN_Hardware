from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


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

PLOT_DIR = (
    ROOT
    / "results"
    / "plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results():

    uncontrolled = pd.read_csv(
        BASELINE_DIR
        / "uncontrolled_results.csv"
    )

    rule_based = pd.read_csv(
        BASELINE_DIR
        / "rule_based_results.csv"
    )

    dqn = pd.read_csv(
        DQN_DIR
        / "dqn_test_results.csv"
    )

    return (
        uncontrolled,
        rule_based,
        dqn
    )


# ============================================================
# PREPARE DATE
# ============================================================

def prepare_dates(df):

    df = df.copy()

    if "date" not in df.columns:

        raise ValueError(
            "The result CSV does not contain a 'date' column."
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["calendar_date"] = (
        df["date"]
        .dt.date
    )

    df["actual_hour"] = (
        df["date"]
        .dt.hour
    )

    return df


# ============================================================
# FIND A COMPLETE 24-HOUR DAY
# ============================================================

def get_complete_day(
    uncontrolled,
    rule_based,
    dqn,
):

    uncontrolled = prepare_dates(
        uncontrolled
    )

    rule_based = prepare_dates(
        rule_based
    )

    dqn = prepare_dates(
        dqn
    )

    # Find dates that contain all 24 hours
    def complete_dates(df):

        counts = (
            df.groupby(
                "calendar_date"
            )["actual_hour"]
            .nunique()
        )

        return set(
            counts[
                counts >= 24
            ].index
        )

    common_dates = (
        complete_dates(uncontrolled)
        & complete_dates(rule_based)
        & complete_dates(dqn)
    )

    if not common_dates:

        raise ValueError(
            "No complete 24-hour date was found "
            "in all three result files."
        )

    selected_date = sorted(
        common_dates
    )[0]

    print(
        f"\nSelected 24-hour test date: "
        f"{selected_date}"
    )

    def extract(df):

        day = df[
            df["calendar_date"]
            == selected_date
        ].copy()

        day = (
            day
            .sort_values("actual_hour")
            .drop_duplicates(
                subset=["actual_hour"]
            )
        )

        # Ensure exactly 24 hours
        day = day[
            day["actual_hour"].between(
                0,
                23
            )
        ]

        if len(day) != 24:

            raise ValueError(
                f"Selected date {selected_date} "
                f"does not contain exactly 24 hourly rows."
            )

        return day.reset_index(
            drop=True
        )

    return (
        extract(uncontrolled),
        extract(rule_based),
        extract(dqn),
        selected_date,
    )


# ============================================================
# GRID IMPORT - 24 HOURS
# ============================================================

def plot_24h_grid(
    uncontrolled,
    rule_based,
    dqn,
    selected_date,
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        uncontrolled["actual_hour"],
        uncontrolled["grid_imported_kW"],
        label="Uncontrolled"
    )

    plt.plot(
        rule_based["actual_hour"],
        rule_based["grid_imported_kW"],
        label="Rule-Based"
    )

    plt.plot(
        dqn["actual_hour"],
        dqn["grid_imported_kW"],
        label="DQN"
    )

    plt.axhline(
        100.0,
        linestyle="--",
        label="Grid Limit"
    )

    plt.xlabel(
        "Hour of day"
    )

    plt.ylabel(
        "Grid import (kW)"
    )

    plt.title(
        f"24-Hour Grid Import Comparison - {selected_date}"
    )

    plt.xticks(
        range(0, 24, 2)
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "01_24h_grid_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# BATTERY SOC - 24 HOURS
# ============================================================

def plot_24h_battery(
    uncontrolled,
    rule_based,
    dqn,
    selected_date,
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        uncontrolled["actual_hour"],
        uncontrolled["battery_soc_pct"],
        label="Uncontrolled"
    )

    plt.plot(
        rule_based["actual_hour"],
        rule_based["battery_soc_pct"],
        label="Rule-Based"
    )

    plt.plot(
        dqn["actual_hour"],
        dqn["battery_soc_pct"],
        label="DQN"
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
        "Hour of day"
    )

    plt.ylabel(
        "Battery SOC (%)"
    )

    plt.title(
        f"24-Hour Battery SOC Comparison - {selected_date}"
    )

    plt.xticks(
        range(0, 24, 2)
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "02_24h_battery_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# EV SOC - 24 HOURS
# ============================================================

def plot_24h_ev(
    uncontrolled,
    rule_based,
    dqn,
    selected_date,
):

    ev_columns = [
        (
            "ev0_soc_pct",
            "EV0"
        ),
        (
            "ev1_soc_pct",
            "EV1"
        ),
        (
            "ev2_soc_pct",
            "EV2"
        ),
    ]

    for column, name in ev_columns:

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            uncontrolled["actual_hour"],
            uncontrolled[column],
            label="Uncontrolled"
        )

        plt.plot(
            rule_based["actual_hour"],
            rule_based[column],
            label="Rule-Based"
        )

        plt.plot(
            dqn["actual_hour"],
            dqn[column],
            label="DQN"
        )

        plt.xlabel(
            "Hour of day"
        )

        plt.ylabel(
            "SOC (%)"
        )

        plt.title(
            f"24-Hour {name} SOC Comparison - {selected_date}"
        )

        plt.xticks(
            range(0, 24, 2)
        )

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        path = (
            PLOT_DIR
            / f"03_24h_{name.lower()}_soc.png"
        )

        plt.savefig(
            path,
            dpi=300
        )

        plt.close()

        print(
            f"Saved: {path}"
        )


# ============================================================
# SOLAR - 24 HOURS
# ============================================================

def plot_24h_solar(
    dqn,
    selected_date,
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        dqn["actual_hour"],
        dqn["solar_generated_kW"],
        label="Solar generated"
    )

    plt.plot(
        dqn["actual_hour"],
        dqn["solar_used_kW"],
        label="Solar used"
    )

    plt.xlabel(
        "Hour of day"
    )

    plt.ylabel(
        "Solar power (kW)"
    )

    plt.title(
        f"24-Hour Solar Generation and Utilization - {selected_date}"
    )

    plt.xticks(
        range(0, 24, 2)
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "04_24h_solar.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# FULL TEST COST
# ============================================================

def plot_full_test_cost(
    uncontrolled,
    rule_based,
    dqn,
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        uncontrolled["grid_cost_inr"]
        .cumsum()
        .values,
        label="Uncontrolled"
    )

    plt.plot(
        rule_based["grid_cost_inr"]
        .cumsum()
        .values,
        label="Rule-Based"
    )

    plt.plot(
        dqn["grid_cost_inr"]
        .cumsum()
        .values,
        label="DQN"
    )

    plt.xlabel(
        "Test hour"
    )

    plt.ylabel(
        "Cumulative grid cost (INR)"
    )

    plt.title(
        "Cumulative Electricity Cost - Full ETT Test Period"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "05_full_test_cost_comparison.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# FULL TEST GRID
# ============================================================

def plot_full_test_grid(
    uncontrolled,
    rule_based,
    dqn,
):

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        uncontrolled["grid_imported_kW"],
        label="Uncontrolled"
    )

    plt.plot(
        rule_based["grid_imported_kW"],
        label="Rule-Based"
    )

    plt.plot(
        dqn["grid_imported_kW"],
        label="DQN"
    )

    plt.axhline(
        100,
        linestyle="--",
        label="Grid Limit"
    )

    plt.xlabel(
        "Test hour"
    )

    plt.ylabel(
        "Grid import (kW)"
    )

    plt.title(
        "Grid Import - Full ETT Test Period"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = (
        PLOT_DIR
        / "06_full_test_grid_import.png"
    )

    plt.savefig(
        path,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# TRAINING REWARD
# ============================================================

def plot_training_reward():

    path = (
        DQN_DIR
        / "dqn_training_history.csv"
    )

    if not path.exists():

        print(
            "Training history not found."
        )

        return

    df = pd.read_csv(
        path
    )

    if "episode" not in df.columns:

        print(
            "Training history missing episode column."
        )

        return

    reward_column = None

    for candidate in [
        "reward",
        "episode_reward",
        "total_reward",
    ]:

        if candidate in df.columns:

            reward_column = candidate
            break

    if reward_column is None:

        print(
            "Training history missing reward column."
        )

        return

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        df["episode"],
        df[reward_column]
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Episode reward"
    )

    plt.title(
        "DQN Training Reward"
    )

    plt.grid(True)
    plt.tight_layout()

    output = (
        PLOT_DIR
        / "07_dqn_training_reward.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# TRAINING LOSS
# ============================================================

def plot_training_loss():

    path = (
        DQN_DIR
        / "dqn_training_history.csv"
    )

    if not path.exists():

        return

    df = pd.read_csv(
        path
    )

    if (
        "episode" not in df.columns
        or "loss" not in df.columns
    ):

        return

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        df["episode"],
        df["loss"]
    )

    plt.xlabel(
        "Training episode"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "DQN Training Loss"
    )

    plt.grid(True)
    plt.tight_layout()

    output = (
        PLOT_DIR
        / "08_dqn_training_loss.png"
    )

    plt.savefig(
        output,
        dpi=300
    )

    plt.close()

    print(
        f"Saved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GENERATING FINAL ETT PLOTS")
    print("=" * 70)

    (
        uncontrolled,
        rule_based,
        dqn
    ) = load_results()

    (
        u_day,
        r_day,
        d_day,
        selected_date,
    ) = get_complete_day(
        uncontrolled,
        rule_based,
        dqn
    )

    # --------------------------------------------------------
    # 24-hour plots
    # --------------------------------------------------------

    plot_24h_grid(
        u_day,
        r_day,
        d_day,
        selected_date
    )

    plot_24h_battery(
        u_day,
        r_day,
        d_day,
        selected_date
    )

    plot_24h_ev(
        u_day,
        r_day,
        d_day,
        selected_date
    )

    plot_24h_solar(
        d_day,
        selected_date
    )

    # --------------------------------------------------------
    # Full test plots
    # --------------------------------------------------------

    plot_full_test_cost(
        uncontrolled,
        rule_based,
        dqn
    )

    plot_full_test_grid(
        uncontrolled,
        rule_based,
        dqn
    )

    # --------------------------------------------------------
    # Training plots
    # --------------------------------------------------------

    plot_training_reward()

    plot_training_loss()

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL PLOT GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()