from pathlib import Path

import pandas as pd

from environment.smart_grid_env import SmartGridEnv
from agents.dqn_agent import DQNAgent


ROOT = Path(__file__).resolve().parent.parent

TEST_DATA = (
    ROOT
    / "data"
    / "processed"
    / "test.csv"
)

MODEL_PATH = (
    ROOT
    / "results"
    / "dqn"
    / "best_dqn_model.pth"
)

OUTPUT_PATH = (
    ROOT
    / "results"
    / "dqn"
    / "dqn_test_results.csv"
)


def main():

    print("=" * 70)
    print("SEQUENTIAL DQN TEST")
    print("=" * 70)

    test_df = pd.read_csv(
        TEST_DATA
    )

    test_length = len(
        test_df
    )

    print(
        f"Test hours: {test_length:,}"
    )

    # ========================================================
    # ENTIRE TEST PERIOD = ONE EPISODE
    # ========================================================

    env = SmartGridEnv(
        data_path=TEST_DATA,
        episode_length=test_length,
        random_start=False,
    )

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
    )

    agent.load(
        MODEL_PATH
    )

    # Absolutely no exploration
    agent.epsilon = 0.0

    state = env.reset(
        start_index=0
    )

    done = False

    records = []

    while not done:

        action = agent.select_action(
            state,
            training=False
        )

        (
            state,
            reward,
            done,
            info,
        ) = env.step(
            action
        )

        row = info.copy()

        row["controller"] = "DQN"

        row["action"] = action

        records.append(
            row
        )

    results = pd.DataFrame(
        records
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )

    print(
        f"Rows generated: {len(results):,}"
    )

    # ========================================================
    # QUICK SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DQN TEST SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Peak grid: "
        f"{results['grid_imported_kW'].max():.2f} kW"
    )

    print(
        f"Total grid energy: "
        f"{results['grid_imported_kW'].sum():.2f} kWh"
    )

    print(
        f"Total grid cost: "
        f"₹{results['grid_cost_inr'].sum():.2f}"
    )

    print(
        f"Overload events: "
        f"{(results['grid_overload_kW'] > 0).sum()}"
    )

    print(
        f"Total overload: "
        f"{results['grid_overload_kW'].sum():.2f} kWh"
    )

    print(
        f"Minimum battery SOC: "
        f"{results['battery_soc_pct'].min():.2f}%"
    )

    print(
        f"Final battery SOC: "
        f"{results.iloc[-1]['battery_soc_pct']:.2f}%"
    )

    print(
        f"Final EV0 SOC: "
        f"{results.iloc[-1]['ev0_soc_pct']:.2f}%"
    )

    print(
        f"Final EV1 SOC: "
        f"{results.iloc[-1]['ev1_soc_pct']:.2f}%"
    )

    print(
        f"Final EV2 SOC: "
        f"{results.iloc[-1]['ev2_soc_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()