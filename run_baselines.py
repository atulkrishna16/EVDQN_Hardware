from pathlib import Path

import pandas as pd

from environment.smart_grid_env import SmartGridEnv
from baselines.uncontrolled import get_uncontrolled_action
from baselines.rule_based import get_rule_based_action


ROOT = Path(__file__).resolve().parent

TEST_DATA = (
    ROOT
    / "data"
    / "processed"
    / "test.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "baselines"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def run_controller(
    controller_name,
    action_function,
    output_path,
):

    # ========================================================
    # IMPORTANT
    # ========================================================
    # Entire test dataset is ONE continuous episode.
    #
    # Battery SOC persists.
    # EV SOC persists.
    #
    # No 24-hour reset.
    # ========================================================

    test_df = pd.read_csv(
        TEST_DATA
    )

    env = SmartGridEnv(
        data_path=TEST_DATA,
        episode_length=len(test_df),
        random_start=False,
    )

    state = env.reset(
        start_index=0
    )

    done = False

    records = []

    while not done:

        action = action_function(
            env
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

        row["controller"] = (
            controller_name
        )

        row["action"] = action

        records.append(
            row
        )

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    return result


def main():

    print("=" * 70)
    print("SEQUENTIAL BASELINE EVALUATION")
    print("=" * 70)

    print(
        "\nTest dataset:"
    )

    print(
        TEST_DATA
    )

    test_df = pd.read_csv(
        TEST_DATA
    )

    print(
        f"Test hours: {len(test_df):,}"
    )

    # ========================================================
    # UNCONTROLLED
    # ========================================================

    print(
        "\nRunning UNCONTROLLED..."
    )

    run_controller(
        controller_name="UNCONTROLLED",
        action_function=get_uncontrolled_action,
        output_path=(
            RESULTS_DIR
            / "uncontrolled_results.csv"
        ),
    )

    # ========================================================
    # RULE BASED
    # ========================================================

    print(
        "\nRunning RULE-BASED..."
    )

    run_controller(
        controller_name="RULE_BASED",
        action_function=get_rule_based_action,
        output_path=(
            RESULTS_DIR
            / "rule_based_results.csv"
        ),
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEQUENTIAL BASELINE EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()