from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run_script(script):

    print("\n")
    print("=" * 70)
    print(
        f"RUNNING: {script}"
    )
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / script)
        ]
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{script} failed."
        )


def main():

    print("=" * 70)
    print("SMART ENERGY GRID MANAGEMENT")
    print("FINAL EXPERIMENT PIPELINE")
    print("=" * 70)

    # ========================================================
    # 1. Sequential baselines
    # ========================================================

    run_script(
        "run_baselines.py"
    )

    # ========================================================
    # 2. DQN training
    # ========================================================

    run_script(
        "train_dqn.py"
    )

    # ========================================================
    # 3. Sequential DQN test
    # ========================================================

    run_script(
        "evaluation/evaluate_dqn.py"
    )

    # ========================================================
    # 4. Final metrics
    # ========================================================

    run_script(
        "evaluation/compare_results.py"
    )

    # ========================================================
    # 5. Final plots
    # ========================================================

    run_script(
        "evaluation/plot_results.py"
    )

    print("\n")
    print("=" * 70)
    print("FINAL EXPERIMENT COMPLETED")
    print("=" * 70)

    print(
        "\nResults:"
    )

    print(
        "results/analysis/"
    )

    print(
        "results/plots/"
    )


if __name__ == "__main__":
    main()