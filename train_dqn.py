from pathlib import Path

import numpy as np
import pandas as pd

from environment.smart_grid_env import SmartGridEnv
from agents.dqn_agent import DQNAgent


ROOT = Path(__file__).resolve().parent

TRAIN_DATA = (
    ROOT
    / "data"
    / "processed"
    / "train.csv"
)

VALIDATION_DATA = (
    ROOT
    / "data"
    / "processed"
    / "validation.csv"
)

RESULTS_DIR = (
    ROOT
    / "results"
    / "dqn"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_PATH = (
    RESULTS_DIR
    / "best_dqn_model.pth"
)

HISTORY_PATH = (
    RESULTS_DIR
    / "dqn_training_history.csv"
)


SEED = 42

EPISODES = 1000

EPISODE_LENGTH = 24

VALIDATION_INTERVAL = 50


def evaluate_validation(
    agent,
    env,
):

    starts = list(
        range(
            0,
            len(env.df)
            - env.episode_length
            + 1,
            env.episode_length
        )
    )

    # Evaluate a representative subset
    starts = starts[:30]

    total_cost = 0.0
    total_overload = 0.0
    peak_grid = 0.0
    total_reward = 0.0

    for start in starts:

        state = env.reset(
            start_index=start
        )

        done = False

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

            total_cost += (
                info["grid_cost_inr"]
            )

            total_overload += (
                info["grid_overload_kW"]
            )

            peak_grid = max(
                peak_grid,
                info["grid_imported_kW"]
            )

            total_reward += reward

    return {
        "cost": total_cost,
        "overload": total_overload,
        "peak": peak_grid,
        "reward": total_reward,
    }


def main():

    print("=" * 70)
    print("DQN TRAINING")
    print("=" * 70)

    print(
        f"Training data: {TRAIN_DATA}"
    )

    train_env = SmartGridEnv(
        data_path=TRAIN_DATA,
        episode_length=EPISODE_LENGTH,
        random_start=True,
        seed=SEED,
    )

    validation_env = SmartGridEnv(
        data_path=VALIDATION_DATA,
        episode_length=EPISODE_LENGTH,
        random_start=False,
        seed=SEED,
    )

    agent = DQNAgent(
        state_size=train_env.state_size,
        action_size=train_env.action_size,

        learning_rate=0.0005,

        gamma=0.95,

        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,

        batch_size=64,

        buffer_size=100000,

        target_update_frequency=250,

        seed=SEED,
    )

    history = []

    best_score = float(
        "inf"
    )

    for episode in range(
        1,
        EPISODES + 1
    ):

        state = train_env.reset()

        done = False

        episode_reward = 0.0
        episode_cost = 0.0
        episode_overload = 0.0
        episode_peak = 0.0

        losses = []

        while not done:

            action = agent.select_action(
                state,
                training=True
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = train_env.step(
                action
            )

            agent.remember(
                state,
                action,
                reward,
                next_state,
                done,
            )

            loss = agent.train_step()

            if loss is not None:
                losses.append(loss)

            state = next_state

            episode_reward += reward

            episode_cost += (
                info["grid_cost_inr"]
            )

            episode_overload += (
                info["grid_overload_kW"]
            )

            episode_peak = max(
                episode_peak,
                info["grid_imported_kW"]
            )

        agent.decay_epsilon()

        average_loss = (
            np.mean(losses)
            if losses
            else 0.0
        )

        record = {
            "episode": episode,

            "reward": episode_reward,

            "cost": episode_cost,

            "overload":
                episode_overload,

            "peak_grid":
                episode_peak,

            "loss":
                average_loss,

            "epsilon":
                agent.epsilon,
        }

        history.append(
            record
        )

        # ----------------------------------------------------
        # Print progress
        # ----------------------------------------------------

        if (
            episode == 1
            or episode % 10 == 0
        ):

            print(
                f"Episode {episode:4d}/{EPISODES} | "
                f"Reward: {episode_reward:9.2f} | "
                f"Cost: ₹{episode_cost:8.2f} | "
                f"Peak: {episode_peak:6.2f} kW | "
                f"Overload: {episode_overload:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            episode % VALIDATION_INTERVAL
            == 0
        ):

            validation = evaluate_validation(
                agent,
                validation_env
            )

            # Weighted validation score
            #
            # Overload is intentionally expensive.
            score = (
                validation["cost"]
                + 100.0
                * validation["overload"]
                + 20.0
                * validation["peak"]
            )

            print(
                "\n"
                "Validation | "
                f"Cost: ₹{validation['cost']:.2f} | "
                f"Peak: {validation['peak']:.2f} kW | "
                f"Overload: {validation['overload']:.2f}"
            )

            if score < best_score:

                best_score = score

                agent.save(
                    MODEL_PATH
                )

                print(
                    ">>> New best model saved."
                )

            print()

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False
    )

    # Save final model too
    final_model_path = (
        RESULTS_DIR
        / "final_dqn_model.pth"
    )

    agent.save(
        final_model_path
    )

    print("=" * 70)
    print("DQN TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best model: {MODEL_PATH}"
    )

    print(
        f"Final model: {final_model_path}"
    )

    print(
        f"Training history: {HISTORY_PATH}"
    )


if __name__ == "__main__":
    main()