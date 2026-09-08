import pandas as pd


def get_uncontrolled_action(env):

    """
    Battery idle.
    All EVs charge whenever they need energy.
    """

    battery_index = 2  # battery idle

    ev_bits = 0

    for i, ev in enumerate(env.evs):

        if ev.soc < ev.target_soc:

            ev_bits |= (
                1 << i
            )

    return (
        battery_index * 8
        + ev_bits
    )


def run_uncontrolled(
    env,
    output_path,
    max_episodes=None,
):

    records = []

    total_rows = (
        len(env.df)
    )

    starts = list(
        range(
            0,
            total_rows
            - env.episode_length
            + 1,
            env.episode_length
        )
    )

    if max_episodes is not None:

        starts = starts[
            :max_episodes
        ]

    for episode, start in enumerate(
        starts
    ):

        state = env.reset(
            start_index=start
        )

        done = False

        while not done:

            action = (
                get_uncontrolled_action(
                    env
                )
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

            row["episode"] = episode

            row["controller"] = (
                "UNCONTROLLED"
            )

            records.append(row)

    result = pd.DataFrame(
        records
    )

    result.to_csv(
        output_path,
        index=False
    )

    return result