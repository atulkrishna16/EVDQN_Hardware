import pandas as pd


def get_rule_based_action(env):

    row = env.df.iloc[
        env.current_index
    ]

    hour = int(
        row["hour"]
    )

    load_kw = float(
        row["real_load_kW"]
    )

    solar_kw = env.solar.get_generation(
        hour
    )

    tariff = env.get_tariff(
        hour
    )

    # --------------------------------------------------------
    # Estimate EV charging
    # --------------------------------------------------------

    ev_bits = 0

    # Low-load / solar / cheap periods
    favorable_time = (
        hour >= 6
        and hour < 18
    )

    cheap_time = tariff <= 6.0

    # During grid peak, avoid charging unless EV is near
    # its deadline (we use the end of the 24h episode).
    near_end = (
        env.steps
        >= env.episode_length - 4
    )

    for i, ev in enumerate(env.evs):

        if ev.soc >= ev.target_soc:
            continue

        if favorable_time or cheap_time or near_end:

            # Avoid charging when grid is already heavily
            # loaded, unless the EV is near completion.
            projected_load = (
                load_kw
                + 7.0
            )

            if (
                projected_load
                <= 95.0
                or near_end
            ):

                ev_bits |= (
                    1 << i
                )

    # --------------------------------------------------------
    # Battery control
    # --------------------------------------------------------

    projected_ev_power = 0.0

    for i, ev in enumerate(env.evs):

        if ev_bits & (1 << i):
            projected_ev_power += 7.0

    projected_grid = (
        load_kw
        + projected_ev_power
        - solar_kw
    )

    # Strong discharge when grid is approaching overload
    if projected_grid > 95.0:

        battery_index = 0

    # Moderate discharge during evening peak
    elif (
        tariff >= 9.0
        and projected_grid > 75.0
    ):

        battery_index = 1

    # Charge using available solar
    elif (
        solar_kw
        > load_kw
        + projected_ev_power
        + 10.0
    ):

        battery_index = 4

    # Moderate solar charging
    elif (
        solar_kw
        > load_kw
        + projected_ev_power
    ):

        battery_index = 3

    else:

        battery_index = 2

    return (
        battery_index * 8
        + ev_bits
    )


def run_rule_based(
    env,
    output_path,
    max_episodes=None,
):

    records = []

    total_rows = len(
        env.df
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
                get_rule_based_action(
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
                "RULE_BASED"
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