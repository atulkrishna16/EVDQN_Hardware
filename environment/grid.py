class Grid:
    """
    Grid model.

    Grid import above the limit is considered an overload.
    """

    def __init__(
        self,
        limit_kw=100.0,
        base_tariff=6.0,
    ):
        self.limit_kw = limit_kw
        self.base_tariff = base_tariff

    def calculate(
        self,
        load_kw,
        ev_power_kw,
        solar_kw,
        battery_charge_kw,
        battery_discharge_kw,
        tariff_rate,
    ):

        total_demand = (
            load_kw
            + ev_power_kw
        )

        solar_used = min(
            solar_kw,
            total_demand + battery_charge_kw
        )

        solar_curtailed = max(
            0.0,
            solar_kw
            - solar_used
        )

        grid_import = (
            load_kw
            + ev_power_kw
            + battery_charge_kw
            - solar_used
            - battery_discharge_kw
        )

        grid_import = max(
            0.0,
            grid_import
        )

        overload = max(
            0.0,
            grid_import - self.limit_kw
        )

        grid_cost = (
            grid_import
            * tariff_rate
        )

        return {
            "grid_import_kw": grid_import,
            "grid_cost_inr": grid_cost,
            "solar_used_kw": solar_used,
            "solar_curtailed_kw": solar_curtailed,
            "overload_kw": overload,
        }