class Battery:
    """
    Battery energy storage model.

    Positive power request  -> charging
    Negative power request  -> discharging

    Power is measured in kW.
    Energy capacity is measured in kWh.
    """

    def __init__(
        self,
        capacity_kwh=100.0,
        initial_soc=50.0,
        min_soc=20.0,
        max_soc=95.0,
        max_charge_kw=25.0,
        max_discharge_kw=25.0,
        efficiency=0.95,
    ):
        self.capacity_kwh = capacity_kwh
        self.initial_soc = initial_soc
        self.min_soc = min_soc
        self.max_soc = max_soc
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw
        self.efficiency = efficiency

        self.soc = initial_soc

    def reset(self):
        self.soc = self.initial_soc

    def step(self, requested_power_kw, dt_hours=1.0):
        """
        Execute one battery action.

        Returns:
            actual_charge_kw
            actual_discharge_kw
        """

        requested_power_kw = float(requested_power_kw)

        actual_charge_kw = 0.0
        actual_discharge_kw = 0.0

        # ----------------------------------------------------
        # Charging
        # ----------------------------------------------------
        if requested_power_kw > 0:

            requested_power_kw = min(
                requested_power_kw,
                self.max_charge_kw
            )

            available_energy_kwh = (
                (self.max_soc - self.soc)
                / 100.0
                * self.capacity_kwh
            )

            # Charging energy stored after efficiency
            max_input_energy = (
                available_energy_kwh
                / self.efficiency
            )

            max_input_power = (
                max_input_energy / dt_hours
            )

            actual_charge_kw = min(
                requested_power_kw,
                max_input_power
            )

            stored_energy = (
                actual_charge_kw
                * dt_hours
                * self.efficiency
            )

            self.soc += (
                stored_energy
                / self.capacity_kwh
                * 100.0
            )

        # ----------------------------------------------------
        # Discharging
        # ----------------------------------------------------
        elif requested_power_kw < 0:

            requested_discharge_kw = min(
                abs(requested_power_kw),
                self.max_discharge_kw
            )

            available_energy_kwh = (
                (self.soc - self.min_soc)
                / 100.0
                * self.capacity_kwh
            )

            max_output_energy = (
                available_energy_kwh
                * self.efficiency
            )

            max_output_power = (
                max_output_energy / dt_hours
            )

            actual_discharge_kw = min(
                requested_discharge_kw,
                max_output_power
            )

            battery_energy_used = (
                actual_discharge_kw
                * dt_hours
                / self.efficiency
            )

            self.soc -= (
                battery_energy_used
                / self.capacity_kwh
                * 100.0
            )

        self.soc = max(
            self.min_soc,
            min(self.max_soc, self.soc)
        )

        return (
            actual_charge_kw,
            actual_discharge_kw
        )