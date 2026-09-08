class EV:
    """
    Individual EV charging model.
    """

    def __init__(
        self,
        capacity_kwh=50.0,
        initial_soc=30.0,
        target_soc=80.0,
        max_charge_kw=7.0,
        efficiency=0.95,
    ):
        self.capacity_kwh = capacity_kwh
        self.initial_soc = initial_soc
        self.target_soc = target_soc
        self.max_charge_kw = max_charge_kw
        self.efficiency = efficiency

        self.soc = initial_soc

    def reset(self):
        self.soc = self.initial_soc

    def charge(self, enabled, dt_hours=1.0):

        if not enabled:
            return 0.0

        if self.soc >= self.target_soc:
            return 0.0

        available_energy_kwh = (
            (self.target_soc - self.soc)
            / 100.0
            * self.capacity_kwh
        )

        max_input_power = (
            available_energy_kwh
            / self.efficiency
            / dt_hours
        )

        actual_power = min(
            self.max_charge_kw,
            max_input_power
        )

        stored_energy = (
            actual_power
            * dt_hours
            * self.efficiency
        )

        self.soc += (
            stored_energy
            / self.capacity_kwh
            * 100.0
        )

        self.soc = min(
            self.target_soc,
            self.soc
        )

        return actual_power

    @property
    def deficit(self):
        return max(
            0.0,
            self.target_soc - self.soc
        )