"""
Time-of-Use Electricity Tariff Model
=====================================
Defines three tariff periods for grid energy cost calculations.

Period      Hours           Rate (₹/kWh)
----------  --------------  ------------
Off-peak    22:00 – 06:00   4.00
Shoulder    06:00 – 09:00   6.50
            18:00 – 22:00   6.50
Peak        09:00 – 18:00   9.00
"""


class TariffModel:
    """
    Simple time-of-use tariff.

    Parameters
    ----------
    peak_rate      : ₹/kWh  during peak hours   (09–18)
    shoulder_rate  : ₹/kWh  during shoulder     (06–09, 18–22)
    offpeak_rate   : ₹/kWh  during off-peak     (22–06)
    """

    PEAK_HOURS      = set(range(9, 18))        # 09 ≤ h < 18
    SHOULDER_HOURS  = set(range(6, 9)) | set(range(18, 22))
    # everything else is off-peak

    def __init__(
        self,
        peak_rate:     float = 9.00,
        shoulder_rate: float = 6.50,
        offpeak_rate:  float = 4.00,
    ):
        self.peak_rate     = peak_rate
        self.shoulder_rate = shoulder_rate
        self.offpeak_rate  = offpeak_rate

    # ------------------------------------------------------------------
    def get_rate(self, hour: int) -> float:
        """Return the tariff rate (₹/kWh) for a given hour (0–23)."""
        hour = hour % 24
        if hour in self.PEAK_HOURS:
            return self.peak_rate
        if hour in self.SHOULDER_HOURS:
            return self.shoulder_rate
        return self.offpeak_rate

    # ------------------------------------------------------------------
    def calculate_cost(
        self,
        power_kw: float,
        hour:     int,
        dt:       float = 1.0,
    ) -> float:
        """
        Cost = power (kW) × duration (h) × rate (₹/kWh)

        Parameters
        ----------
        power_kw : grid import power in kW  (≥ 0)
        hour     : current hour (0–23)
        dt       : timestep duration in hours (default 1 h)

        Returns
        -------
        cost in ₹
        """
        return max(power_kw, 0.0) * dt * self.get_rate(hour)

    # ------------------------------------------------------------------
    def rate_profile(self) -> list:
        """Return the full 24-hour rate profile (useful for plotting)."""
        return [self.get_rate(h) for h in range(24)]
