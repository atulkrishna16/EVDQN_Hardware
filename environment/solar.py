import math


class SolarModel:
    """
    Simple deterministic solar generation model.

    Peak generation occurs around noon.
    Generation is zero during nighttime.
    """

    def __init__(
        self,
        peak_kw=60.0,
        sunrise=6,
        sunset=18,
    ):

        self.peak_kw = float(peak_kw)
        self.sunrise = float(sunrise)
        self.sunset = float(sunset)

    def get_generation(self, hour):

        hour = float(hour)

        # Night
        if (
            hour < self.sunrise
            or hour >= self.sunset
        ):
            return 0.0

        solar_angle = math.pi * (
            hour - self.sunrise
        ) / (
            self.sunset - self.sunrise
        )

        generation = (
            self.peak_kw
            * math.sin(solar_angle)
        )

        return max(
            0.0,
            generation
        )