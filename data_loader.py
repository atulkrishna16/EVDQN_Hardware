# data_loader.py

import os
import pandas as pd


DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data",
    "energy_data.csv"
)


REQUIRED_COLUMNS = [
    "hour",
    "tariff_rate",
    "solar_kW",
    "house_kW",
    "factory_kW",
    "hospital_kW",
    "ev_base_kW"
]


def load_energy_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing = [
        col
        for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df.sort_values("hour").reset_index(drop=True)

    if len(df) != 24:
        raise ValueError(
            "Dataset must contain exactly 24 hourly records."
        )

    # Base electrical load
    df["base_load_kW"] = (
        df["house_kW"]
        + df["factory_kW"]
        + df["hospital_kW"]
    )

    # Total load including predefined EV demand
    df["total_load_kW"] = (
        df["base_load_kW"]
        + df["ev_base_kW"]
    )

    return df