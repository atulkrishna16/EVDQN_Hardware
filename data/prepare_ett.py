"""
Phase 1 - ETT Dataset Preparation

Input:
    data/raw/ETTh1.csv

Output:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv

The ETT dataset provides real-world electricity load measurements.
We construct a composite load signal from the six electricity-related
features and calibrate it to the kW scale used by the smart-grid simulator.

Chronological split:
    60% training
    20% validation
    20% testing
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ETTh1.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_RATIO = 0.60
VALIDATION_RATIO = 0.20
TEST_RATIO = 0.20

# ------------------------------------------------------------
# Smart-grid calibration
# ------------------------------------------------------------
# The ETT load features are not directly the same as the kW
# scale of our simulated smart-grid environment.
#
# We therefore map the composite ETT load profile into a
# realistic operating range for our simulated grid.
#
# IMPORTANT:
# This does NOT change the temporal pattern of the real data.
# It only establishes the project's simulation capacity scale.
# ------------------------------------------------------------

TARGET_LOAD_MIN_KW = 35.0
TARGET_LOAD_MAX_KW = 95.0

# Quantiles are used instead of raw min/max so extreme ETT
# observations don't dominate the calibration.
CALIBRATION_LOW_QUANTILE = 0.05
CALIBRATION_HIGH_QUANTILE = 0.95


# Electricity-related columns in ETTh1
LOAD_COLUMNS = [
    "HUFL",
    "HULL",
    "MUFL",
    "MULL",
    "LUFL",
    "LULL",
]

REQUIRED_COLUMNS = ["date"] + LOAD_COLUMNS + ["OT"]


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def check_input_file():
    """Verify that the input dataset exists."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nETT dataset not found:\n{INPUT_FILE}\n\n"
            "Make sure ETTh1.csv is inside:\n"
            "data/raw/ETTh1.csv"
        )

    print("=" * 70)
    print("INPUT FILE")
    print("=" * 70)
    print(INPUT_FILE)


def load_dataset():
    """Load and validate the ETT dataset."""

    print("\n" + "=" * 70)
    print("LOADING ETTH1 DATASET")
    print("=" * 70)

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    return df


def clean_dataset(df):
    """Clean dates, numeric values and duplicate rows."""

    print("\n" + "=" * 70)
    print("CLEANING DATA")
    print("=" * 70)

    df = df.copy()

    # --------------------------------------------------------
    # Parse date
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    invalid_dates = df["date"].isna().sum()

    if invalid_dates > 0:
        print(f"Removing {invalid_dates} invalid dates.")
        df = df.dropna(subset=["date"])

    # --------------------------------------------------------
    # Convert load columns to numeric
    # --------------------------------------------------------

    numeric_columns = LOAD_COLUMNS + ["OT"]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove rows with missing numeric values
    # --------------------------------------------------------

    missing_before = df[numeric_columns].isna().sum().sum()

    if missing_before > 0:
        print(
            f"Removing {missing_before} missing numeric values."
        )

        df = df.dropna(
            subset=numeric_columns
        )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    duplicate_count = df["date"].duplicated().sum()

    if duplicate_count > 0:
        print(
            f"Removing {duplicate_count} duplicate timestamps."
        )

        df = df.drop_duplicates(
            subset=["date"],
            keep="first"
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values("date").reset_index(drop=True)

    print(f"Rows after cleaning: {len(df):,}")

    print(
        f"Date range: "
        f"{df['date'].min()} → {df['date'].max()}"
    )

    return df


def create_composite_load(df):
    """
    Create a composite electricity load signal.

    ETTh1 contains six electricity-related variables:
        HUFL, HULL, MUFL, MULL, LUFL, LULL

    We sum these variables to preserve the combined temporal
    behavior of the real-world load measurements.

    Because ETT variables can contain negative values and are
    not directly calibrated to our project's kW scale, the
    composite signal is later calibrated using training data.
    """

    print("\n" + "=" * 70)
    print("CREATING COMPOSITE LOAD")
    print("=" * 70)

    df = df.copy()

    df["raw_composite_load"] = df[LOAD_COLUMNS].sum(axis=1)

    print(
        "Raw composite load statistics:"
    )

    print(
        df["raw_composite_load"].describe()
    )

    return df


def chronological_split(df):
    """
    Split the dataset chronologically.

    60% train
    20% validation
    20% test
    """

    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    n = len(df)

    train_end = int(n * TRAIN_RATIO)
    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    print(
        f"Train      : {len(train_df):,} rows "
        f"({len(train_df) / n:.1%})"
    )

    print(
        f"Validation : {len(validation_df):,} rows "
        f"({len(validation_df) / n:.1%})"
    )

    print(
        f"Test       : {len(test_df):,} rows "
        f"({len(test_df) / n:.1%})"
    )

    print("\nDate ranges:")

    print(
        f"Train      : "
        f"{train_df['date'].min()} → "
        f"{train_df['date'].max()}"
    )

    print(
        f"Validation : "
        f"{validation_df['date'].min()} → "
        f"{validation_df['date'].max()}"
    )

    print(
        f"Test       : "
        f"{test_df['date'].min()} → "
        f"{test_df['date'].max()}"
    )

    return train_df, validation_df, test_df


def calibrate_load(train_df, validation_df, test_df):
    """
    Calibrate the composite ETT load into the project's kW scale.

    IMPORTANT:
    Calibration parameters are calculated ONLY from the training
    data to prevent test-data leakage.

    Training composite-load 5th percentile
        -> TARGET_LOAD_MIN_KW

    Training composite-load 95th percentile
        -> TARGET_LOAD_MAX_KW

    Validation and test are transformed using exactly the same
    training-derived calibration.
    """

    print("\n" + "=" * 70)
    print("CALIBRATING LOAD TO SMART-GRID kW SCALE")
    print("=" * 70)

    train = train_df.copy()
    validation = validation_df.copy()
    test = test_df.copy()

    low_value = train["raw_composite_load"].quantile(
        CALIBRATION_LOW_QUANTILE
    )

    high_value = train["raw_composite_load"].quantile(
        CALIBRATION_HIGH_QUANTILE
    )

    print(
        f"Training {CALIBRATION_LOW_QUANTILE:.0%} percentile: "
        f"{low_value:.4f}"
    )

    print(
        f"Training {CALIBRATION_HIGH_QUANTILE:.0%} percentile: "
        f"{high_value:.4f}"
    )

    if high_value <= low_value:
        raise ValueError(
            "Invalid calibration range."
        )

    def transform(series):
        # Normalize according to training distribution
        normalized = (
            (series - low_value)
            / (high_value - low_value)
        )

        # Convert to project kW scale
        load_kw = (
            TARGET_LOAD_MIN_KW
            + normalized
            * (
                TARGET_LOAD_MAX_KW
                - TARGET_LOAD_MIN_KW
            )
        )

        # Prevent unrealistic negative or excessive values
        load_kw = load_kw.clip(
            lower=TARGET_LOAD_MIN_KW,
            upper=TARGET_LOAD_MAX_KW
        )

        return load_kw

    train["real_load_kW"] = transform(
        train["raw_composite_load"]
    )

    validation["real_load_kW"] = transform(
        validation["raw_composite_load"]
    )

    test["real_load_kW"] = transform(
        test["raw_composite_load"]
    )

    print("\nCalibrated load statistics:")

    print(
        "\nTRAIN:"
    )

    print(
        train["real_load_kW"].describe()
    )

    print(
        "\nVALIDATION:"
    )

    print(
        validation["real_load_kW"].describe()
    )

    print(
        "\nTEST:"
    )

    print(
        test["real_load_kW"].describe()
    )

    return train, validation, test


def add_time_features(df):
    """Add useful temporal features."""

    df = df.copy()

    # Hour of day
    df["hour"] = df["date"].dt.hour

    # Day of week
    df["day_of_week"] = df["date"].dt.dayofweek

    # Weekend indicator
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    # Cyclic encoding of hour
    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Cyclic encoding of day of week
    df["day_sin"] = np.sin(
        2 * np.pi * df["day_of_week"] / 7
    )

    df["day_cos"] = np.cos(
        2 * np.pi * df["day_of_week"] / 7
    )

    return df


def select_output_columns(df):
    """
    Select columns that the smart-grid environment will use.

    Original ETT measurements are retained for traceability.
    """

    columns = [
        "date",

        # Original ETT load variables
        "HUFL",
        "HULL",
        "MUFL",
        "MULL",
        "LUFL",
        "LULL",

        # Transformer oil temperature
        "OT",

        # Composite real-world signal
        "raw_composite_load",

        # Calibrated load used by environment
        "real_load_kW",

        # Time information
        "hour",
        "day_of_week",
        "is_weekend",
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ]

    return df[columns].copy()


def save_dataset(df, filename):
    """Save processed dataset."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = OUTPUT_DIR / filename

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )

    return output_path


def verify_outputs(train, validation, test):
    """Perform final checks."""

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    datasets = {
        "TRAIN": train,
        "VALIDATION": validation,
        "TEST": test,
    }

    for name, dataset in datasets.items():

        print(f"\n{name}")

        print(
            f"Rows: {len(dataset):,}"
        )

        print(
            f"Missing values: "
            f"{dataset.isna().sum().sum()}"
        )

        print(
            f"Load range: "
            f"{dataset['real_load_kW'].min():.2f} - "
            f"{dataset['real_load_kW'].max():.2f} kW"
        )

        print(
            f"Average load: "
            f"{dataset['real_load_kW'].mean():.2f} kW"
        )

    # --------------------------------------------------------
    # Chronological validation
    # --------------------------------------------------------

    if not (
        train["date"].max()
        < validation["date"].min()
    ):
        raise ValueError(
            "Train and validation dates overlap."
        )

    if not (
        validation["date"].max()
        < test["date"].min()
    ):
        raise ValueError(
            "Validation and test dates overlap."
        )

    print(
        "\nChronological split verified successfully."
    )

    # --------------------------------------------------------
    # Check duplicate timestamps
    # --------------------------------------------------------

    for name, dataset in datasets.items():

        duplicate_dates = dataset["date"].duplicated().sum()

        if duplicate_dates > 0:
            raise ValueError(
                f"{name} contains duplicate timestamps."
            )

    print(
        "No duplicate timestamps found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("SMART ENERGY GRID - PHASE 1")
    print("ETT DATA PREPARATION")
    print("=" * 70)

    # 1. Check input
    check_input_file()

    # 2. Load
    df = load_dataset()

    # 3. Clean
    df = clean_dataset(df)

    # 4. Create composite load
    df = create_composite_load(df)

    # 5. Chronological split
    train_df, validation_df, test_df = chronological_split(
        df
    )

    # 6. Calibrate using TRAIN ONLY
    train_df, validation_df, test_df = calibrate_load(
        train_df,
        validation_df,
        test_df
    )

    # 7. Add temporal features
    train_df = add_time_features(
        train_df
    )

    validation_df = add_time_features(
        validation_df
    )

    test_df = add_time_features(
        test_df
    )

    # 8. Select final columns
    train_df = select_output_columns(
        train_df
    )

    validation_df = select_output_columns(
        validation_df
    )

    test_df = select_output_columns(
        test_df
    )

    # 9. Save
    save_dataset(
        train_df,
        "train.csv"
    )

    save_dataset(
        validation_df,
        "validation.csv"
    )

    save_dataset(
        test_df,
        "test.csv"
    )

    # 10. Verify
    verify_outputs(
        train_df,
        validation_df,
        test_df
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        "  data/processed/train.csv"
    )

    print(
        "  data/processed/validation.csv"
    )

    print(
        "  data/processed/test.csv"
    )

    print(
        "\nThese files are ready for Phase 2:"
    )

    print(
        "ETT real load → SmartGridEnv"
    )

    print("\n")


if __name__ == "__main__":
    main()