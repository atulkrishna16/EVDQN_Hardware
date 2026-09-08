import pandas as pd


class LoadModel:
    """
    Real electricity load model based on processed ETT data.
    """

    def __init__(self, csv_path):
        self.df = pd.read_csv(
            csv_path,
            parse_dates=["date"]
        )

        if "real_load_kW" not in self.df.columns:
            raise ValueError(
                "real_load_kW column not found."
            )

        self.loads = (
            self.df["real_load_kW"]
            .astype(float)
            .values
        )

        self.hours = (
            self.df["hour"]
            .astype(int)
            .values
        )

        self.weekends = (
            self.df["is_weekend"]
            .astype(int)
            .values
        )

    def __len__(self):
        return len(self.df)

    def get(self, index):
        return {
            "load_kw": float(
                self.loads[index]
            ),
            "hour": int(
                self.hours[index]
            ),
            "is_weekend": int(
                self.weekends[index]
            ),
        }