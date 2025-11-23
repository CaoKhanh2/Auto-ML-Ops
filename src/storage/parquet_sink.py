from pathlib import Path
import pandas as pd

DL = Path("data_lake/predictions")


def append_prediction(df: pd.DataFrame):
    DL.mkdir(parents=True, exist_ok=True)
    date = pd.to_datetime(df["timestamp"].iloc[0]).strftime("%Y%m%d")
    fp = DL / f"pred_{date}.parquet"

    if fp.exists():
        old = pd.read_parquet(fp)
        new = pd.concat([old, df], ignore_index=True)
        new.to_parquet(fp, index=False)
    else:
        df.to_parquet(fp, index=False)
