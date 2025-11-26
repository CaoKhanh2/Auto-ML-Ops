"""Simple offline scraper stub to slice the historical CSV by draw range.

It allows the Airflow DAG and scraper job to keep running even without a
real network scraper by slicing the local historical file and exporting to
CSV/Excel.
"""

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data/multi_hot_matrix.csv"


def load_source(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Source data not found at {csv_path}. Please provide the historical file."
        )

    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(512)

    sep = ";" if ";" in sample else ","

    df = pd.read_csv(csv_path, sep=sep)
    return df


def filter_range(df: pd.DataFrame, start: int, end: Optional[int]) -> pd.DataFrame:
    if "draw_id" in df.columns:
        mask = df["draw_id"].astype(int) >= start
        if end is not None:
            mask &= df["draw_id"].astype(int) <= end
        return df.loc[mask].reset_index(drop=True)

    # Fallback: use positional index if draw_id is missing
    end_idx = end if end is not None else len(df)
    return df.iloc[start - 1 : end_idx]


def save_output(df: pd.DataFrame, output_dir: Path, fmt: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lower()

    if fmt == "excel":
        try:
            # Lazy import to avoid hard dependency when not needed
            import openpyxl  # noqa: F401

            output_path = output_dir / "results.xlsx"
            df.to_excel(output_path, index=False)
        except ModuleNotFoundError:
            # Fallback to CSV if openpyxl is unavailable
            output_path = output_dir / "results.csv"
            df.to_csv(output_path, index=False)
    else:
        output_path = output_dir / "results.csv"
        df.to_csv(output_path, index=False)

    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline scraper stub for Mega 6/45")
    parser.add_argument("--start", type=int, default=1, help="Start draw id (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End draw id (inclusive)")
    parser.add_argument("--output", type=str, default="data/output", help="Output directory")
    parser.add_argument("--format", type=str, default="csv", choices=["excel", "csv"], help="Output format")
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE), help="Path to multi_hot_matrix.csv")

    args = parser.parse_args(argv)

    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path

    df = load_source(source_path)
    filtered = filter_range(df, args.start, args.end)

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_path = save_output(filtered, output_dir, args.format)
    print(f"Saved {len(filtered)} rows to {output_path}")


if __name__ == "__main__":
    main()
