from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, Any

LOG_DIR = Path("data_lake/logs")


def append_log(event_type: str, payload: Dict[str, Any]):
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow().isoformat()
    date_str = now[:10].replace("-", "")

    record = {
        "timestamp": now,
        "event_type": event_type,
        **payload,
    }

    df = pd.DataFrame([record])
    fp = LOG_DIR / f"log_{event_type}_{date_str}.parquet"

    if fp.exists():
        old = pd.read_parquet(fp)
        pd.concat([old, df], ignore_index=True).to_parquet(fp, index=False)
    else:
        df.to_parquet(fp, index=False)
