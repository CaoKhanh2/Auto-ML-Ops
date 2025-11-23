from pathlib import Path
import json
import os
import sys
from typing import Dict, Any, List

# Allow running directly: python src/ingestion/scraper_job.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

PROJECT_ROOT = SRC_DIR.parent

from ingestion import getData
from storage.log_parquet import append_log

CONFIG_PATH = PROJECT_ROOT / "data/scraper_config.json"


def load_scraper_config():
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
    else:
        cfg = {}

    env_map = {
        "PN_START": "start_draw",
        "PN_END": "end_draw",
        "PN_OUTPUT": "output_dir",
    }
    for env_key, cfg_key in env_map.items():
        if os.getenv(env_key):
            cfg[cfg_key] = os.getenv(env_key)

    cfg.setdefault("start_draw", 1)
    cfg.setdefault("end_draw", None)
    cfg.setdefault("output_dir", "data/output")
    cfg.setdefault("output_format", "excel")

    return cfg


def build_argv(cfg):
    argv = []
    if cfg.get("start_draw") is not None:
        argv += ["--start", str(cfg["start_draw"])]
    if cfg.get("end_draw") is not None:
        argv += ["--end", str(cfg["end_draw"])]
    argv += ["--output", cfg["output_dir"]]
    argv += ["--format", cfg["output_format"]]
    return argv


def run_scraper():
    cfg = load_scraper_config()
    argv = build_argv(cfg)

    append_log("scraper_run", {"stage": "start", **cfg})

    try:
        getData.main(argv)
        append_log("scraper_run", {"stage": "success"})
    except Exception as e:
        append_log("scraper_run", {"stage": "error", "error": str(e)})
        raise


if __name__ == "__main__":
    run_scraper()
