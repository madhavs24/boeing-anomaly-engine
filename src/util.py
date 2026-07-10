"""Shared paths, config loader, logging."""
from __future__ import annotations
from pathlib import Path
import sys, datetime as _dt
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROC = DATA / "processed"
RESULTS = ROOT / "results"
for _d in (PROC, RESULTS):
    _d.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r") as fh:
        return yaml.safe_load(fh)


def log(msg: str) -> None:
    ts = _dt.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", file=sys.stderr)
