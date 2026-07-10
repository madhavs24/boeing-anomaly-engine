"""Daily refresh job: fetch fresh BA data, retrain the cached models, rebuild the dashboard.
Run by the Render cron job (see render.yaml) or locally / via Task Scheduler.

    python -m src.refresh
"""
from __future__ import annotations
from . import data, cache, dashboard
from .features import build_features


def main():
    panel = data.get_panel("live")                 # network needed (host/your machine)
    print(f"refreshed panel: {panel.shape}, through {panel.index[-1].date()}")
    cache.train_and_cache(build_features(panel))    # retrain on the fresh data
    dashboard.build(mode="cached")                     # regenerate the static snapshot
    print("refresh complete: models + dashboard updated")


if __name__ == "__main__":
    main()
