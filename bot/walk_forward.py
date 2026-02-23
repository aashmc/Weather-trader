"""
Walk-forward quality validation utility (Item #4).

Uses model_quality_state.json resolved metrics and reports rolling
train/test performance by city.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

STATE_FILE = Path("model_quality_state.json")


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text())


def _metric_mean(rows: list[dict], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r.get(key, 0.0)) for r in rows) / len(rows)


def _parse_ts(row: dict) -> datetime:
    ts = row.get("timestamp", "")
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def run_walk_forward(city: str, records: list[dict], train: int, test: int):
    rows = sorted(records, key=_parse_ts)
    n = len(rows)
    print(f"\n=== {city} ({n} resolved samples) ===")

    if n < train + test:
        print(f"Not enough samples for train={train} test={test}")
        return

    fold = 0
    start = 0
    while start + train + test <= n:
        fold += 1
        train_rows = rows[start : start + train]
        test_rows = rows[start + train : start + train + test]

        tr_brier = _metric_mean(train_rows, "brier")
        te_brier = _metric_mean(test_rows, "brier")
        tr_wp = _metric_mean(train_rows, "winner_prob")
        te_wp = _metric_mean(test_rows, "winner_prob")
        tr_ll = _metric_mean(train_rows, "logloss")
        te_ll = _metric_mean(test_rows, "logloss")

        print(
            f"Fold {fold:02d} | "
            f"Train Brier {tr_brier:.3f}, WP {tr_wp:.3f}, LL {tr_ll:.3f} | "
            f"Test Brier {te_brier:.3f}, WP {te_wp:.3f}, LL {te_ll:.3f} | "
            f"Drift ΔBrier {te_brier - tr_brier:+.3f}"
        )
        start += test


def main():
    ap = argparse.ArgumentParser(description="Walk-forward quality validation")
    ap.add_argument("--city", default="", help="City name filter (e.g., London)")
    ap.add_argument("--train", type=int, default=30, help="Train window size")
    ap.add_argument("--test", type=int, default=10, help="Test window size")
    args = ap.parse_args()

    state = _load_state()
    metrics = state.get("resolved_metrics", {})
    if not metrics:
        print("No resolved quality metrics found in model_quality_state.json")
        return

    for city_name, rows in metrics.items():
        if args.city and args.city.lower() not in city_name.lower():
            continue
        run_walk_forward(city_name, rows, args.train, args.test)


if __name__ == "__main__":
    main()
