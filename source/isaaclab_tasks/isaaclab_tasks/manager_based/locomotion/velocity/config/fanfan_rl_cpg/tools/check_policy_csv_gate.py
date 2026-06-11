#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict


def _quantile(values: list[float], p: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    k = (len(values) - 1) * p
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - k) + values[hi] * (k - lo)


def _float(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _cycle_stats(rows: list[dict]) -> list[dict[str, float]]:
    by_time: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_time[row["time"]].append(row)

    cycles = []
    for items in by_time.values():
        cycles.append(
            {
                "cmd_x": _float(items[0], "cmd_x"),
                "raw_abs_max": max(abs(_float(x, "action_raw_policy")) for x in items),
                "rate_limited_count": sum(int(_float(x, "rate_limited_joint_mask")) for x in items),
                "final_limited_count": sum(int(_float(x, "final_limited_joint_mask")) for x in items),
                "tau_est_max": max(abs(_float(x, "tau_est_real")) for x in items),
            }
        )
    return cycles


def _summarize(label: str, cycles: list[dict[str, float]]) -> dict[str, float]:
    raw = [x["raw_abs_max"] for x in cycles]
    rate = [x["rate_limited_count"] for x in cycles]
    final = [x["final_limited_count"] for x in cycles]
    tau = [x["tau_est_max"] for x in cycles]
    if not raw:
        print(f"{label}: no cycles")
        return {}

    out = {
        "raw_mean": sum(raw) / len(raw),
        "raw_p90": _quantile(raw, 0.90),
        "raw_max": max(raw),
        "raw_gt2_pct": 100.0 * sum(v > 2.0 for v in raw) / len(raw),
        "raw_gt3_pct": 100.0 * sum(v > 3.0 for v in raw) / len(raw),
        "raw_gt5_pct": 100.0 * sum(v > 5.0 for v in raw) / len(raw),
        "rate_mean": sum(rate) / len(rate),
        "final_mean": sum(final) / len(final),
        "tau_mean": sum(tau) / len(tau),
        "tau_p90": _quantile(tau, 0.90),
        "tau_max": max(tau),
    }
    print(
        f"{label}: cycles={len(cycles)} "
        f"raw_mean={out['raw_mean']:.2f} raw_p90={out['raw_p90']:.2f} raw_max={out['raw_max']:.2f} "
        f"raw>2={out['raw_gt2_pct']:.1f}% raw>3={out['raw_gt3_pct']:.1f}% raw>5={out['raw_gt5_pct']:.1f}% "
        f"rate_mean={out['rate_mean']:.2f}/12 final_mean={out['final_mean']:.2f}/12 "
        f"tau_mean={out['tau_mean']:.2f} tau_p90={out['tau_p90']:.2f} tau_max={out['tau_max']:.2f}"
    )
    return out


def _verdict(stats: dict[str, float], moving: bool) -> list[str]:
    warnings = []
    if not stats:
        return warnings

    if moving:
        if stats["raw_gt5_pct"] > 5.0 or stats["raw_p90"] > 3.0:
            warnings.append("moving raw action is too large")
        if stats["rate_mean"] > 6.0:
            warnings.append("rate limiter is active too often")
        if stats["final_mean"] > 2.0:
            warnings.append("final torque safety is active too often")
        if stats["tau_p90"] > 5.8:
            warnings.append("tau_est is too close to the 6 Nm continuous budget")
    else:
        if stats["raw_gt3_pct"] > 5.0 or stats["raw_p90"] > 2.0:
            warnings.append("standing raw action is too large")
        if stats["final_mean"] > 2.0:
            warnings.append("standing final safety is active too often")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Gate a real-policy CSV before real walking.")
    parser.add_argument("csv_path")
    parser.add_argument("--skip-sec", type=float, default=1.0)
    parser.add_argument("--moving-cmd-threshold", type=float, default=0.03)
    args = parser.parse_args()

    with open(args.csv_path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row.get("mode", "policy") == "policy"]
    if not rows:
        raise SystemExit("No policy rows found.")

    t0 = min(_float(row, "time") for row in rows)
    rows = [row for row in rows if _float(row, "time") - t0 >= args.skip_sec]
    cycles = _cycle_stats(rows)
    standing = [x for x in cycles if abs(x["cmd_x"]) < args.moving_cmd_threshold]
    moving = [x for x in cycles if abs(x["cmd_x"]) >= args.moving_cmd_threshold]

    standing_stats = _summarize("cmd_x=0", standing)
    moving_stats = _summarize("cmd_x>0", moving)
    warnings = _verdict(standing_stats, moving=False) + _verdict(moving_stats, moving=True)
    if warnings:
        print("DO NOT walk on the real robot yet: raw actions or safety intervention are too high.")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("CSV gate passed: proceed only to cautious suspended/support-frame testing first.")


if __name__ == "__main__":
    main()
