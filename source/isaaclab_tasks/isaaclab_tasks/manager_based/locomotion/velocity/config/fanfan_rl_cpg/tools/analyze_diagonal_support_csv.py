#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LEG_ORDER = ("FR", "FL", "RR", "RL")


def f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def expected_pair_from_phase(rows: list[dict]) -> str:
    phases = {}
    for row in rows:
        leg = row.get("leg_name", "")
        if leg in LEG_ORDER:
            phases[leg] = f(row, "cpg_leg_phase")
    if len(phases) < 4:
        return rows[0].get("expected_support_pair", "unknown") or "unknown"
    swing_fraction = 0.40
    stance = {leg: (phases[leg] % 1.0) >= swing_fraction for leg in LEG_ORDER}
    fr_rl = float(stance["FR"]) + float(stance["RL"])
    fl_rr = float(stance["FL"]) + float(stance["RR"])
    return "FR_RL" if fr_rl >= fl_rr else "FL_RR"


def analyze(path: Path) -> int:
    groups: dict[str, list[dict]] = defaultdict(list)
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            groups[row.get("time", "")].append(row)

    if not groups:
        print("[FAIL] Empty CSV.")
        return 2

    counts = defaultdict(int)
    wins = defaultdict(int)
    gate_clamps = []
    raw_error_max = []
    qcmd_error_max = []
    final_limited = []

    for rows in groups.values():
        first = rows[0]
        expected = first.get("expected_support_pair", "") or expected_pair_from_phase(rows)
        if expected not in ("FR_RL", "FL_RR"):
            expected = expected_pair_from_phase(rows)

        if "diag_FR_RL_torque_abs_sum" in first and first.get("diag_FR_RL_torque_abs_sum", "") != "":
            diag_fr_rl = f(first, "diag_FR_RL_torque_abs_sum")
            diag_fl_rr = f(first, "diag_FL_RR_torque_abs_sum")
        else:
            leg_sum = defaultdict(float)
            for row in rows:
                leg = row.get("leg_name", "")
                if leg in LEG_ORDER:
                    leg_sum[leg] += abs(f(row, "torque_measured"))
            diag_fr_rl = leg_sum["FR"] + leg_sum["RL"]
            diag_fl_rr = leg_sum["FL"] + leg_sum["RR"]

        winner = "FR_RL" if diag_fr_rl >= diag_fl_rr else "FL_RR"
        counts[expected] += 1
        wins[(expected, winner)] += 1
        gate_clamps.append(f(first, "hip_gate_clamp_count"))
        raw_error_max.append(f(first, "q_raw_error_abs_max"))
        qcmd_error_max.append(f(first, "q_cmd_error_abs_max"))
        final_limited.append(f(first, "final_limited_count"))

    total = sum(counts.values())
    print(f"CSV: {path}")
    print(f"steps: {total}")
    for pair in ("FR_RL", "FL_RR"):
        n = counts[pair]
        ok = wins[(pair, pair)]
        acc = ok / n if n else 0.0
        other = "FL_RR" if pair == "FR_RL" else "FR_RL"
        print(f"{pair}: expected_steps={n} winner_ok={ok} accuracy={acc:.3f} other_wins={wins[(pair, other)]}")

    def mean(values: list[float]) -> float:
        return sum(values) / max(len(values), 1)

    print(f"hip_gate_clamp_count_mean={mean(gate_clamps):.3f}")
    print(f"q_raw_error_abs_max_mean={mean(raw_error_max):.3f}")
    print(f"q_cmd_error_abs_max_mean={mean(qcmd_error_max):.3f}")
    print(f"final_limited_count_mean={mean(final_limited):.3f}")

    fl_rr_acc = wins[("FL_RR", "FL_RR")] / counts["FL_RR"] if counts["FL_RR"] else 0.0
    if counts["FL_RR"] and fl_rr_acc < 0.45:
        print("[WARN] FL+RR support phase is not taking over torque proxy reliably. Do not continue hardware test.")
        return 1
    print("[OK] No strong one-sided diagonal support failure found in this CSV.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze diagonal support proxy from policy debug CSV.")
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    return analyze(args.csv_path)


if __name__ == "__main__":
    raise SystemExit(main())
