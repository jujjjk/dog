#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cpg_tool_common import JOINT_ORDER, csv_rows, package_dir, quantile, write_text, write_yaml


def _f(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row and row[key] != "":
            try:
                return float(row[key])
            except ValueError:
                return None
    return None


def analyze(csv_path: Path, skip_sec: float = 0.0) -> dict:
    rows = csv_rows(csv_path)
    if rows and skip_sec > 0 and "time" in rows[0]:
        t0 = _f(rows[0], "time") or 0.0
        rows = [r for r in rows if (_f(r, "time") or t0) - t0 >= skip_sec]
    raw_abs = []
    clip_counts = []
    by_joint = defaultdict(list)
    residual_by_leg = defaultdict(list)
    torque = []
    qerr = []
    qdelta = []
    for row in rows:
        joint = row.get("policy_joint_name") or row.get("joint_name")
        raw = _f(row, "action_raw_policy", "action_raw", "raw_action")
        if raw is not None:
            raw_abs.append(abs(raw))
            if joint in JOINT_ORDER:
                by_joint[joint].append(abs(raw))
        leg = row.get("leg_name") or (joint[:2] if joint else "")
        residual = _f(row, "delta_q_rl", "action_scaled_policy", "q_raw_target_minus_default_policy")
        if residual is not None and leg:
            residual_by_leg[leg].append(abs(residual))
        clip = _f(row, "clip_count", "final_limited_count")
        if clip is not None:
            clip_counts.append(clip)
        tq = _f(row, "tau_est_real", "torque_measured", "torque")
        if tq is not None:
            torque.append(abs(tq))
        qe = _f(row, "q_error_real", "q_error")
        if qe is not None:
            qerr.append(abs(qe))
        qd = _f(row, "smooth_step_real", "raw_to_final_delta_real", "q_cmd_delta")
        if qd is not None:
            qdelta.append(abs(qd))

    leg_sat = {}
    for leg in ("FR", "FL", "RR", "RL"):
        vals = [v for joint, xs in by_joint.items() if joint.startswith(leg) for v in xs]
        leg_sat[leg] = {
            "raw_gt_1_ratio": sum(v > 1.0 for v in vals) / max(len(vals), 1),
            "raw_gt_08_ratio": sum(v > 0.8 for v in vals) / max(len(vals), 1),
            "residual_rms_proxy": (sum(v * v for v in residual_by_leg[leg]) / max(len(residual_by_leg[leg]), 1)) ** 0.5,
        }
    result = {
        "source_csv": str(csv_path),
        "rows": len(rows),
        "raw_action": {
            "gt_075_ratio": sum(v > 0.75 for v in raw_abs) / max(len(raw_abs), 1),
            "gt_08_ratio": sum(v > 0.8 for v in raw_abs) / max(len(raw_abs), 1),
            "gt_1_ratio": sum(v > 1.0 for v in raw_abs) / max(len(raw_abs), 1),
            "abs_mean": sum(raw_abs) / max(len(raw_abs), 1),
            "abs_max": max(raw_abs) if raw_abs else None,
        },
        "clip_count": {
            "mean": sum(clip_counts) / max(len(clip_counts), 1),
            "p90": quantile(clip_counts, 0.90),
            "max": max(clip_counts) if clip_counts else None,
        },
        "per_joint_raw_gt_1_ratio": {
            joint: sum(v > 1.0 for v in vals) / max(len(vals), 1) for joint, vals in by_joint.items()
        },
        "per_leg": leg_sat,
        "q_error_p95": quantile(qerr, 0.95),
        "q_cmd_delta_p95": quantile(qdelta, 0.95),
        "torque_p95": quantile(torque, 0.95),
        "warnings": [],
    }
    fr = leg_sat["FR"]["raw_gt_1_ratio"]
    others = [leg_sat[x]["raw_gt_1_ratio"] for x in ("FL", "RR", "RL")]
    if others and fr > max(others) * 1.5 and fr > 0.05:
        result["warnings"].append("FR raw action saturation is still much higher than other legs.")
    return result


def render_text(result: dict) -> str:
    lines = ["CPG policy log analysis", "", f"source_csv: {result['source_csv']}", f"rows: {result['rows']}", ""]
    for key, value in result.items():
        if key not in ("source_csv", "rows"):
            lines.append(f"{key}: {value}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("--skip-sec", type=float, default=0.0)
    parser.add_argument("--out-dir", default=str(package_dir()))
    args = parser.parse_args()
    result = analyze(Path(args.csv), skip_sec=args.skip_sec)
    out_dir = Path(args.out_dir)
    write_yaml(out_dir / "logs" / "cpg_policy_analysis_report.yaml", result)
    write_text(out_dir / "logs" / "cpg_policy_analysis_report.txt", render_text(result))
    print(f"wrote {out_dir / 'logs' / 'cpg_policy_analysis_report.yaml'}")


if __name__ == "__main__":
    main()
