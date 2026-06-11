#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from cpg_tool_common import JOINT_ORDER, csv_rows, package_dir, quantile, read_yaml, workspace_root, write_text, write_yaml


def _float(row: dict[str, str], *names: str) -> float | None:
    for name in names:
        if name in row and row[name] not in ("", None):
            try:
                return float(row[name])
            except ValueError:
                return None
    return None


def analyze_csv(csv_path: Path, limit: int | None = None) -> dict:
    rows = csv_rows(csv_path, limit=limit)
    by_joint: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        joint = row.get("policy_joint_name") or row.get("joint_name")
        if not joint or joint not in JOINT_ORDER:
            continue
        q_error = _float(row, "q_error_real", "q_error", "q_smooth_error_real")
        dq = _float(row, "dq_current_real", "dq_meas", "dq")
        torque = _float(row, "torque_measured", "tau_est_real", "tau_est", "torque")
        current = _float(row, "current", "current_a")
        q_delta = _float(row, "smooth_step_real", "raw_to_final_delta_real", "q_cmd_delta")
        for key, value in (
            ("q_error", q_error),
            ("dq", dq),
            ("torque", torque),
            ("current", current),
            ("q_cmd_delta", q_delta),
        ):
            if value is not None:
                by_joint[joint][key].append(abs(value))

    joint_stats = {}
    for joint in JOINT_ORDER:
        stats = {}
        for key, values in by_joint[joint].items():
            stats[key] = {
                "mean": sum(values) / max(len(values), 1),
                "p90": quantile(values, 0.90),
                "p95": quantile(values, 0.95),
                "max": max(values) if values else None,
            }
        joint_stats[joint] = stats
    return {"source_csv": str(csv_path), "rows": len(rows), "joint_stats": joint_stats}


def fitted_profile(analysis: dict, motor_profile: dict) -> dict:
    q95 = [s.get("q_error", {}).get("p95") for s in analysis["joint_stats"].values()]
    torque95 = [s.get("torque", {}).get("p95") for s in analysis["joint_stats"].values()]
    current95 = [s.get("current", {}).get("p95") for s in analysis["joint_stats"].values()]
    q_delta95 = [s.get("q_cmd_delta", {}).get("p95") for s in analysis["joint_stats"].values()]
    q95_clean = [v for v in q95 if v is not None]
    torque95_clean = [v for v in torque95 if v is not None]
    current95_clean = [v for v in current95 if v is not None]
    q_delta95_clean = [v for v in q_delta95 if v is not None]
    safe_torque = (
        motor_profile.get("motor", {}).get("safe_training_torque_nm")
        or motor_profile.get("safety", {}).get("torque_limit_for_training_nm")
        or 5.0
    )
    max_delta = min(0.03, max(q_delta95_clean) * 1.5) if q_delta95_clean else 0.03
    return {
        "actuator_dynamics": {
            "estimated_delay_ms": None,
            "delay_random_range_ms": [0.0, 60.0],
            "estimated_bandwidth_hz": None,
            "q_error_p90_rad": max([s.get("q_error", {}).get("p90") or 0.0 for s in analysis["joint_stats"].values()]),
            "q_error_p95_rad": max(q95_clean) if q95_clean else None,
            "torque_p95_nm": max(torque95_clean) if torque95_clean else None,
            "current_p95_a": max(current95_clean) if current95_clean else None,
        },
        "control_limits": {
            "residual_limit_hip_rad": 0.04,
            "residual_limit_thigh_rad": 0.06,
            "residual_limit_calf_rad": 0.06,
            "max_delta_per_step_rad": max_delta,
            "lowpass_alpha": 0.35,
            "torque_limit_train_nm": safe_torque,
            "torque_limit_deploy_nm": min(6.0, float(safe_torque) * 1.2),
        },
        "training_randomization": {
            "motor_strength_range": [0.65, 1.0],
            "kp_scale_range": [0.85, 1.15],
            "kd_scale_range": [0.85, 1.15],
            "action_delay_frames": [0, 3],
            "joint_zero_offset_rad": [-0.03, 0.03],
        },
        "cpg_limits": {
            "freq_min_hz": 0.8,
            "freq_max_hz": 1.8,
            "step_height_m": 0.030,
            "step_length_min_m": 0.015,
            "step_length_max_m": 0.060,
        },
        "warnings": ["Delay/bandwidth need paired q_cmd/q_meas time-series validation; current report uses conservative defaults."],
    }


def render_text(analysis: dict, fitted: dict) -> str:
    lines = ["Motor dynamics fit report", "", f"source_csv: {analysis['source_csv']}", f"rows: {analysis['rows']}", ""]
    for joint, stats in analysis["joint_stats"].items():
        lines.append(f"[{joint}] {stats}")
    lines.append("")
    lines.append("fitted recommendations:")
    for section, data in fitted.items():
        lines.append(f"{section}: {data}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(workspace_root() / "walk_as014_kp45_kd4_cmd012_basecmd_real.csv"))
    parser.add_argument("--motor-profile", default=str(package_dir() / "config" / "motor_profile.yaml"))
    parser.add_argument("--out-dir", default=str(package_dir()))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    analysis = analyze_csv(Path(args.csv), limit=args.limit)
    fitted = fitted_profile(analysis, read_yaml(Path(args.motor_profile)))
    write_yaml(out_dir / "logs" / "motor_dynamics_fit_report.yaml", {"analysis": analysis, "fitted": fitted})
    write_yaml(out_dir / "config" / "motor_profile_fitted.yaml", fitted)
    write_text(out_dir / "logs" / "motor_dynamics_fit_report.txt", render_text(analysis, fitted))
    print(f"wrote {out_dir / 'config' / 'motor_profile_fitted.yaml'}")


if __name__ == "__main__":
    main()
