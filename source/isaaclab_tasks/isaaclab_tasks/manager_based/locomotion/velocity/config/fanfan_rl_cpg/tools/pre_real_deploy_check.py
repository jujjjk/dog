#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_cpg_policy_log import analyze, render_text
from cpg_tool_common import package_dir, read_yaml, write_text, write_yaml


def check(result: dict, motor_profile_fitted: dict) -> dict:
    safe_torque = motor_profile_fitted.get("control_limits", {}).get("torque_limit_train_nm", 5.0)
    reasons = []
    status = "PASS"
    raw_gt_1 = result["raw_action"]["gt_1_ratio"]
    raw_gt_08 = result["raw_action"]["gt_08_ratio"]
    clip_mean = result["clip_count"]["mean"]
    if raw_gt_1 >= 0.03:
        status = "FAIL"
        reasons.append(f"raw>|1| ratio {raw_gt_1:.3f} >= 0.03")
    if raw_gt_08 >= 0.10:
        status = "FAIL"
        reasons.append(f"raw>|0.8| ratio {raw_gt_08:.3f} >= 0.10")
    if clip_mean >= 0.3:
        status = "FAIL"
        reasons.append(f"clip_count mean {clip_mean:.3f} >= 0.3")
    for leg, data in result["per_leg"].items():
        if data["raw_gt_1_ratio"] >= 0.10:
            status = "FAIL"
            reasons.append(f"{leg} raw>|1| ratio {data['raw_gt_1_ratio']:.3f} >= 0.10")
    residuals = [max(v["residual_rms_proxy"], 1.0e-6) for v in result["per_leg"].values()]
    if max(residuals) / min(residuals) >= 2.0:
        status = "WARNING" if status == "PASS" else status
        reasons.append("per-leg residual RMS max/min >= 2.0")
    torque_p95 = result.get("torque_p95")
    if torque_p95 is not None and torque_p95 >= float(safe_torque) * 0.7:
        status = "WARNING" if status == "PASS" else status
        reasons.append(f"torque p95 {torque_p95:.3f} is near safe training torque budget")
    if result["warnings"]:
        status = "WARNING" if status == "PASS" else status
        reasons.extend(result["warnings"])
    return {"status": status, "reasons": reasons, "metrics": result}


def render_report(report: dict) -> str:
    lines = ["Pre-real deploy check", "", f"STATUS: {report['status']}", ""]
    if report["reasons"]:
        lines.append("Reasons:")
        for reason in report["reasons"]:
            lines.append(f"  - {reason}")
    else:
        lines.append("All configured gates passed.")
    lines.append("")
    lines.append(render_text(report["metrics"]))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--motor-profile-fitted", default=str(package_dir() / "config" / "motor_profile_fitted.yaml"))
    parser.add_argument("--out-dir", default=str(package_dir()))
    parser.add_argument("--skip-sec", type=float, default=1.0)
    args = parser.parse_args()
    result = analyze(Path(args.csv), skip_sec=args.skip_sec)
    report = check(result, read_yaml(Path(args.motor_profile_fitted)))
    out_dir = Path(args.out_dir)
    write_yaml(out_dir / "logs" / "pre_real_deploy_check_report.yaml", report)
    write_text(out_dir / "logs" / "pre_real_deploy_check_report.txt", render_report(report))
    print(report["status"])
    print(f"wrote {out_dir / 'logs' / 'pre_real_deploy_check_report.yaml'}")


if __name__ == "__main__":
    main()
