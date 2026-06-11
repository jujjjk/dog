#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


POLICY_JOINT_ORDER = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def default_deploy_pkg() -> Path:
    return Path(r"e:\python_text\机器狗\控制神网\mydog_ros2_ws\src\mydog_policy\mydog_policy")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check policy/deploy joint-order consistency.")
    parser.add_argument("--deploy-pkg", type=Path, default=default_deploy_pkg())
    args = parser.parse_args()

    deploy_pkg = args.deploy_pkg
    if deploy_pkg.exists():
        sys.path.insert(0, str(deploy_pkg.parent))

    try:
        from mydog_policy.semantic_mapper import JointSemanticMapper
    except Exception as exc:
        print(f"[WARN] Could not import deploy semantic mapper: {exc}")
        print("Policy order:")
        for i, name in enumerate(POLICY_JOINT_ORDER):
            leg, joint_type, *_ = name.split("_")
            print(f"policy[{i:02d}] leg={leg:2s} joint={joint_type:5s} name={name}")
        return 1

    mapper = JointSemanticMapper()
    print("policy_index,policy_joint,real_index,real_joint,leg,joint_type")
    for policy_i, policy_name in enumerate(mapper.policy_joint_names):
        real_i = int(mapper.policy_to_real_index[policy_i])
        real_name = mapper.real_joint_names[real_i]
        leg, joint_type, *_ = policy_name.split("_")
        print(f"{policy_i:02d},{policy_name},{real_i:02d},{real_name},{leg},{joint_type}")

    if tuple(mapper.policy_joint_names) != POLICY_JOINT_ORDER:
        print("[FAIL] Deploy policy_joint_names differ from Fanfan policy order.")
        return 2

    expected_real_legs = [name.split("_")[0] for name in mapper.real_joint_names]
    if expected_real_legs[6:12] != ["RL", "RL", "RL", "RR", "RR", "RR"]:
        print("[WARN] Real rear order is not RL then RR; verify semantic mapper intentionally handles it.")

    print("[OK] Policy joint order matches. RR/RL differences are handled through semantic mapper indices.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
