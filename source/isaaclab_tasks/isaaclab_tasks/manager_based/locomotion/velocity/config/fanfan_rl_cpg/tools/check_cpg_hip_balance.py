#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed. Run this inside the IsaacLab training environment.")
    raise SystemExit(0)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(PACKAGE_DIR.parent))

try:
    from fanfan_rl_cpg.cpg_cfg import CPGCfg, FANFAN_POLICY_JOINT_ORDER
    from fanfan_rl_cpg.cpg_generator import QuadrupedCPG

    HAVE_CPG = True
except Exception as exc:
    print(f"WARN: could not import QuadrupedCPG ({exc}); using the same pure torch hip-balance math.")
    HAVE_CPG = False
    FANFAN_POLICY_JOINT_ORDER = (
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


DEFAULT_JOINT_POS = torch.tensor(
    [[-0.1571, 0.3491, -0.7854, 0.1571, 0.3491, -0.7854, -0.1571, 0.2269, -0.3491, 0.1571, 0.2269, -0.3491]],
    dtype=torch.float32,
)
HIP_CLIPS = {
    "FR": (-0.16, 0.08),
    "FL": (-0.08, 0.16),
    "RR": (-0.16, 0.08),
    "RL": (-0.08, 0.16),
}


def _fallback_cycle(samples: int = 240) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    leg_order = ("FR", "FL", "RR", "RL")
    signs = torch.tensor((-1.0, 1.0, -1.0, 1.0), dtype=torch.float32).view(1, -1)
    hip_amp = 0.025
    stance_amp = 0.020
    relax_amp = 0.008
    max_abs = 0.06
    duty = 0.60
    swing_fraction = max(1.0 - duty, 0.05)
    offsets = torch.tensor((0.0, 0.5, 0.5, 0.0), dtype=torch.float32).view(1, -1)
    base = torch.linspace(0.0, 1.0, samples + 1, dtype=torch.float32)[:-1].view(-1, 1)
    phase01 = torch.remainder(base + offsets, 1.0)
    swing = phase01 < swing_fraction
    s_swing = torch.clamp(phase01 / swing_fraction, 0.0, 1.0)
    s_stance = torch.clamp((phase01 - swing_fraction) / max(1.0 - swing_fraction, 1.0e-5), 0.0, 1.0)
    swing_shape = torch.where(swing, torch.sin(math.pi * s_swing), torch.zeros_like(phase01))
    stance_shape = torch.where(swing, torch.zeros_like(phase01), torch.sin(math.pi * s_stance))
    stride = torch.where(swing, -1.0 + 2.0 * s_swing, 1.0 - 2.0 * s_stance)
    hip_stride = hip_amp * stride
    hip_balance = torch.clamp(signs * stance_amp * stance_shape - signs * relax_amp * swing_shape, -max_abs, max_abs)
    hip_targets = DEFAULT_JOINT_POS[:, (0, 3, 6, 9)] + hip_stride + hip_balance
    cfg = {
        "leg_order": leg_order,
        "hip_balance_signs": tuple(float(x) for x in signs.flatten()),
        "hip_amp": hip_amp,
        "hip_stance_widen_amp": stance_amp,
        "hip_swing_relax_amp": relax_amp,
        "residual_limit_hip": 0.08,
    }
    return cfg, hip_targets, hip_balance


def _cpg_cycle(samples: int = 240) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
    cfg = CPGCfg()
    cfg.initial_phase_random = False
    cfg.joint_sine.hip_amp = 0.025
    cfg.joint_sine.enable_hip_balance = True
    cfg.joint_sine.hip_stance_widen_amp = 0.020
    cfg.joint_sine.hip_swing_relax_amp = 0.008
    cfg.joint_sine.hip_balance_signs = (-1.0, 1.0, -1.0, 1.0)
    cfg.joint_sine.hip_balance_max_abs = 0.06
    cfg.residual_limit_hip = 0.08
    cpg = QuadrupedCPG(cfg, "cpu", 1, 1.0 / samples, DEFAULT_JOINT_POS)
    hip_rows = []
    balance_rows = []
    for _ in range(samples):
        q = cpg.update(torch.tensor([[0.12, 0.0, 0.0]], dtype=torch.float32)).detach()
        hip_rows.append(q[:, (0, 3, 6, 9)])
        balance_rows.append(cpg.last_hip_balance_delta.detach())
    info = {
        "leg_order": cfg.leg_order,
        "hip_balance_signs": cfg.joint_sine.hip_balance_signs,
        "hip_amp": cfg.joint_sine.hip_amp,
        "hip_stance_widen_amp": cfg.joint_sine.hip_stance_widen_amp,
        "hip_swing_relax_amp": cfg.joint_sine.hip_swing_relax_amp,
        "residual_limit_hip": cfg.residual_limit_hip,
    }
    return info, torch.cat(hip_rows, dim=0), torch.cat(balance_rows, dim=0)


def main() -> int:
    info, hip_targets, hip_balance = _cpg_cycle() if HAVE_CPG else _fallback_cycle()
    leg_order = tuple(info["leg_order"])
    print("leg_order:", leg_order)
    print("hip_balance_signs:", info["hip_balance_signs"])
    print("hip_amp:", info["hip_amp"])
    print("hip_stance_widen_amp:", info["hip_stance_widen_amp"])
    print("hip_swing_relax_amp:", info["hip_swing_relax_amp"])
    print("residual_limit_hip:", info["residual_limit_hip"])
    print()

    expected_sign = {"FR": -1.0, "FL": 1.0, "RR": -1.0, "RL": 1.0}
    had_warning = False
    for idx, leg in enumerate(leg_order):
        values = hip_targets[:, idx]
        balance = hip_balance[:, idx]
        print(
            f"{leg}_hip target min/max/mean: "
            f"{float(values.min()): .5f} / {float(values.max()): .5f} / {float(values.mean()): .5f}"
        )
        print(f"{leg}_hip balance mean: {float(balance.mean()): .5f}")
        if float(balance.mean()) * expected_sign[leg] <= 0.0:
            print(f"WARN: {leg} mean hip balance does not match expected side sign {expected_sign[leg]:+.0f}.")
            had_warning = True
        lo, hi = HIP_CLIPS[leg]
        if float(values.min()) < lo or float(values.max()) > hi:
            print(
                f"WARN: {leg} hip target crosses rough action clip ({lo:.3f}, {hi:.3f}); "
                "reduce hip_amp/hip_stance_widen_amp or adjust clips after review."
            )
            had_warning = True
    print()
    print("CHECK:", "WARNINGS" if had_warning else "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
