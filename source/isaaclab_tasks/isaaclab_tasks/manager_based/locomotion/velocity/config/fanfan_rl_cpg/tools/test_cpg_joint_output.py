#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed in this Python. Run this inside the IsaacLab training environment.")
    raise SystemExit(0)

sys.path.append(str(Path(__file__).resolve().parents[1].parent))
from fanfan_rl_cpg.cpg_cfg import CPGCfg, FANFAN_POLICY_JOINT_ORDER
from fanfan_rl_cpg.cpg_generator import QuadrupedCPG


out = Path(__file__).resolve().parents[1] / "logs" / "test_cpg_joint_output.csv"
out.parent.mkdir(parents=True, exist_ok=True)
cfg = CPGCfg()
cfg.initial_phase_random = False
default = torch.tensor([[-0.1571, 0.3491, -0.7854, 0.1571, 0.3491, -0.7854, -0.1571, 0.2269, -0.3491, 0.1571, 0.2269, -0.3491]])
cpg = QuadrupedCPG(cfg, "cpu", 1, 0.02, default)
prev = None
max_delta = 0.0
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["t", *FANFAN_POLICY_JOINT_ORDER])
    for i in range(250):
        q = cpg.update(torch.tensor([[0.10, 0.0, 0.0]])).detach().cpu()[0]
        if prev is not None:
            max_delta = max(max_delta, float(torch.max(torch.abs(q - prev))))
        prev = q.clone()
        writer.writerow([i * 0.02, *[float(x) for x in q]])
print("wrote", out)
print("max_delta", max_delta)
assert max_delta < 0.05
print("PASS")
