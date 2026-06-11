#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed in this Python. Run this inside the IsaacLab training environment.")
    raise SystemExit(0)

sys.path.append(str(Path(__file__).resolve().parents[1].parent))
from fanfan_rl_cpg.cpg_cfg import CPGCfg
from fanfan_rl_cpg.cpg_generator import QuadrupedCPG


cfg = CPGCfg()
cfg.initial_phase_random = False
default = torch.zeros(1, 12)
cpg = QuadrupedCPG(cfg, "cpu", 1, 0.02, default)
phase = cpg.compute_phase(torch.tensor([[0.10, 0.0, 0.0]]))
cycles = torch.remainder(phase / (2.0 * math.pi), 1.0)[0]
print("phase cycles FR FL RR RL:", cycles.tolist())
assert abs(float(cycles[0] - cycles[3])) < 1.0e-5, "FR/RL should be in phase"
assert abs(float(cycles[1] - cycles[2])) < 1.0e-5, "FL/RR should be in phase"
assert abs(float(torch.remainder(cycles[1] - cycles[0], 1.0)) - 0.5) < 1.0e-5, "diagonal pairs should differ by 0.5 cycle"
print("PASS")
