#!/usr/bin/env python3
from __future__ import annotations

try:
    import torch
except ImportError:
    print("SKIP: torch is not installed in this Python. Run this inside the IsaacLab training environment.")
    raise SystemExit(0)

limits = torch.tensor([0.04, 0.06, 0.06] * 4).view(1, 12)
actions = torch.randn(100, 12) * 2.0
clipped = torch.clamp(actions, -1.0, 1.0)
delta = torch.clamp(clipped * limits, -limits, limits)
clip_count = torch.sum(torch.abs(actions) > 1.0, dim=1)
assert torch.max(torch.abs(delta - torch.clamp(delta, -limits, limits))) < 1.0e-6
assert torch.all(clip_count >= 0)
print("max_residual", float(torch.max(torch.abs(delta))))
print("clip_count_mean", float(torch.mean(clip_count.float())))
print("PASS")
