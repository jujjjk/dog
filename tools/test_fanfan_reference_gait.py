from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg_residual"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


gait_mod = load("reference_gait")
residual_mod = load("residual_math")


def make_gait(num_envs=8, dt=0.02):
    default = torch.tensor([0.0, 0.70, -1.40] * 4).repeat(num_envs, 1)
    return gait_mod.FanfanReferenceGait(
        gait_mod.FanfanReferenceGaitCfg(), num_envs, "cpu", dt, default
    )


def test_smooth_gate():
    gait = make_gait(4)
    commands = torch.zeros(4, 3)
    commands[:, 0] = torch.tensor([0.0, 0.005, 0.0175, 0.030])
    gate, _, _, _, height = gait._command_parameters(commands)
    assert torch.allclose(gate[:2], torch.zeros(2))
    assert 0.0 < gate[2] < 1.0
    assert torch.isclose(gate[3], torch.tensor(1.0))
    assert height[2] < 0.047


def test_shape_finite_and_warmup():
    gait = make_gait()
    command = torch.tensor([[0.10, 0.0, 0.0]]).repeat(8, 1)
    q = gait.update(command)
    assert q.shape == (8, 12)
    assert torch.isfinite(q).all()
    assert torch.all(gait.last_warmup < 1.0)
    assert gait.get_phase_features().shape == (8, 8)
    assert gait.last_active_swing_one_hot.shape == (8, 4)


def test_single_leg_and_order():
    gait = make_gait(1, dt=0.005)
    command = torch.tensor([[0.15, 0.0, 0.0]])
    seen = []
    previous = None
    for _ in range(4000):
        gait.update(command)
        active = gait.last_active_swing_one_hot[0]
        assert int(active.sum().item()) <= 1
        current = int(active.argmax()) if active.sum() else None
        if current is not None and current != previous:
            seen.append(gait_mod.LEG_ORDER[current])
        previous = current
        if len(seen) >= 5:
            break
    assert seen[:5] == ["RR", "FR", "RL", "FL", "RR"]


def test_reset():
    gait = make_gait()
    gait.update(torch.tensor([[0.15, 0.0, 0.0]]).repeat(8, 1))
    gait.reset(torch.tensor([1, 3]))
    assert torch.all(gait.base_phase[[1, 3]] == 0.0)
    assert torch.all(gait.last_active_swing_one_hot[[1, 3]] == 0.0)


def test_residual_limit_and_filter():
    raw = torch.full((2, 12), 100.0)
    previous = torch.zeros_like(raw)
    scale = torch.tensor([0.05, 0.08, 0.10] * 4)
    filtered = residual_mod.filter_residual(raw, previous, scale, 0.30)
    assert torch.all(filtered <= scale)
    assert torch.allclose(filtered, 0.30 * scale.expand_as(filtered), atol=1.0e-5)


if __name__ == "__main__":
    test_smooth_gate()
    test_shape_finite_and_warmup()
    test_single_leg_and_order()
    test_reset()
    test_residual_limit_and_filter()
    print("Fanfan reference gait pure-Torch tests passed.")
