from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg_residual"


TEST_PACKAGE = "_fanfan_residual_test"
package = types.ModuleType(TEST_PACKAGE)
package.__path__ = [str(PKG)]
sys.modules[TEST_PACKAGE] = package


def load(name: str):
    full_name = f"{TEST_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(full_name, PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


semantics_mod = load("joint_semantics")
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


def test_semantic_round_trips_and_joint_isolation():
    adapter = semantics_mod.FanfanJointSemanticAdapter()
    policy = torch.randn(5, 12)
    assert torch.allclose(adapter.sim_to_policy(adapter.policy_to_sim(policy)), policy)
    assert torch.allclose(adapter.real_to_policy(adapter.policy_to_real(policy)), policy)

    for joint_index in range(12):
        impulse = torch.zeros(1, 12)
        impulse[0, joint_index] = 1.0
        changed = torch.nonzero(adapter.policy_to_sim(impulse)[0], as_tuple=False).flatten().tolist()
        assert changed == [joint_index]

    adapter.assert_sim_joint_names(semantics_mod.SIM_JOINT_NAMES)
    wrong_order = list(semantics_mod.SIM_JOINT_NAMES)
    wrong_order[6], wrong_order[9] = wrong_order[9], wrong_order[6]
    try:
        adapter.assert_sim_joint_names(wrong_order)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid simulator joint order was accepted.")


def test_real_rear_leg_reorder():
    adapter = semantics_mod.FanfanJointSemanticAdapter()
    policy = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    real = adapter.policy_to_real(policy)
    policy_signed = policy * adapter.real_sign
    assert torch.equal(real[:, 6:9], policy_signed[:, 9:12])
    assert torch.equal(real[:, 9:12], policy_signed[:, 6:9])


def test_active_foot_lifts_and_hip_direction():
    cfg = gait_mod.FanfanReferenceGaitCfg(
        warmup_sec=0.0,
        diag_support_hip_amp=0.0,
        same_rear_unload_hip_amp=0.0,
    )
    default = torch.tensor([
        -0.1571, 0.3491, -0.7854,
        0.1571, 0.3491, -0.7854,
        -0.1571, 0.2269, -0.3491,
        0.1571, 0.2269, -0.3491,
    ]).unsqueeze(0)
    gait = gait_mod.FanfanReferenceGait(cfg, 1, "cpu", 0.005, default)
    command = torch.tensor([[0.15, 0.0, 0.0]])
    checked = set()
    for _ in range(4000):
        gait.update(command)
        active = gait.last_active_swing_one_hot[0]
        if int(active.sum().item()) != 1:
            continue
        leg_index = int(active.argmax().item())
        phase = float(gait.last_leg_phase[0, leg_index])
        swing_fraction = 1.0 - cfg.duty_factor
        if 0.35 * swing_fraction < phase < 0.65 * swing_fraction:
            lift = gait.last_predicted_foot_lift[0]
            assert lift[leg_index] > 0.0
            support_indices = [index for index in range(4) if index != leg_index]
            assert lift[leg_index] > torch.max(lift[support_indices])
            hip_index = leg_index * 3
            hip_delta = gait.last_q_ref[0, hip_index] - gait.default_joint_pos[0, hip_index]
            expected_sign = semantics_mod.SIM_HIP_SIDE_SIGNS[leg_index]
            assert float(hip_delta) * expected_sign > 0.0
            checked.add(leg_index)
        if len(checked) == 4:
            break
    assert checked == {0, 1, 2, 3}


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
    test_semantic_round_trips_and_joint_isolation()
    test_real_rear_leg_reorder()
    test_active_foot_lifts_and_hip_direction()
    test_reset()
    test_residual_limit_and_filter()
    print("Fanfan reference gait pure-Torch tests passed.")
