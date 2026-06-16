from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
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
profiles_mod = load("curriculum_profiles")
csv_mod = load("csv_playback")
urdf_mod = load("urdf_model")


def make_gait(num_envs=8, dt=0.02):
    default = torch.tensor([0.0, 0.70, -1.40] * 4).repeat(num_envs, 1)
    return gait_mod.FanfanReferenceGait(
        gait_mod.FanfanReferenceGaitCfg(), num_envs, "cpu", dt, default
    )


def make_small_gait(num_envs=1, dt=0.02):
    default = torch.tensor([
        -0.1571, 0.3491, -0.7854,
        0.1571, 0.3491, -0.7854,
        -0.1571, 0.3491, -0.7854,
        0.1571, 0.3491, -0.7854,
    ]).repeat(num_envs, 1)
    model = urdf_mod.load_fanfan_urdf_model()
    cfg = gait_mod.FanfanSmallHighFreqReferenceGaitCfg(
        thigh_length=model.thigh_length_m,
        calf_length=model.calf_length_m,
    )
    return gait_mod.FanfanReferenceGait(cfg, num_envs, "cpu", dt, default)


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


def test_big_stride_command_scaling_and_saturation():
    gait = make_gait(4)
    commands = torch.zeros(4, 3)
    commands[:, 0] = torch.tensor([0.10, 0.15, 0.18, 0.22])
    _, _, stride, frequency, swing_height = gait._command_parameters(commands)

    expected_stride = torch.tensor([0.038 * (2.0 / 3.0), 0.038, 0.0456, 0.0456])
    expected_frequency = torch.tensor([
        0.62 * (0.35 + 0.65 * (2.0 / 3.0) ** 0.5),
        0.62,
        0.682,
        0.682,
    ])
    expected_height = torch.tensor([0.058, 0.072, 0.072, 0.072])
    assert torch.allclose(stride, expected_stride, atol=1.0e-6)
    assert torch.allclose(frequency, expected_frequency, atol=1.0e-6)
    assert torch.allclose(swing_height, expected_height, atol=1.0e-6)
    assert abs(1.0 / float(frequency[1]) - 1.6129) < 1.0e-3


def test_phase_uses_control_dt():
    gait = make_gait(1, dt=0.02)
    command = torch.tensor([[0.15, 0.0, 0.0]])
    gait.update(command)
    assert torch.isclose(gait.base_phase[0], torch.tensor(0.62 * 0.02), atol=1.0e-7)
    assert abs(1.0 / float(gait.last_frequency[0]) - 1.6129) < 1.0e-3


def test_reference_hip_semantics_match_real_node():
    assert gait_mod.LEGACY_REFERENCE_HIP_OUTWARD_SIGNS == (1.0, 1.0, -1.0, 1.0)
    assert gait_mod.URDF_HIP_OUTWARD_SIGNS == (-1.0, 1.0, -1.0, 1.0)
    expected_swing_delta_signs = (-1.0, -1.0, 1.0, -1.0)
    assert tuple(-value for value in gait_mod.LEGACY_REFERENCE_HIP_OUTWARD_SIGNS) == expected_swing_delta_signs


def test_heavy_urdf_structure_and_default_fk():
    model = urdf_mod.load_fanfan_urdf_model()
    urdf_mod.validate_fanfan_urdf(model)
    assert abs(model.total_mass_kg - 7.242158331537168) < 1.0e-9
    assert abs(model.trunk_mass_kg - 2.76230213761) < 1.0e-9
    assert abs(model.thigh_length_m - 0.15606) < 1.0e-8
    assert abs(model.calf_length_m - 0.148940918050628) < 1.0e-6
    assert model.joint_order == semantics_mod.SIM_JOINT_NAMES

    poses = {
        "FR": (-0.1571, 0.3491, -0.7854),
        "FL": (0.1571, 0.3491, -0.7854),
        "RR": (-0.1571, 0.2269, -0.3491),
        "RL": (0.1571, 0.2269, -0.3491),
    }
    expected = {
        "FR": (0.199561, -0.163733, -0.265605),
        "FL": (0.199561, 0.163733, -0.265605),
        "RR": (-0.206952, -0.166589, -0.283634),
        "RL": (-0.206952, 0.166589, -0.283634),
    }
    for leg, pose in poses.items():
        actual = urdf_mod.forward_foot_position(model, leg, pose)
        assert all(abs(value - target) < 2.0e-6 for value, target in zip(actual, expected[leg]))


def test_small_high_frequency_defaults_and_continuity():
    gait = make_small_gait(dt=0.02)
    cfg = gait.cfg
    assert cfg.step_hz == 0.95
    assert cfg.stride_length == 0.024
    assert cfg.swing_height == 0.050
    assert cfg.duty_factor == 0.78
    assert cfg.front_swing_height_gain == 1.05
    assert cfg.rear_swing_height_gain == 0.64
    assert cfg.rear_lift_rise_fraction == 0.42
    assert cfg.rear_lift_fall_start == 0.58
    assert cfg.rear_stride_gain == 0.80
    assert cfg.warmup_sec == 2.0
    assert cfg.preload_fraction == 0.10
    assert cfg.post_touchdown_hold == 0.04
    assert cfg.reference_rate_limit_rad_s == 0.0
    assert cfg.apply_default_pose_offsets is False
    assert torch.allclose(gait.default_joint_pos[0], torch.tensor([
        -0.1571, 0.3491, -0.7854,
        0.1571, 0.3491, -0.7854,
        -0.1571, 0.3491, -0.7854,
        0.1571, 0.3491, -0.7854,
    ]))

    command = torch.tensor([[0.15, 0.0, 0.0]])
    previous = gait.get_q_ref().clone()
    seen = []
    previous_active = None
    for _ in range(round(12.0 / gait.dt)):
        q_ref = gait.update(command)
        previous = q_ref.clone()
        active = gait.last_active_swing_one_hot[0]
        assert int(active.sum()) <= 1
        active_index = int(active.argmax()) if active.sum() else None
        if active_index is not None and active_index != previous_active:
            seen.append(gait_mod.LEG_ORDER[active_index])
        previous_active = active_index

    assert seen[:5] == ["RR", "FR", "RL", "FL", "RR"]
    assert abs(1.0 / cfg.step_hz - 1.052632) < 1.0e-5

    x, z = gait._forward_sagittal(gait.last_q_ref[:, 1::3], gait.last_q_ref[:, 2::3])
    reach = torch.sqrt(x * x + z * z)
    max_reach = cfg.thigh_length + cfg.calf_length
    assert torch.all(reach <= max_reach - 0.0045)


def test_small_high_frequency_parameter_ranges():
    invalid = gait_mod.FanfanSmallHighFreqReferenceGaitCfg(step_hz=1.2)
    try:
        invalid.validate_parameters()
    except ValueError:
        pass
    else:
        raise AssertionError("Unsafe small-high-frequency step rate was accepted.")


def test_rear_stand_pose_candidates_and_lift_ik():
    cfg = gait_mod.FanfanSmallHighFreqReferenceGaitCfg(warmup_sec=0.0)
    candidates = ((0.30, -0.60), (0.36, -0.75), (0.42, -0.90))
    expected_margins = (0.0136, 0.0212, 0.0303)
    for (thigh, calf), expected_margin in zip(candidates, expected_margins, strict=True):
        default = torch.tensor([
            -0.1571, 0.3491, -0.7854,
            0.1571, 0.3491, -0.7854,
            -0.1571, thigh, calf,
            0.1571, thigh, calf,
        ]).unsqueeze(0)
        gait = gait_mod.FanfanReferenceGait(cfg, 1, "cpu", 0.01, default)
        x, z = gait._forward_sagittal(
            torch.tensor([[thigh]]), torch.tensor([[calf]])
        )
        reach = torch.sqrt(x * x + z * z)
        margin = cfg.thigh_length + cfg.calf_length - float(reach)
        assert abs(margin - expected_margin) < 5.0e-4
        target_lift = 0.030
        target_thigh, target_calf = gait._inverse_sagittal(x, z + target_lift)
        _, lifted_z = gait._forward_sagittal(target_thigh, target_calf)
        assert abs(float(lifted_z - z) - target_lift) < 1.0e-5
        assert float(target_calf) < calf


def test_level_symmetric_small_gait_stand_pose():
    cfg = gait_mod.FanfanSmallHighFreqReferenceGaitCfg(warmup_sec=0.0)
    thigh = 0.3491
    calf = -0.7854
    default = torch.tensor([
        -0.1571, thigh, calf,
        0.1571, thigh, calf,
        -0.1571, thigh, calf,
        0.1571, thigh, calf,
    ]).unsqueeze(0)
    gait = gait_mod.FanfanReferenceGait(cfg, 1, "cpu", 0.01, default)
    assert torch.max(gait.default_foot_x) - torch.min(gait.default_foot_x) < 1.0e-7
    assert torch.max(gait.default_foot_z) - torch.min(gait.default_foot_z) < 1.0e-7
    foot_radius = 0.018
    initial_base_height = 0.300
    initial_foot_center_height = initial_base_height + float(gait.default_foot_z[0, 0])
    assert initial_foot_center_height >= foot_radius - 1.0e-4


def test_small_high_frequency_preload_and_support_timing():
    gait = make_small_gait(dt=0.002)
    gait.cfg.warmup_sec = 0.0
    command = torch.tensor([[0.15, 0.0, 0.0]])
    saw_preload_before_each_leg = set()
    saw_post_touchdown_each_leg = set()
    for _ in range(round(6.0 / gait.dt)):
        gait.update(command)
        assert torch.all(
            gait.last_support_gate
            == (~gait.last_swing_mask).to(gait.last_support_gate.dtype)
        )
        for leg_index in range(4):
            leg_phase = float(gait.last_leg_phase[0, leg_index])
            post = float(gait.last_post_touchdown_gate[0, leg_index])
            if leg_phase > 0.98:
                assert not bool(gait.last_swing_mask[0, leg_index])
                other_legs = [index for index in range(4) if index != leg_index]
                assert float(gait.last_preload_gate[0, leg_index]) == 0.0
                assert torch.all(gait.last_preload_gate[0, other_legs] > 0.8)
                saw_preload_before_each_leg.add(leg_index)
            if (
                1.0 - gait.cfg.duty_factor
                <= leg_phase
                < 1.0 - gait.cfg.duty_factor + gait.cfg.post_touchdown_hold
                and post > 0.0
            ):
                saw_post_touchdown_each_leg.add(leg_index)
    assert saw_preload_before_each_leg == {0, 1, 2, 3}
    assert saw_post_touchdown_each_leg == {0, 1, 2, 3}


def test_front_swing_does_not_unload_same_side_rear():
    gait = make_small_gait(dt=0.001)
    gait.cfg.warmup_sec = 0.0
    command = torch.tensor([[0.15, 0.0, 0.0]])
    checked = set()
    swing_fraction = 1.0 - gait.cfg.duty_factor
    for _ in range(round(5.0 / gait.dt)):
        gait.update(command)
        active = gait.last_active_swing_one_hot[0]
        if int(active.sum()) != 1:
            continue
        active_index = int(active.argmax())
        if active_index not in (0, 1):
            continue
        phase = float(gait.last_leg_phase[0, active_index])
        if not 0.45 * swing_fraction < phase < 0.55 * swing_fraction:
            continue
        same_rear = 2 if active_index == 0 else 3
        assert float(gait.last_predicted_foot_lift[0, same_rear]) <= 5.0e-4
        checked.add(active_index)
    assert checked == {0, 1}


def test_rear_swing_has_lift_plateau():
    gait = make_small_gait(dt=0.002)
    gait.cfg.warmup_sec = 0.0
    command = torch.tensor([[0.15, 0.0, 0.0]])
    plateau_samples = {2: 0, 3: 0}
    expected_height = gait.cfg.swing_height * gait.cfg.rear_swing_height_gain
    for _ in range(round(5.0 / gait.dt)):
        gait.update(command)
        for leg_index in (2, 3):
            if not bool(gait.last_swing_mask[0, leg_index]):
                continue
            swing_fraction = 1.0 - gait.cfg.duty_factor
            progress = float(gait.last_leg_phase[0, leg_index]) / swing_fraction
            if 0.44 <= progress <= 0.56:
                lift = float(gait.last_predicted_foot_lift[0, leg_index])
                assert lift >= expected_height - 2.0e-4
                plateau_samples[leg_index] += 1
    assert plateau_samples[2] > 10
    assert plateau_samples[3] > 10


def test_csv_wide_policy_and_interpolation():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "policy.csv"
        columns = [f"q_policy_{index}" for index in range(12)]
        path.write_text(
            "time," + ",".join(columns) + "\n"
            + "0.0," + ",".join(["0"] * 12) + "\n"
            + "1.0," + ",".join(["1"] * 12) + "\n",
            encoding="utf-8",
        )
        times, values, value_space = csv_mod.load_joint_csv(path)
        assert value_space == "policy"
        playback = csv_mod.LoopingJointCsvPlayback(times, values, device=torch.device("cpu"))
        sample = playback.sample(torch.tensor([0.5]))
        assert torch.allclose(sample, torch.full((1, 12), 0.5))
        assert torch.allclose(playback.sample(torch.tensor([1.25])), torch.full((1, 12), 0.25))


def test_csv_wide_real_and_ros_long_form():
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        real_path = directory / "real.csv"
        real_columns = [f"q_real_{index}" for index in range(12)]
        real_path.write_text(
            "time," + ",".join(real_columns) + "\n"
            + "0.0," + ",".join(str(index) for index in range(12)) + "\n"
            + "1.0," + ",".join(str(index + 1) for index in range(12)) + "\n",
            encoding="utf-8",
        )
        _, real_values, value_space = csv_mod.load_joint_csv(real_path)
        assert value_space == "real"
        assert real_values.shape == (2, 12)

        long_path = directory / "long.csv"
        rows = ["time,elapsed,policy_joint_name,q_target_policy"]
        for elapsed in (0.0, 1.0):
            for index, name in enumerate(semantics_mod.POLICY_JOINT_NAMES):
                rows.append(f"{1000.0 + elapsed},{elapsed},{name},{index + elapsed}")
        long_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        times, policy_values, value_space = csv_mod.load_joint_csv(long_path)
        assert value_space == "policy"
        assert torch.allclose(times, torch.tensor([0.0, 1.0]))
        assert torch.allclose(policy_values[0], torch.arange(12, dtype=torch.float32))


def test_csv_rejects_non_monotonic_time():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "bad.csv"
        columns = [f"q_policy_{index}" for index in range(12)]
        path.write_text(
            "time," + ",".join(columns) + "\n"
            + "1.0," + ",".join(["0"] * 12) + "\n"
            + "0.5," + ",".join(["1"] * 12) + "\n",
            encoding="utf-8",
        )
        try:
            csv_mod.load_joint_csv(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-monotonic CSV time was accepted.")


def test_curriculum_boundaries_and_profiles():
    expected = {
        0: (1, (0.10, 0.15), 0.18),
        4_999: (1, (0.10, 0.15), 0.18),
        5_000: (2, (0.10, 0.18), 0.10),
        29_999: (2, (0.10, 0.18), 0.10),
        30_000: (3, (0.12, 0.20), 0.05),
        59_999: (3, (0.12, 0.20), 0.05),
        60_000: (4, (0.10, 0.22), 0.05),
    }
    for iteration, (stage_number, command_range, standing) in expected.items():
        stage = profiles_mod.get_wave_stage(iteration)
        assert stage["stage"] == stage_number
        assert stage["lin_vel_x"] == command_range
        assert stage["standing"] == standing

    stage_1 = profiles_mod.get_wave_stage(0)
    assert stage_1["mass_delta"] == (0.0, 0.0)
    assert stage_1["motor_strength"] == (1.0, 1.0)
    assert stage_1["delay_steps"] == (0, 0)
    assert stage_1["noise_level"] == 0.0
    stage_4 = profiles_mod.get_wave_stage(60_000)
    assert stage_4["mass_delta"] == (-0.30, 0.30)
    assert stage_4["motor_strength"] == (0.90, 1.05)
    assert stage_4["delay_steps"] == (0, 3)
    assert stage_4["push_enabled"] == 1.0


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
            expected_sign = -cfg.hip_outward_signs[leg_index]
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


def test_raw_joint_limit_clamp():
    targets = torch.tensor([[-2.0, 0.0, 2.0]])
    lower = torch.tensor([[-1.0, -1.0, -1.0]])
    upper = torch.tensor([[1.0, 1.0, 1.0]])
    clamped, mask = residual_mod.clamp_joint_targets(targets, lower, upper)
    assert torch.equal(clamped, torch.tensor([[-1.0, 0.0, 1.0]]))
    assert torch.equal(mask, torch.tensor([[True, False, True]]))


def test_joint_mapping_active_and_rest_schedule():
    kwargs = {
        "control_dt": 0.02,
        "initial_hold_sec": 2.0,
        "active_hold_sec": 1.0,
        "rest_sec": 1.0,
    }
    assert residual_mod.joint_mapping_index(0, **kwargs) == -1
    assert residual_mod.joint_mapping_index(99, **kwargs) == -1
    assert residual_mod.joint_mapping_index(100, **kwargs) == 0
    assert residual_mod.joint_mapping_index(149, **kwargs) == 0
    assert residual_mod.joint_mapping_index(150, **kwargs) == -1
    assert residual_mod.joint_mapping_index(200, **kwargs) == 1
    assert residual_mod.joint_mapping_index(1200, **kwargs) == 11


def test_reference_stage_and_vmc_limits():
    residual_mod.validate_reference_control_stage(0, False, "off")
    residual_mod.validate_reference_control_stage(1, False, "off")
    residual_mod.validate_reference_control_stage(2, True, "light")
    residual_mod.validate_reference_control_stage(3, True, "full")
    for invalid in ((0, True, "light"), (1, False, "full"), (2, False, "off")):
        try:
            residual_mod.validate_reference_control_stage(*invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid stage/VMC combination was accepted: {invalid}")

    raw = torch.full((1, 12), 1.0)
    previous = torch.zeros_like(raw)
    filtered = residual_mod.filter_vmc_delta(
        raw,
        previous,
        joint_limit_rad=0.03,
        rate_limit_rad_s=0.5,
        lowpass_alpha=0.20,
        dt=0.02,
    )
    assert torch.all(torch.abs(filtered) <= 0.01 + 1.0e-7)
    assert torch.all(torch.abs(filtered) <= 0.03 + 1.0e-7)


def test_rear_lift_three_stage_profile():
    profile = residual_mod.rear_lift_phase_profile
    assert profile(0.0, settle_sec=1.5, preload_sec=0.75, cycle_sec=2.0) == (0, 0.0, 0.0)
    phase, preload, lift = profile(1.875, settle_sec=1.5, preload_sec=0.75, cycle_sec=2.0)
    assert phase == 1
    assert 0.45 < preload < 0.55
    assert lift == 0.0
    phase, preload, lift = profile(2.25, settle_sec=1.5, preload_sec=0.75, cycle_sec=2.0)
    assert phase == 2 and preload == 1.0 and lift == 0.0
    phase, preload, lift = profile(3.25, settle_sec=1.5, preload_sec=0.75, cycle_sec=2.0)
    assert phase == 2 and preload == 1.0 and abs(lift - 1.0) < 1.0e-7


if __name__ == "__main__":
    test_smooth_gate()
    test_shape_finite_and_warmup()
    test_big_stride_command_scaling_and_saturation()
    test_phase_uses_control_dt()
    test_reference_hip_semantics_match_real_node()
    test_heavy_urdf_structure_and_default_fk()
    test_small_high_frequency_defaults_and_continuity()
    test_small_high_frequency_parameter_ranges()
    test_rear_stand_pose_candidates_and_lift_ik()
    test_level_symmetric_small_gait_stand_pose()
    test_small_high_frequency_preload_and_support_timing()
    test_front_swing_does_not_unload_same_side_rear()
    test_rear_swing_has_lift_plateau()
    test_csv_wide_policy_and_interpolation()
    test_csv_wide_real_and_ros_long_form()
    test_csv_rejects_non_monotonic_time()
    test_curriculum_boundaries_and_profiles()
    test_single_leg_and_order()
    test_semantic_round_trips_and_joint_isolation()
    test_real_rear_leg_reorder()
    test_active_foot_lifts_and_hip_direction()
    test_reset()
    test_residual_limit_and_filter()
    test_raw_joint_limit_clamp()
    test_joint_mapping_active_and_rest_schedule()
    test_reference_stage_and_vmc_limits()
    test_rear_lift_three_stage_profile()
    print("Fanfan reference gait pure-Torch tests passed.")
