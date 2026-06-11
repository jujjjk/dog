# Fanfan RL+CPG Usage

This branch keeps `fanfan_a1_clean` untouched and registers new tasks under `FanfanRlCpg`.

## Generate Profiles

Run from the IsaacLab workspace root:

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_urdf_for_cpg.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/extract_motor_specs.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/fit_motor_dynamics_from_real_data.py --csv /path/to/real_log.csv
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/generate_randomization_from_urdf.py
```

Generated files:

- `config/motor_profile.yaml`
- `config/motor_profile_fitted.yaml`
- `config/randomization_profile.yaml`
- `logs/cpg_urdf_report.*`
- `logs/motor_dynamics_fit_report.*`
- `logs/randomization_from_urdf_report.*`

## CPG Checks

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_phase.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_joint_output.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/test_cpg_residual_action.py
```

These tests require the IsaacLab Python environment with `torch`.

The default simulation CPG basis is `joint_sine`: it keeps the validated stand
pose, uses the URDF joint order, and adds a small swing-phase lift envelope.
`foot_ik` remains experimental and is not the default path.

## Hip Balance

The current RL+CPG path adds conservative hip participation for diagonal trot
support.  The previous defaults used `hip_amp=0.0` and
`residual_limit_hip=0.04`, so the hip joints had almost no periodic CPG motion
and only about 2.3 deg of residual authority.  The new defaults add a small
hip stride term, widen stance legs slightly, relax swing legs slightly, and
use a small 0.03 rad hip residual limit while keeping the deploy-like target
filter enabled.  Residual hip motion is also phase-gated: stance legs keep at
least 0.008 rad outward support, while swing legs are capped at 0.035 rad
outward so the policy cannot pin one diagonal through residuals.

The side direction is configurable through:

```python
cpg_cfg.joint_sine.hip_balance_signs = (-1.0, 1.0, -1.0, 1.0)
```

The order is `("FR", "FL", "RR", "RL")`.  The default assumes right hips
abduct in the negative direction and left hips abduct in the positive direction.
If real-hardware semantic checks show the direction is reversed, change
`hip_balance_signs`; do not edit `compute_joint_sine()`.

Minimal rollback for ablation:

```python
cpg_cfg.joint_sine.enable_hip_balance = False
cpg_cfg.joint_sine.hip_amp = 0.0
cpg_cfg.residual_limit_hip = 0.04
cpg_cfg.enable_phase_aware_hip_gate = False
```

Before a long run, inspect TensorBoard for hip residual saturation, the gap
between `q_raw_hip` and `q_cmd_hip`, diagonal contact pattern, base roll/roll
rate, and applied torque.  The helper below prints one CPG cycle and warns if
hip targets cross the rough action clips:

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/check_cpg_hip_balance.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/check_joint_order_consistency.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_diagonal_support_csv.py /path/to/policy_log.csv
```


## Play And Train

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-CPGOnly-v0 \
  --num_envs 16

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-v0 \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless
```

## Log Gate

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_cpg_policy_log.py /path/to/policy_log.csv --skip-sec 1.0
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/pre_real_deploy_check.py --csv /path/to/policy_log.csv
```

## Deploy-Side ONNX Test

Default mode is pure RL. Enable CPG only explicitly:

```bash
python policy_onnx_test.py --onnx policy.onnx --vx 0.05 --action-mode cpg_only
python policy_onnx_test.py --onnx policy.onnx --vx 0.05 --action-mode cpg_residual
```

Do not walk the robot if `pre_real_deploy_check.py` reports `FAIL`.
