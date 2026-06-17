# Fanfan Wave Gait Residual

## Small high-frequency reference-only gait

The residual task now has a separate conservative small-step wave gait. It
does not replace or modify the original big-stride reference. Reference tasks
use the 7.242 kg
`fanfan_mass_scaled_only_trunk_plus_800g.urdf` and a distinct converted USD
filename so a cached USD from the old 2.92 kg URDF cannot be reused.

URDF selection order is:

1. `FANFAN_HEAVY_URDF_PATH`
2. `FANFAN_URDF_PATH`
3. `<workspace>/fanfan/urdf/fanfan_mass_scaled_only_trunk_plus_800g.urdf`

Startup validates the 12-joint `FR, FL, RR, RL` order, every joint origin,
axis and limit, the `17 N*m / 44 rad/s` URDF limits, total and Trunk mass, and
the `0.15606 / 0.14894 m` leg lengths. A mismatch stops the task immediately.
The small gait uses the unmodified real-machine stand pose as its reference
zero; it does not inherit the legacy gait's hidden hip or rear-leg offsets.

Default small-high-frequency parameters:

```text
step_hz       = 0.82 Hz
stride_length = 0.026 m
swing_height  = 0.062 m
duty_factor   = 0.78
warmup_sec    = 4.0 s
swing_order   = RR -> FR -> RL -> FL
```

The gait remains one-leg-at-a-time. The front swing-height gain is `1.08`,
rear gain is `1.00`, and the front/rear stride gains are `1.00/0.92`. Its IK
keeps a `5 mm` radial workspace margin. The current reference uses `0.95 Hz`,
`24 mm` front stride, about `19 mm` rear stride, about `52.5 mm` front lift,
and `32 mm` rear lift with a short smooth plateau. Stage 0 deliberately has no reference or deployment
slew filter so the raw CPG/IK timing remains observable.

### Reference stages

All stages ignore the zero-agent action, use fixed `cmd_x=0.15`, and disable
policy loading, residuals, delay, randomization, observation noise, pushes,
rewards, curriculum and automatic fall reset.

| Stage | Control path |
| --- | --- |
| 0 | CPG/IK -> policy-to-sim -> URDF joint clamp |
| 1 / Debug | Stage 0 + `10 rad/s` rate + `240 rad/s^2` acceleration + `12 N*m` target-error limits |
| 1 / Safe | Stage 0 + `5 rad/s` rate + `180 rad/s^2` acceleration + `6 N*m` target-error limits |
| 2 | CPG/IK + bounded light VMC -> Stage-1 safety chain |
| 3 | Reserved full-VMC provider; deliberately refuses to run until calibrated |

Light VMC only changes stance-leg sagittal targets. It uses roll/pitch angular
feedback and body-height error, limits foot-height correction to `6 mm`,
joint correction to `0.03 rad`, correction rate to `0.5 rad/s`, and applies a
`0.20` low-pass factor. Swing legs receive no VMC correction.

Run and record each stage:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage0-Reference-v0 \
  --num_envs 1 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage1-Debug-Reference-v0 \
  --num_envs 1 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage1-Safe-Reference-v0 \
  --num_envs 1 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage2-Reference-v0 \
  --num_envs 1 --duration 60
```

`SmallHighFreq-Reference-v0` and the legacy `Stage1-Reference-v0` name are
Stage-1 Debug aliases. CSV output defaults to a
task-specific file under `logs/reference_debug/`, so Stage 0/1/2 runs do not
overwrite one another. It includes stage, gait parameters,
swing/stance masks, `q_cpg`, `q_vmc_delta`, `q_ref`, every safety-chain target,
`q_final`, `q_actual`, errors, attitude, angular velocity, estimated torque,
predicted foot lift, actual world foot height, and every clamp mask.

Do not start residual learning until Stage 0 has correct leg order, continuous
lift/touchdown and stable tripod support. Use Stage-1 Debug to decide whether
the simulated joints can execute the trajectory. Stage-1 Safe is only a
hardware-proximity check: if it reports persistent clipping or less than 60%
actual/predicted foot lift, the trajectory exceeds the `5 rad/s / 6 N*m`
profile and the filtered near-static motion must not be interpreted as a
valid gait. If Stage 0 is unstable, tune stride, swing height, duty factor,
support preload and stand pose before touching VMC.

### Rear-leg isolated lift test

The small-high-frequency reference now uses the same sagittal stand angles
for front and rear legs: `thigh=0.3491 rad`, `calf=-0.7854 rad`. The URDF
front/rear links and joint axes are symmetric, so this places all four foot
centers at the same FK height and removes the old straight-rear-leg bias.
The small-reference initial Trunk height is `0.300 m`, which avoids starting
with the `18 mm` foot collision spheres embedded in the ground.

The earlier rear-only candidates remain useful for comparison:

```text
0.30 / -0.60
0.36 / -0.75
0.42 / -0.90
```

Run RR or RL alone with the other three legs fixed:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftTest-v0 \
  --num_envs 1 --duration 20 \
  --rear_leg RR --rear_thigh 0.3491 --rear_calf -0.7854 \
  --rear_lift_height 0.030 \
  --output logs/reference_debug/rear_lift_RR_level.csv

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftTest-v0 \
  --num_envs 1 --duration 20 \
  --rear_leg RL --rear_thigh 0.3491 --rear_calf -0.7854 \
  --rear_lift_height 0.030 \
  --output logs/reference_debug/rear_lift_RL_level.csv
```

Repeat with the other two paired candidates. This test has no policy, VMC,
delay, rate filter, acceleration filter, or torque target clip. CSV records
both world foot height and foot height transformed into the Trunk frame,
along with rear thigh/calf reference, command, position, error, and estimated
torque.

The free-base test uses a contact-force-gated state machine:

```text
DEFAULT_POSE -> PRE_SHIFT -> PRELOAD -> UNLOAD -> WAIT_FORCE_DROP -> LIFT
```

The measured RR unload naturally uses `FR+RL` as the support pair while
`FL+RR` unload together. Therefore RR applies the default `15/15 mm` support
push to `FR/RL`, not FL. RL mirrors this to the `FL+RR` support pair. The
default target-leg unload is `12 mm`. RR uses body shift
`x=+30 mm, y=+10 mm`; RL mirrors to `x=+30 mm, y=-10 mm`.
`LIFT` starts only after target-foot force remains below `3 N` for `0.2 s`.
If that does not happen within `1.0 s`, the state becomes `FAILED` with
`failure_reason=force_drop_timeout` and never enters `LIFT`.

Run the fixed-base diagnostic first:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftFixedBaseTest-v0 \
  --num_envs 1 --duration 12 --rear_leg RR \
  --output logs/reference_debug/rear_lift_fixed_RR.csv

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftFixedBaseTest-v0 \
  --num_envs 1 --duration 12 --rear_leg RL \
  --output logs/reference_debug/rear_lift_fixed_RL.csv
```

Then run the same RR/RL commands with `SmallHighFreq-RearLiftTest-v0` for the
free-base comparison. The CSV additionally records base height/RPY, foot
contact state, normal force, world/body foot height, per-leg support preload,
and target-leg unload. Fixed-base world foot lift validates IK and semantics;
free-base swing normal force near zero and at least `10 mm` world clearance
validate support transfer.

Before tuning that state machine, measure each leg's foot-z press sign:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-PressSignTest-v0 \
  --num_envs 1 --duration 17 \
  --output logs/reference_debug/press_sign_test.csv
```

This applies `+10 mm` and `-10 mm` to each leg independently and prints
`normal_force_before`, `normal_force_after`, and the inferred downward press
sign. Pass the reported `FR,FL,RR,RL` signs to later tests with
`--foot_down_signs=-1,-1,-1,-1`.

Sweep body shift for RR and RL separately:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-BodyShiftSweep-v0 \
  --rear_leg RR --num_envs 1 --duration 39 \
  --output logs/reference_debug/body_shift_sweep_RR.csv

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-BodyShiftSweep-v0 \
  --rear_leg RL --num_envs 1 --duration 39 \
  --output logs/reference_debug/body_shift_sweep_RL.csv
```

The sweep covers `body_shift_x/y = -30...+30 mm`. The summary first rejects
poses above 3 degrees roll/pitch, then minimizes target rear-foot force while
increasing diagonal front-foot force.

After sign and shift direction are confirmed, sweep unload and support Kp:

```bash
for unload in 0.012 0.018 0.024 0.030; do
  for kp in mid high very_high; do
    ./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
      --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftTest-v0 \
      --rear_leg RR --num_envs 1 --duration 8 \
      --target_unload_z ${unload} \
      --main_support_push_z 0.015 \
      --front_support_push_z 0.015 \
      --rear_support_push_z 0.015 \
      --support_kp_level ${kp} \
      --output logs/reference_debug/rear_lift_RR_unload_${unload}_kp_${kp}.csv
  done
done
```

Use `--rear_leg RL` for the mirrored sweep. Kp levels are `mid=140/160`,
`high=180/200`, and `very_high=220/220` for support thigh/calf, with
`Kd=5.0`. These are simulation-only sweeps; they do not enable RL or VMC and
are not deployment gains.

Run the mirrored RL confirmation with the same successful RR parameters:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-RearLiftTest-v0 \
  --rear_leg RL --num_envs 1 --duration 8 \
  --target_unload_z 0.024 \
  --main_support_push_z 0.018 \
  --front_support_push_z 0.010 \
  --rear_support_push_z 0.010 \
  --support_kp_level high \
  --output logs/reference_debug/rear_lift_RL_024_high.csv
```

Expected mirror behavior is `actual_support_pair=FL+RR`, target RL force near
zero, and FR also tending to unload during the diagonal transition.

## Fast diagonal trot reference

`Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-Reference-v0`
is a reference-only diagonal gait. The default profile is conservative:

```text
Pair A = FR + RL
Pair B = FL + RR
step_hz = 1.10
duty_factor = 0.62
stride_length = 0.020 m
front_swing_height = 0.045 m
rear_swing_height = 0.065 m
support_preload_z = -0.008 m
warmup_sec = 2.0
swing_lift_peak_phase = 0.45
touchdown_phase = 0.82
early_stance_blend = 0.12
```

It does not use crawl rear-lift force gates. Pair A swing commands FR/RL with
swing trajectories while FL/RR receive pair-wise support preload; Pair B
mirrors this. Rear swing height is higher than front swing height so RR/RL can
clear the ground cleanly. The last 15% of the swing is touchdown/weight-transfer
time, so the swing foot returns to ground before the pair fully becomes support.
CSV includes `active_swing_pair`, `support_pair`, `actual_support_pair`, all
four foot normal forces, foot world heights, base roll/pitch, q errors,
estimated torques, and per-joint `kp_0..11` / `kd_0..11`.

Gain profiles:

```text
real_safe: swing 40/70/70, support 60/120/140, kd 4.2/5.0
mid:       swing 50/80/80, support 70/160/180, kd 4.5/5.0
high:      swing 50/80/80, support 70/180/200, kd 4.5/5.0
very_high: swing 50/80/80, support 70/220/220, kd 4.5/5.0
```

`mid` and above are simulation morphology/debug profiles. Do not treat them as
hardware defaults. Use `real_safe` for the first safety-chain check.

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-Reference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset conservative \
  --support_kp_level mid \
  --output logs/reference_debug/fast_diagonal_trot_conservative_v2.csv
```

The balanced profile is:

```text
step_hz = 1.15
duty_factor = 0.61
stride_length = 0.022 m
front_swing_height = 0.048 m
rear_swing_height = 0.067 m
support_preload_z = -0.009 m
```

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-Reference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid \
  --output logs/reference_debug/fast_diagonal_trot_balanced_v2.csv
```

The fast profile is for simulation stress testing only:

```text
step_hz = 1.20
duty_factor = 0.60
stride_length = 0.024 m
front_swing_height = 0.050 m
rear_swing_height = 0.070 m
support_preload_z = -0.010 m
```

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-Reference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset fast \
  --support_kp_level mid \
  --output logs/reference_debug/fast_diagonal_trot_fast_v2.csv
```

You can also sweep only preload with
`--fast_trot_support_preload_z 0.006`, `0.008`, or `0.010`.

### Fast diagonal trot safety chain

`FastDiagonalTrot-Reference-v0` intentionally bypasses the deploy target filter;
it is for gait shape and contact timing. Safety diagnostics use
`FastDiagonalTrot-SafeReference-v0` plus an explicit `--safety_profile`:

```text
monitor_only:      do not modify q_ref; only record risk metrics
performance_safe: keep balanced_v2_mid shape; only compress dangerous peaks
performance_soft_output: mid_soft gains plus no-overshoot rate limiting and 10/14/17 N.m backoff
performance_soft_output_v2: reduced preload, faster no-overshoot rate limit, slower stance Kp handoff, higher damping
real_safe:         conservative first hardware-proximity check
```

The torque-error math remains per-joint Kp aware, not the old scalar
`sim_kp=40` assumption:

```text
err_limit_joint = torque_budget_joint / kp_actual_joint
monitor_only:      compute only, do not apply
performance_safe: continuous warn 8 N.m, soft zone 12..17 N.m, hard 17 N.m
performance_soft_output: 8 N.m statistics, soft backoff starts at 10 N.m, stronger at 14 N.m, hard 17 N.m
performance_soft_output_v2: same torque backoff as v1, tuned to reduce bounce and ground-force spikes
real_safe:         strict conservative budget
```

The CSV includes `kp_actual_0..11`, `kd_actual_0..11`,
`torque_budget_0..11`, `err_limit_0..11`, raw/filtered/final torque estimates,
rate/accel/torque clip deltas, `torque_clip_mask_0..11`, `torque_clip_ratio`,
and raw/final `over_8/12/17nm` ratios.

`performance_soft_output` also records `q_ref_cmd_diff_0..11`,
`q_ref_cmd_diff_max`, `q_cmd_error_0..11`, `q_cmd_error_max`,
`q_ref_error_0..11`, and `q_ref_error_max`. Use `q_ref_cmd_diff` to judge
trajectory damage, and `q_cmd_error` plus `tau_est_cmd_final` to judge
execution risk.

The v2 profile additionally records and summarizes `force_sum`,
`contact_count`, `roll_abs`, `pitch_abs`, `yaw_abs`, preload gates, and
support preload deltas. Use these to check whether the robot is bouncing or
spiking ground reaction forces.

Monitor-only, preserving the original CPG target exactly:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid \
  --safety_profile monitor_only \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_monitor_only.csv
```

Performance-safe, for the future light-VMC base:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid \
  --safety_profile performance_safe \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_performance_safe.csv
```

Performance-soft-output, preserving the diagonal trot while smoothing the
command sent to the actuator model:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid_soft \
  --safety_profile performance_soft_output \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_soft_performance_soft_output.csv
```

Performance-soft-output v2, tuned to reduce bounce rather than chase the
lowest possible torque:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid_soft \
  --safety_profile performance_soft_output_v2 \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_soft_performance_soft_output_v2.csv
```

Baseline comparison for the soft-output run:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level mid \
  --safety_profile performance_safe \
  --output logs/reference_debug/fast_diagonal_trot_balanced_mid_performance_safe_baseline.csv
```

Real-safe, for later low-risk hardware bring-up:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-FastDiagonalTrot-SafeReference-v0 \
  --num_envs 1 --duration 20 \
  --trot_preset balanced \
  --support_kp_level real_safe \
  --safety_profile real_safe \
  --output logs/reference_debug/fast_diagonal_trot_balanced_real_safe.csv
```

If `real_safe` cannot keep the rear legs airborne, raise support Kp gradually;
do not jump directly to `160/180` as a hardware default.

The future training entry point is registered but is not run by reference
validation:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-v0 \
  --num_envs 64 --max_iterations 1000

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-v0 \
  --num_envs 1024 --max_iterations 9000 --headless
```

This task uses the existing 7.24 kg Fanfan asset. The actor outputs a 12-DoF
residual around `FanfanReferenceGait`; it never outputs absolute motor targets.
Play and Reference-only use a `0.15 m/s` command so the reference generator
runs the same full `0.038 m` stride and `0.62 Hz` cycle as the real-machine
big-stride node. Smooth output comes from the five-second gait warmup and
target-rate limiting; residual training additionally uses residual low-pass
and target-acceleration limiting.

Reference-only is a parity check for the real-machine gait node. It keeps the
five-second warmup and `2.1 rad/s` target-rate limit, but disables the
training-only torque-error and acceleration target shapers. IsaacLab's RS01
actuator model still applies the `17 N*m` hard torque limit. Its command is
fixed at `0.15 m/s`, residual and delay are zero, and domain randomization,
actor noise, and pushes are disabled.

## Dead-gait player

`ReferenceRaw-v0` is a direct IsaacLab dead-gait player. IsaacLab does not have
a button that synchronizes an external ROS2 gait automatically, so the custom
action term generates `q_ref` internally on every control step:

```text
zero_agent action = 0
FanfanReferenceGait -> policy_to_sim -> URDF joint-limit clamp -> position target
```

The zero action is intentionally ignored. This is reference playback, not PPO,
imitation learning, behavior cloning, Mimic, or demonstration collection.
ReferenceRaw uses the full `0.15 m/s / 0.62 Hz / 0.038 m / 0.072 m / 0.78`
wave gait with a one-second warmup. It disables rewards, residual processing,
all deployment filters, delay, randomization, noise, pushes, and automatic
fall termination. Joint-limit clipping prints the joint name, requested and
clamped values, and the URDF limits.

Run the raw player:

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceRaw-v0 \
  --num_envs 1
```

Check every semantic joint direction:

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-JointMappingDebug-v0 \
  --num_envs 1
```

JointMappingDebug holds each joint at `default + 0.1 rad` for one second, then
returns to default for one second before advancing through FR, FL, RR, and RL.
The older `JointMapping-v0` task remains an alias.

Replay a recorded real-machine trajectory:

```bash
FANFAN_CSV_PLAYBACK_PATH=/home/nszb/python_text/fanfan_gait.csv \
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-CsvPlayback-v0 \
  --num_envs 1
```

Or select the input and record detailed state through the dedicated runner:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-CsvPlayback-v0 \
  --csv_path /home/nszb/python_text/fanfan_gait.csv \
  --duration 60
```

CsvPlayback accepts wide `q_policy_0..11`, wide `q_real_0..11`, and the
long-form CSV emitted by `fanfan_big_stride_walk_node.py`. Real motor targets
are converted real-order -> policy semantic order -> simulator order. Frames
are linearly interpolated using the CSV time column and loop continuously.
Without an override it reads `logs/reference_debug/fanfan_gait_playback.csv`.

## Reference debug ladder

Do not start residual training until the deterministic reference ladder has
been checked in order. Every debug task uses one environment, `cmd_x=0.15`,
the full `0.038 m / 0.62 Hz / 0.072 m / 0.78 duty` gait, zero residual, nominal
motor parameters, and no noise, randomization, or pushes.

| Task suffix | Enabled layers |
| --- | --- |
| `ReferenceRaw` | Joint clamp only |
| `ReferenceRate` | Joint clamp + `2.1 rad/s` target-rate limit |
| `ReferenceTorqueMonitor` | Rate mode plus PD torque estimate, without torque clipping |
| `ReferenceTorqueClip` | Rate mode plus configurable target-error torque clip, default `10 N*m` |
| `ReferenceDelay` | TorqueClip mode plus configurable 0/1/2-step delay |
| `ReferenceFiltered` | Joint, rate, acceleration, torque, and delay chain |
| `JointMappingDebug` / `JointMapping` | One semantic joint at `+0.1 rad`, then a default-pose interval |
| `CsvPlayback` | Direct linearly interpolated CSV joint-target playback |

`Reference-v0` remains a compatibility alias for the `ReferenceRate` behavior.
The reference phase uses control dt (`0.02 s = 4 * 0.005 s`), so the nominal
phase increment is `0.0124` and the cycle time is about `1.6129 s`.

The dedicated recorder writes all intermediate targets, clipping ratios,
estimated torque, phase, masks, joint state, and base attitude to CSV:

```bash
./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceRaw-v0 \
  --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceRate-v0 \
  --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceTorqueMonitor-v0 \
  --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceTorqueClip-v0 \
  --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceDelay-v0 \
  --delay_steps 1 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-ReferenceFiltered-v0 \
  --delay_steps 0 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-JointMapping-v0 \
  --duration 26
```

CSV files default to `logs/reference_debug/<action_mode>.csv`. The same tasks
can be viewed without recording through `scripts/environments/zero_agent.py`.

Judge the layers in this order: stable default pose, 12-joint mapping,
ReferenceRaw leg order and continuous lift, phase cycle, CSV parity with the
ROS2 node, rate-limit distortion, torque demand, torque clipping, delay lag,
then the complete filter. Only after Raw, CSV playback, and Filtered are
acceptable should residual training resume.

## Big-stride curriculum

This is reference-gait plus residual learning, not pure RL locomotion. The
training distribution is deliberately centered near the real big-stride wave
gait instead of spending early training at `0.00-0.05 m/s`.

| Stage | Iterations | Command x | Standing | Randomization |
| --- | ---: | ---: | ---: | --- |
| 1 | 0-5k | 0.10-0.15 m/s | 18% | Off; nominal physics, no delay or actor noise |
| 2 | 5k-30k | 0.10-0.18 m/s | 10% | Light mass, joint, gain, and motor variation |
| 3 | 30k-60k | 0.12-0.20 m/s | 5% | Delay 0-2 and normal IMU/joint noise |
| 4 | 60k+ | 0.10-0.22 m/s | 5% | Stronger variation, delay 0-3, noise, and pushes |

At `0.10 m/s`, the reference is approximately `0.0253 m / 0.546 Hz /
0.058 m`. At `0.15 m/s`, it is exactly `0.038 m / 0.62 Hz / 0.072 m`.
Between `0.15` and `0.18 m/s`, stride and frequency increase smoothly to
`0.0456 m / 0.682 Hz`; commands above `0.18 m/s` do not enlarge the reference
further. The policy must correct the remaining speed error through its bounded
residual, not by changing reference phase, stride, or frequency.

TensorBoard logs the current iteration, stage, command range, standing ratio,
reference scales, mass/friction/motor ranges, delay, noise level, and push
state under `Curriculum/auto_speed`.

Residual limits remain fixed for every stage: hip `0.05 rad`, thigh `0.08 rad`,
and calf `0.10 rad`. Low-speed commands below `0.05 m/s` are reserved for
standing transitions or manual debugging, not the main training distribution.

## Linux validation

```bash
./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-Reference-v0 \
  --num_envs 1

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-v0 \
  --num_envs 64 --max_iterations 200

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-Play-v0 \
  --checkpoint /absolute/path/to/model.pt

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-v0 \
  --num_envs 1024 --max_iterations 90000 --headless
```

Start this curriculum from a new run. Checkpoints trained with the old
low-speed curriculum should not be resumed.

Pure Torch checks:

```bash
./isaaclab.sh -p tools/test_fanfan_reference_gait.py
```

## ONNX contract

For this task, `rsl_rl/play.py` automatically re-exports `policy.onnx` through
`ResidualOnnxWrapper` and writes `fanfan_residual_contract.json`. The ONNX
output is the scaled residual in radians. Deployment must generate the same `q_ref`, apply
the residual low-pass filter, add `q_ref + residual`, and then run the same
joint, rate, and torque safety filters. Sending ONNX output directly to motors
is explicitly unsupported.
