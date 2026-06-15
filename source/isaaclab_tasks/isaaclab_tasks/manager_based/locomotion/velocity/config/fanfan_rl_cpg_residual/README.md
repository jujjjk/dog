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
keeps a `5 mm` radial workspace margin and caps generated reference changes at
`10 rad/s`, below the RS01 rated-speed value of about `10.47 rad/s`. A 36 V supply gives more speed-response
headroom than 24 V, but does not change the normal `6 N*m` reference budget,
the URDF limits, or the `17 N*m` actuator hard cap.

The URDF-based trajectory audit measures about `11.8 rad/s` without this
reference slew cap. A full `62-67 mm` lift at `0.82 Hz` is therefore not
compatible with a raw `2.1 rad/s` joint-target ceiling from the selected
near-extended stand pose. Stage 1 deliberately keeps the requested
`2.1 rad/s` deployment limiter and logs its clipping ratio; frequent clipping
means swing height/frequency must be reduced or the deployment rate budget
must be revalidated before hardware use.

### Reference stages

All stages ignore the zero-agent action, use fixed `cmd_x=0.15`, and disable
policy loading, residuals, delay, randomization, observation noise, pushes,
rewards, curriculum and automatic fall reset.

| Stage | Control path |
| --- | --- |
| 0 | CPG/IK -> policy-to-sim -> URDF joint clamp |
| 1 | Stage 0 + `2.1 rad/s` rate + `100 rad/s^2` acceleration + `6 N*m` target-error limits |
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
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage1-Reference-v0 \
  --num_envs 1 --duration 60

./isaaclab.sh -p scripts/environments/fanfan_reference_debug.py \
  --task Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-Stage2-Reference-v0 \
  --num_envs 1 --duration 60
```

`SmallHighFreq-Reference-v0` is a Stage-1 alias. CSV output defaults to a
task-specific file under `logs/reference_debug/`, so Stage 0/1/2 runs do not
overwrite one another. It includes stage, gait parameters,
swing/stance masks, `q_cpg`, `q_vmc_delta`, `q_ref`, every safety-chain target,
`q_final`, `q_actual`, errors, attitude, angular velocity, estimated torque,
predicted foot lift, actual world foot height, and every clamp mask.

Do not start residual learning until Stage 0 has correct leg order, continuous
lift/touchdown and stable tripod support, Stage 1 tracks without persistent
clipping or torque above the normal budget, and Stage 2 only makes small
posture improvements. If Stage 0 is unstable, tune stride, swing height,
duty factor, support preload and stand pose before touching VMC.

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
