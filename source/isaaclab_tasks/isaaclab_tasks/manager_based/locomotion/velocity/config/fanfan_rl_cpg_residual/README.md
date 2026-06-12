# Fanfan Wave Gait Residual

This task uses the existing 7.24 kg Fanfan asset. The actor outputs a 12-DoF
residual around `FanfanReferenceGait`; it never outputs absolute motor targets.
Play and Reference-only use a `0.15 m/s` command so the reference generator
runs the same full `0.038 m` stride and `0.62 Hz` cycle as the real-machine
big-stride node. Smooth output comes from the five-second gait warmup,
residual low-pass filter, and target acceleration/rate limits.

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
