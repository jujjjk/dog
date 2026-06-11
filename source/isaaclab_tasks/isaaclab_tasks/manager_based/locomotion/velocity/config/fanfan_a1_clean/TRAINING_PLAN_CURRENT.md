# FanfanA1Clean 当前训练方案

更新时间：2026-05-24

本文档记录当前用于真机部署导向的 FanfanA1Clean IsaacLab / RSL-RL 训练方案。目标不是让仿真里跑得最快，而是训练出一个真机可执行、动作不过激、部署 safety 少介入的保守低速步态策略。

## 1. 当前真机问题结论

前几版 ONNX 在真机上无法正常行走，主要原因已经基本定位为：

- ONNX raw action 过大，真机日志里 `raw_action_abs_max` 经常到 5 甚至 10 以上。
- 部署端 `SafeTargetLimiter`、rate limit、accel limit、final torque safety 几乎每帧介入。
- 机器人通信、ONNX 推理、电机跟踪本身不是第一矛盾；velocity feedforward 后 `current/final` 已明显改善。
- 默认站姿已经修复为 `stand_pose_source=policy_default`，不再使用错误手写 stand pose。
- 当前训练目标是让 policy 在仿真阶段就学会小动作、慢速稳定、后腿不瘸、6 Nm 连续力矩预算内可执行。

## 2. 关键原则

- 不把 RS01 的 17 Nm 峰值力矩当长期可用能力。
- 训练主力预算围绕 6 Nm 连续额定负载，允许少量 10-14 Nm 短时峰值随机化。
- 不继续靠真机端放宽 torque/safety 解决问题。
- 不用 actuator-net MLP。
- 保留 `DeployFilteredJointPositionAction`，让训练端经过类似真机部署的目标滤波器。
- 保留自动速度课程，先学站稳和 0.05 m/s 附近低速，再逐步扩展。
- 导出 ONNX 前必须做 raw action gate，不允许 raw action 长期爆大。

## 3. PPO 配置

文件：

- `agents/rsl_rl_ppo_cfg.py`

当前核心配置：

```python
experiment_name = "fanfan_a1_clean_rough_std_safe"
run_name = "auto_speed_std_safe"
clip_actions = None
resume = False

actor.hidden_dims = [512, 256, 128]
critic.hidden_dims = [512, 256, 128]
actor.distribution_cfg.init_std = 0.10
actor.distribution_cfg.std_type = "log"

algorithm.clip_param = 0.10
algorithm.entropy_coef = 0.0
algorithm.num_learning_epochs = 2
algorithm.num_mini_batches = 8
algorithm.learning_rate = 5.0e-5
algorithm.desired_kl = 0.004
algorithm.max_grad_norm = 0.20
algorithm.symmetry_cfg = None
```

关键解释：

- `clip_actions=None` 是当前最重要的修复之一。之前 wrapper 把动作裁到 +/-1，环境奖励只能看到裁剪后的动作，看不到 PPO 高斯分布的 std 已经爆炸。现在不在 RSL-RL wrapper 裁动作，让 `action_l2` 能惩罚真实 raw sample。
- action term 仍然有 joint target clip 和 deploy-like filter，所以不是完全无保护。
- `entropy_coef=0.0` 防止 PPO 主动奖励扩大探索方差。
- `init_std=0.10` 用于硬件导向的保守探索。
- 当前健康参考：训练初期 `Mean action std` 应在 `0.1~0.8`，超过 `1.5` 要警惕，超过 `2.0` 直接停。

## 4. 自动速度课程

文件：

- `auto_curriculum.py`
- `rough_env_cfg.py`
- `flat_env_cfg.py`

当前自动课程：

| Stage | Iteration | lin_vel_x | lin_vel_y | ang_vel_z | rel_standing_envs |
| --- | ---: | --- | --- | --- | ---: |
| 1 | 0-10000 | 0.00-0.08 | 0.0 | 0.0 | 0.50 |
| 2 | 10000-30000 | 0.00-0.15 | 0.0 | 0.0 | 0.35 |
| 3 | 30000-60000 | 0.03-0.25 | 0.0 | 0.0 | 0.25 |
| 4 | 60000+ | 0.05-0.35 | 0.0 | 0.0 | 0.15 |

实现逻辑：

- `auto_speed_curriculum()` 根据 `env.common_step_counter // num_steps_per_iter` 推算当前 learner iteration。
- 每个 reset 周期更新 `base_velocity` command range。
- 返回日志字段：
  - `Curriculum/auto_speed/iteration`
  - `Curriculum/auto_speed/stage`
  - `Curriculum/auto_speed/lin_vel_x_min`
  - `Curriculum/auto_speed/lin_vel_x_max`
  - `Curriculum/auto_speed/rel_standing_envs`

Stage 1 的目的：

- cmd=0 能稳。
- 0.03-0.08 m/s 能小步起步。
- 后腿不乱甩。
- raw action 不长期大于 2。
- torque 不长期贴近 6 Nm。

## 5. Action Scale

文件：

- `rough_env_cfg.py`

当前 `DeployFilteredJointPositionActionCfg.scale` 和后续 `self.actions.joint_pos.scale` 必须保持一致：

```python
"FR_hip_joint": 0.12
"FL_hip_joint": 0.12
"RR_hip_joint": 0.12
"RL_hip_joint": 0.12

"FR_thigh_joint": 0.22
"FL_thigh_joint": 0.22
"RR_thigh_joint": 0.26
"RL_thigh_joint": 0.26

"FR_calf_joint": 0.30
"FL_calf_joint": 0.30
"RR_calf_joint": 0.42
"RL_calf_joint": 0.42
```

原因：

- 后腿默认 calf 更直，如果 rear calf 仍使用 0.30 rad，摆动相收腿空间不足。
- 但 raw action 必须通过 reward 和 PPO std 控住，不能靠无限 action scale 解决。

## 6. 训练端 Deploy-Like Target Filter

文件：

- `deploy_actions.py`
- `rough_env_cfg.py`

当前训练 action 处理路径：

1. policy 输出 raw action。
2. `JointPositionAction` 按 scale 和 default offset 得到 q raw target。
3. `DeployFilteredJointPositionAction` 近似真机部署滤波：
   - q target 根据 `torque_budget / kp` 限制 `q_target - q_current`。
   - 限制 target velocity。
   - 限制 target acceleration。
   - 加 1-3 帧 motor delay。
   - 再发给仿真 PD actuator。

当前参数范围：

```python
sim_target_rate_limit_range = (2.0, 3.0)
sim_target_accel_limit_range = (60.0, 120.0)
sim_torque_budget_range = (5.0, 10.0)
sim_short_peak_torque_range = (10.0, 14.0)
sim_short_peak_prob = 0.05
sim_motor_delay_steps_range = (1, 3)
sim_motor_strength_scale_range = (0.65, 1.05)
sim_kp = 40.0
sim_kp_scale_range = (0.8, 1.2)
sim_kd_scale_range = (0.7, 1.3)
```

按关节类型倍率：

```python
hip_err_limit_mul = 1.0
thigh_err_limit_mul = 1.2
calf_err_limit_mul = 1.4

hip_target_rate_mul = 1.0
thigh_target_rate_mul = 1.3
calf_target_rate_mul = 1.6

hip_target_accel_mul = 1.0
thigh_target_accel_mul = 1.3
calf_target_accel_mul = 1.6
```

目的：

- 让训练中 policy 提前适应真机部署端的 target limiter。
- 避免训练出真机必须大面积 safety 介入才能执行的动作。

## 7. Base Linear Velocity Observation

文件：

- `mdp_observations.py`
- `rough_env_cfg.py`

当前 observation 中的 base linear velocity 使用 `base_lin_vel_deploy_corrupted()`，模拟真机 `base_lin_vel_source=zero/command/estimator` 不可靠的问题。

当前随机化：

```python
zero_prob = 0.15
command_prob = 0.20
noise_std = (0.08, 0.05, 0.02)
bias_range = (-0.05, 0.05)
delay_steps_range = (0, 3)
scale_range = (0.7, 1.2)
```

目的：

- 不让 policy 强依赖仿真完美 base velocity。
- 让真机上使用 `base_lin_vel_source=command` 或 estimator 时不至于崩。

## 8. 步态与后腿奖励

文件：

- `rough_env_cfg.py`
- `flat_env_cfg.py`
- `mdp_rewards.py`

当前主要步态项：

- `phase_trot_foot_clearance`
- `phase_trot_swing_contact`
- `phase_trot_contact_pattern`
- `phase_trot_calf_flexion`
- `swing_foot_clearance`
- `rear_swing_foot_clearance`
- `swing_calf_flexion`
- `contact_foot_drag`
- `rear_foot_drag`
- `long_contact`
- `rear_long_contact`

关键点：

- 这些项的 `command_threshold` 已统一降到 `0.03`。这是为了让 Stage 1 的 0.03-0.08 m/s 低速阶段也真正触发抬脚、拖地、相位奖励。
- `gait` old reward 权重为 `0.0`，避免老的 contact-time 同步奖励鼓励对称拖行。
- `phase_trot_calf_flexion` 和 `swing_calf_flexion` 的 calf target：

```python
target_calf_pos = (-0.95, -0.95, -0.72, -0.72)
```

顺序为：

```text
FR, FL, RR, RL
```

原因：

- 前腿默认 calf 约 -0.785，要求摆动相到 -0.95 合理。
- 后腿默认 calf 约 -0.349，直接要求后腿到 -0.95 过激，Stage-1B/1C 用 -0.72 作为可达目标。

## 9. 真机可执行性 Reward

当前核心约束：

```python
dof_torques_l2.weight = -2.0e-4
action_l2.weight = -0.008
action_rate_l2.weight = -0.025
dof_acc_l2.weight = -6.0e-6
track_lin_vel_xy_exp.weight = 0.85
track_ang_vel_z_exp.weight = 0.50
lin_vel_z_l2.weight = -12.0
ang_vel_xy_l2.weight = -1.5
base_height.weight = -60.0
```

意义：

- `action_l2` 限制 raw action 过大。
- `action_rate_l2` 限制动作跳变。
- `dof_acc_l2` 限制腿高速甩动。
- `dof_torques_l2` 和 RS01 torque 项限制靠大力矩硬顶。
- `track_lin_vel_xy_exp` 不设太高，避免为了追速度牺牲稳定性。

RS01 相关 torque reward：

```python
continuous_torque.weight = -0.02
low_speed_high_torque.weight = -0.01
power.weight = -0.01
applied_torque_limit.weight = -0.01
estimated_pd_torque_limit.weight = 0.0
```

说明：

- `estimated_pd_torque_limit` 当前保持关闭，先不让它和 deploy filter 同时强夹动作。
- 17 Nm 是 peak/hard cap，不是长期训练目标。

## 10. 默认站姿

文件：

- `fanfan_robot_cfg.py`

当前仿真默认站姿使用从真机 `/home/jetson/text` motion script 对齐后的姿态。

仿真 joint order：

```text
FR, FL, RR, RL
```

当前默认角：

```python
"FR_hip_joint": -0.1571
"FR_thigh_joint": 0.3491
"FR_calf_joint": -0.7854

"FL_hip_joint": 0.1571
"FL_thigh_joint": 0.3491
"FL_calf_joint": -0.7854

"RR_hip_joint": -0.1571
"RR_thigh_joint": 0.2269
"RR_calf_joint": -0.3491

"RL_hip_joint": 0.1571
"RL_thigh_joint": 0.2269
"RL_calf_joint": -0.3491
```

真机部署必须继续使用：

```bash
stand_pose_source:=policy_default
enable_rear_leg_posture_bias:=false
```

## 11. Domain Randomization

当前保守随机化：

```python
base_mass.mass_distribution_params = (-0.3, 0.8)
base_com.x = (-0.02, 0.02)
base_com.y = (-0.01, 0.01)
base_com.z = (-0.01, 0.01)

RS01_STIFFNESS_SCALE_RANGE = (0.8, 1.2)
RS01_DAMPING_SCALE_RANGE = (0.7, 1.3)
RS01_JOINT_FRICTION_RANGE = (0.03, 0.15)
RS01_ARMATURE_RANGE = (0.005, 0.02)
RS01_MOTOR_STRENGTH_SCALE_RANGE = (0.65, 1.05)
```

目的：

- 模拟电池、线束、外壳、安装误差。
- 模拟真机压角度和执行器偏弱。
- 不做过猛随机化，避免训练第一阶段还没学会站稳就被随机化打崩。

## 12. 当前健康训练指标

最近一次健康趋势：

- `Mean action std = 0.10`，正常。
- `Mean entropy loss` 为负，正常。
- `Mean episode length = 1000`，无明显摔倒。
- `Episode_Termination/base_contact = 0`
- `Episode_Termination/low_base = 0`
- `Episode_Termination/bad_orientation = 0`
- Stage 1：
  - `lin_vel_x_min = 0.0`
  - `lin_vel_x_max = 0.08`
  - `rel_standing_envs = 0.50`
- `phase_trot_*`、`swing_foot_clearance`、`rear_swing_foot_clearance` 已经开始非零，说明低速步态奖励正在生效。

需要停训的指标：

- `Mean action std > 1.5`：警惕。
- `Mean action std > 2.0`：停。
- value loss 变成 `inf/nan`：停。
- reward 突然大幅崩掉，同时 terminations 上升：停。

## 13. 推荐训练命令

从头训练，不 resume 旧的炸 std run：

```bash
cd /home/nszb/python_text/lsaacGym/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanA1Clean-v0 \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless \
  agent.resume=False
```

如果从某个健康 checkpoint 继续，必须确认该 checkpoint 的：

- `Mean action std < 1.0`
- 没有 NaN/inf
- 没有从旧 `clip_actions=1.0` 和高 std run 继承 optimizer 状态

否则不要 resume。

## 14. Play 检查顺序

不要直接测 0.35。按顺序检查：

1. `cmd_x=0.00`：站立，动作应该小。
2. `cmd_x=0.03`：小步起步。
3. `cmd_x=0.05`：低速。
4. `cmd_x=0.08`：Stage 1 上限。
5. `cmd_x=0.10`
6. `cmd_x=0.15`
7. `cmd_x=0.25`
8. `cmd_x=0.35`

每个速度检查：

- 不明显跳。
- 不后退。
- 不交叉腿。
- 后腿不偏瘸。
- 小腿不靠大幅硬顶身体。
- 不出现长期四脚拖地。

## 15. ONNX 导出前检查

不要默认最后一个 checkpoint 最好。优先选：

- cmd=0 能稳。
- 0.03/0.05 能起步。
- 0.10/0.15 能慢走。
- action std 低。
- raw action 不爆。
- 后腿不瘸。
- safety-like filter 下动作不被严重改形。

导出 ONNX 后用：

```bash
python3 source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_a1_clean/tools/verify_policy_onnx.py \
  /path/to/policy.onnx
```

如果有真机 CSV，可回放 deployment-like obs：

```bash
python3 source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_a1_clean/tools/verify_policy_onnx.py \
  /path/to/policy.onnx \
  --csv /tmp/mydog_policy_debug.csv \
  --csv-limit 100
```

## 16. 真机 CSV Gate

真机悬空或支撑架测试后，用：

```bash
python3 source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_a1_clean/tools/check_policy_csv_gate.py \
  /tmp/mydog_policy_debug.csv \
  --skip-sec 1.0 \
  --moving-cmd-threshold 0.03
```

通过标准：

- cmd=0：
  - `raw_p90 < 2.0`
  - `raw>3` 很少
  - `final_mean <= 1-2/12`
- cmd>0：
  - `raw_p90 < 3.0`
  - `raw>5` 很少
  - `rate_mean` 不长期超过 6/12
  - `final_mean <= 2/12`
  - `tau_p90` 不长期贴近 6 Nm

如果 gate 输出：

```text
DO NOT walk on the real robot yet
```

就不要下地。

## 17. 真机测试参数建议

新 ONNX 上真机，先悬空或支撑架：

```bash
ros2 launch mydog_policy policy_walk.launch.py \
  enable_send:=true \
  startup_stand_first:=true \
  stand_pose_source:=policy_default \
  debug_print_arrays:=false \
  debug_csv_path:=/tmp/mydog_policy_debug_new.csv \
  debug_csv_period_sec:=0.02 \
  torque_safety_budget_nm:=6.0 \
  max_target_rate_rad_s:=2.0 \
  max_target_accel_rad_s2:=60.0 \
  err_limit_safety_factor:=1.0 \
  enable_velocity_ff:=true \
  velocity_ff_scale:=0.3 \
  max_motor_vel_cmd_rad_s:=8.0 \
  enable_rear_leg_posture_bias:=false
```

速度递增：

```text
0.03 -> 0.05 -> 0.08 -> 0.10 -> 0.12 -> 0.15
```

不要一上来 0.15 或 0.35。

真机必须观察：

- `[SEND] motion batch ok`
- `motor_error_code = 0`
- `faultSta = 0`
- `motor_online = 12/12`
- `rate_limited_count` 不长期 8-12/12
- `final_limited_count` 长期最好 <= 1-2/12
- `tau_est_max` 不长期贴近 6 Nm
- 电机温度不快速升高
- 后腿 calf/thigh 不长期 final_limited

## 18. 当前不要做的事

- 不要继续放宽真机 torque budget 到 10/17 Nm 当常规参数。
- 不要默认启用 rear bias。
- 不要只在 ROS2 端缩 action scale。
- 不要用已经 `Mean action std` 爆到 10 的 checkpoint resume。
- 不要在 Stage 1 还没稳定时追求 0.35 m/s。
- 不要把 old `gait` reward 打开到高权重。
- 不要打开 symmetry，当前前后腿默认姿态和 rear-specific scale 还不是严格镜像一致。

## 19. 最终目标

训练出的统一 policy 应满足：

- cmd=0 能稳。
- cmd=0.03/0.05 能低速起步。
- cmd=0.10/0.15 能稳定慢走。
- 后腿不偏瘸。
- 小腿能抬脚，不长期拖地。
- raw action 不爆。
- deploy-like target filter 不大幅改形。
- 真机 `SafeTargetLimiter` 和 final safety 只偶尔兜底。
- 6-8 Nm 预算下能执行，不依赖 17 Nm 峰值力矩。
