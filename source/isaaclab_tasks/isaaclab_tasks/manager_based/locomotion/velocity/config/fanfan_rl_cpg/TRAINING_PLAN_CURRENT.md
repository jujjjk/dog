# FanfanRlCpg 当前训练方案

更新时间：2026-05-25

本文档只对应新建的 `fanfan_rl_cpg` 分支。旧的 `Isaac-Velocity-Flat-FanfanA1Clean-v0` 是 pure RL / 参考配置，不是本分支训练命令。

## 1. 任务名

当前新分支注册在：

```text
Isaac-Velocity-Flat-FanfanRlCpg-v0
Isaac-Velocity-Flat-FanfanRlCpg-Play-v0
Isaac-Velocity-Flat-FanfanRlCpg-CPGOnly-v0
Isaac-Velocity-Rough-FanfanRlCpg-v0
Isaac-Velocity-Rough-FanfanRlCpg-Play-v0
Isaac-Velocity-Rough-FanfanRlCpg-CPGOnly-v0
```

## 2. 先生成 profile

从 IsaacLab 根目录运行：

```bash
cd /home/nszb/python_text/lsaacGym/IsaacLab

python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_urdf_for_cpg.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/extract_motor_specs.py
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/fit_motor_dynamics_from_real_data.py --csv /home/nszb/python_text/walk_as014_kp45_kd4_cmd012_basecmd_real.csv
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/generate_randomization_from_urdf.py
```

输出文件：

```text
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/config/motor_profile.yaml
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/config/motor_profile_fitted.yaml
source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/config/randomization_profile.yaml
```

## 3. CPG-only 检查

先看 CPG 相位和关节方向，不训练 residual。

注意：不要用 `rsl_rl/play.py` 跑 CPG-only 的第一次检查。`rsl_rl/play.py` 会先找训练 checkpoint；新分支还没训练时会报 `logs/rsl_rl/fanfan_rl_cpg_flat` 不存在。CPG-only action term 会忽略输入 action，所以用 IsaacLab 自带 zero agent 即可。

```bash
cd /home/nszb/python_text/lsaacGym/IsaacLab

./isaaclab.sh -p scripts/environments/zero_agent.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-CPGOnly-v0 \
  --num_envs 16
```

目标：

- `FR/RL` 同相；
- `FL/RR` 同相；
- 两组相差半周期；
- q_cpg 平滑、不过限；
- 低速站立时不明显摆腿。

## 4. CPG residual 训练命令

从头训练，不 resume 旧 pure RL checkpoint：

```bash
cd /home/nszb/python_text/lsaacGym/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-v0 \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless \
  agent.resume=False
```

如果要看 rough 版本：

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Velocity-Rough-FanfanRlCpg-v0 \
  --num_envs 1024 \
  --max_iterations 90000 \
  --headless \
  agent.resume=False
```

## 5. Play 检查

```bash
cd /home/nszb/python_text/lsaacGym/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Velocity-Flat-FanfanRlCpg-Play-v0 \
  --num_envs 50
```

速度检查顺序：

```text
cmd_x = 0.00 -> 0.03 -> 0.05 -> 0.08 -> 0.10 -> 0.12 -> 0.15
```

第一版不要追 0.35 m/s。

## 6. ONNX 和日志检查

导出 ONNX 后先做 policy 检查：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/verify_policy_onnx.py \
  --onnx /path/to/policy.onnx
```

真机或 rollout CSV 检查：

```bash
python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/analyze_cpg_policy_log.py \
  /path/to/policy_debug.csv \
  --skip-sec 1.0

python source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/fanfan_rl_cpg/tools/pre_real_deploy_check.py \
  --csv /path/to/policy_debug.csv
```

如果 `pre_real_deploy_check.py` 输出 `FAIL`，不要下地。

## 7. 下地前目标

- raw>|1| 总比例 `< 3%`
- raw>|0.8| 总比例 `< 10%`
- clip_count mean `< 0.3 joints/frame`
- 任意单腿 raw>|1| `< 10%`
- 四腿 residual RMS 最大值 / 最小值 `< 2`
- torque p95 `< safe_training_torque * 0.7`
- FR 不再长期显著高于其他腿
- q_cmd_delta 和 torque_rate 明显低于旧策略

一句话：这个分支训练的是 `CPG + 小 residual + 电机约束`，不要再用 `FanfanA1Clean` 的旧任务名启动训练。
