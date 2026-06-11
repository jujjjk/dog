"""Fanfan 12-DOF quadruped symmetry helpers for RSL-RL.

第一版只做工程上最稳的左右镜像 data augmentation，不实现 equivariant network。
所有关节 index/sign 都集中放在文件顶部，后续如果真实关节顺序不同，只需要改这些常量。
"""

from __future__ import annotations

from collections.abc import MutableMapping

import torch

try:
    from tensordict import TensorDict
except ImportError:  # pragma: no cover - IsaacLab/RSL-RL 环境通常会安装 tensordict。
    TensorDict = None


# 训练动作/观测默认关节顺序：
# [FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
#  RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]
JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# Sagittal-plane mirror: FR <-> FL, RR <-> RL。
JOINT_MIRROR_INDEX = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
JOINT_MIRROR_SIGN = (-1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0)

# 当前 action 是 12 维 joint position residual，默认与关节顺序一致。
ACTION_MIRROR_INDEX = JOINT_MIRROR_INDEX
ACTION_MIRROR_SIGN = JOINT_MIRROR_SIGN

# 拼接后的 policy observation 默认布局：
# base_lin_vel(3), base_ang_vel(3), projected_gravity(3), command(3),
# joint_pos(12), joint_vel(12), [height_scan(187)], last_action(12)。
BASE_LIN_VEL_SLICE = slice(0, 3)
BASE_ANG_VEL_SLICE = slice(3, 6)
PROJECTED_GRAVITY_SLICE = slice(6, 9)
COMMAND_SLICE = slice(9, 12)
JOINT_POS_SLICE = slice(12, 24)
JOINT_VEL_SLICE = slice(24, 36)
LAST_ACTION_SLICE_NO_HEIGHT = slice(36, 48)
HEIGHT_SCAN_SLICE_WITH_ACTION = slice(36, 223)
LAST_ACTION_SLICE_WITH_HEIGHT = slice(223, 235)

BASE_LIN_VEL_SIGN = (1.0, -1.0, 1.0)
BASE_ANG_VEL_SIGN = (-1.0, 1.0, -1.0)
PROJECTED_GRAVITY_SIGN = (1.0, -1.0, 1.0)
COMMAND_SIGN = (1.0, -1.0, -1.0)

POLICY_KEY = "policy"
DUPLICATE_BATCH_FOR_RSL_RL = True
"""RSL-RL 训练时返回 original + mirrored batch；debug 时使用纯 mirror 检查 involution。"""


def _sign_tensor(values: tuple[float, ...], ref: torch.Tensor) -> torch.Tensor:
    """按参考 tensor 的 device/dtype 创建符号 tensor。"""
    return torch.tensor(values, device=ref.device, dtype=ref.dtype)


def _mirror_vector(data: torch.Tensor, index: tuple[int, ...], sign: tuple[float, ...]) -> torch.Tensor:
    """镜像一段关节或 action 向量，保持 dtype/device 不变。"""
    sign_tensor = _sign_tensor(sign, data)
    return data[..., list(index)] * sign_tensor


def _mirror_policy_tensor(policy_obs: torch.Tensor) -> torch.Tensor:
    """镜像已经 concatenate 的 policy observation。"""
    mirrored = policy_obs.clone()

    mirrored[..., BASE_LIN_VEL_SLICE] = policy_obs[..., BASE_LIN_VEL_SLICE] * _sign_tensor(
        BASE_LIN_VEL_SIGN, policy_obs
    )
    mirrored[..., BASE_ANG_VEL_SLICE] = policy_obs[..., BASE_ANG_VEL_SLICE] * _sign_tensor(
        BASE_ANG_VEL_SIGN, policy_obs
    )
    mirrored[..., PROJECTED_GRAVITY_SLICE] = policy_obs[..., PROJECTED_GRAVITY_SLICE] * _sign_tensor(
        PROJECTED_GRAVITY_SIGN, policy_obs
    )
    mirrored[..., COMMAND_SLICE] = policy_obs[..., COMMAND_SLICE] * _sign_tensor(COMMAND_SIGN, policy_obs)
    mirrored[..., JOINT_POS_SLICE] = _mirror_vector(policy_obs[..., JOINT_POS_SLICE], JOINT_MIRROR_INDEX, JOINT_MIRROR_SIGN)
    mirrored[..., JOINT_VEL_SLICE] = _mirror_vector(policy_obs[..., JOINT_VEL_SLICE], JOINT_MIRROR_INDEX, JOINT_MIRROR_SIGN)

    obs_dim = policy_obs.shape[-1]
    if obs_dim >= LAST_ACTION_SLICE_WITH_HEIGHT.stop:
        # Rough env 可能包含 height scan。左右镜像下横向网格翻转；如果以后 height scan 尺寸变化，改这里即可。
        mirrored[..., HEIGHT_SCAN_SLICE_WITH_ACTION] = (
            policy_obs[..., HEIGHT_SCAN_SLICE_WITH_ACTION].view(*policy_obs.shape[:-1], 11, 17).flip(dims=[-2]).reshape(
                *policy_obs.shape[:-1], -1
            )
        )
        mirrored[..., LAST_ACTION_SLICE_WITH_HEIGHT] = _mirror_vector(
            policy_obs[..., LAST_ACTION_SLICE_WITH_HEIGHT], ACTION_MIRROR_INDEX, ACTION_MIRROR_SIGN
        )
    elif obs_dim >= LAST_ACTION_SLICE_NO_HEIGHT.stop:
        mirrored[..., LAST_ACTION_SLICE_NO_HEIGHT] = _mirror_vector(
            policy_obs[..., LAST_ACTION_SLICE_NO_HEIGHT], ACTION_MIRROR_INDEX, ACTION_MIRROR_SIGN
        )

    return mirrored


def _mirror_obs_dict(obs):
    """镜像 TensorDict 或普通 dict，尽量同时支持拼接向量和未拼接 key。"""
    mirrored = obs.clone() if hasattr(obs, "clone") else dict(obs)

    if POLICY_KEY in obs and isinstance(obs[POLICY_KEY], torch.Tensor):
        mirrored[POLICY_KEY] = _mirror_policy_tensor(obs[POLICY_KEY])

    # 如果未来关闭 concatenate_terms，RSL-RL 传入未拼接 dict，这些 key 也能工作。
    key_signs = {
        "base_lin_vel": BASE_LIN_VEL_SIGN,
        "base_ang_vel": BASE_ANG_VEL_SIGN,
        "projected_gravity": PROJECTED_GRAVITY_SIGN,
        "velocity_commands": COMMAND_SIGN,
        "commands": COMMAND_SIGN,
    }
    for key, sign in key_signs.items():
        if key in obs and isinstance(obs[key], torch.Tensor):
            mirrored[key] = obs[key] * _sign_tensor(sign, obs[key])

    for key in ("joint_pos", "joint_vel"):
        if key in obs and isinstance(obs[key], torch.Tensor):
            mirrored[key] = _mirror_vector(obs[key], JOINT_MIRROR_INDEX, JOINT_MIRROR_SIGN)

    for key in ("actions", "last_action"):
        if key in obs and isinstance(obs[key], torch.Tensor):
            mirrored[key] = _mirror_vector(obs[key], ACTION_MIRROR_INDEX, ACTION_MIRROR_SIGN)

    return mirrored


@torch.no_grad()
def fanfan_symmetry_data_augmentation(
    env=None,
    obs=None,
    actions: torch.Tensor | None = None,
    action: torch.Tensor | None = None,
):
    """RslRlSymmetryCfg 使用的左右镜像 data augmentation function。

    Args:
        env: RSL-RL VecEnv wrapper 或 IsaacLab env。训练时非 None，会返回 original + mirror。
        obs: TensorDict、普通 dict 或 None。
        actions/action: shape 为 ``[N, 12]`` 的 action tensor，或 None。
            RSL-RL 当前版本传入关键字 ``actions``；保留 ``action`` 是为了兼容手动调试。

    Returns:
        训练时返回 duplicated_obs/duplicated_action；debug 时 env=None 返回 mirrored_obs/mirrored_action。
        若输入为 None，对应输出也为 None。
    """
    if actions is None:
        actions = action

    mirrored_obs = None
    if obs is not None:
        if TensorDict is not None and isinstance(obs, TensorDict):
            mirrored_obs = _mirror_obs_dict(obs)
        elif isinstance(obs, MutableMapping):
            mirrored_obs = _mirror_obs_dict(obs)
        elif isinstance(obs, torch.Tensor):
            mirrored_obs = _mirror_policy_tensor(obs)
        else:
            raise TypeError(f"Unsupported obs type for symmetry augmentation: {type(obs)!r}")

    mirrored_action = None
    if actions is not None:
        mirrored_action = _mirror_vector(actions, ACTION_MIRROR_INDEX, ACTION_MIRROR_SIGN)

    if env is not None and DUPLICATE_BATCH_FOR_RSL_RL:
        return _concat_augmented(obs, mirrored_obs), _concat_augmented(actions, mirrored_action)

    return mirrored_obs, mirrored_action


def _concat_augmented(original, mirrored):
    """把 original 和 mirrored 沿 batch 维拼接，用于 RSL-RL data duplication。"""
    if original is None:
        return None
    if isinstance(original, torch.Tensor):
        return torch.cat((original, mirrored), dim=0)
    if TensorDict is not None and isinstance(original, TensorDict):
        return torch.cat((original, mirrored), dim=0)
    if isinstance(original, MutableMapping):
        return {key: _concat_augmented(original[key], mirrored[key]) for key in original.keys()}
    raise TypeError(f"Unsupported type for symmetry batch duplication: {type(original)!r}")


def _max_abs_error(a, b) -> float:
    """递归计算两份 obs/action 的最大绝对误差。"""
    if a is None and b is None:
        return 0.0
    if isinstance(a, torch.Tensor):
        return float((a - b).abs().max().item()) if a.numel() else 0.0
    if isinstance(a, MutableMapping) or (TensorDict is not None and isinstance(a, TensorDict)):
        err = 0.0
        for key in a.keys():
            err = max(err, _max_abs_error(a[key], b[key]))
        return err
    return 0.0


def check_mirror_involution(obs=None, action: torch.Tensor | None = None):
    """检查 mirror 两次是否恢复原样，并打印 obs/action 最大误差。"""
    mirrored_obs, mirrored_action = fanfan_symmetry_data_augmentation(None, obs, actions=action)
    restored_obs, restored_action = fanfan_symmetry_data_augmentation(None, mirrored_obs, actions=mirrored_action)

    obs_error = _max_abs_error(obs, restored_obs)
    action_error = _max_abs_error(action, restored_action)
    print(f"[fanfan symmetry] mirror(mirror(obs)) max error: {obs_error:.6g}")
    print(f"[fanfan symmetry] mirror(mirror(action)) max error: {action_error:.6g}")
    return obs_error, action_error
