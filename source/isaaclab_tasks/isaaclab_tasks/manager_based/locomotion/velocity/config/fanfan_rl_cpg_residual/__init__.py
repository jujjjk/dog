import gymnasium as gym

from . import agents


_COMMON = {
    "entry_point": "isaaclab.envs:ManagerBasedRLEnv",
    "disable_env_checker": True,
}

gym.register(
    id="Isaac-Velocity-Flat-FanfanRlCpgResidual-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FanfanRlCpgResidualFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FanfanRlCpgResidualPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Flat-FanfanRlCpgResidual-Play-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FanfanRlCpgResidualFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FanfanRlCpgResidualPPORunnerCfg",
    },
)
gym.register(
    id="Isaac-Velocity-Flat-FanfanRlCpgResidual-Reference-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:FanfanRlCpgResidualReferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FanfanRlCpgResidualPPORunnerCfg",
    },
)
