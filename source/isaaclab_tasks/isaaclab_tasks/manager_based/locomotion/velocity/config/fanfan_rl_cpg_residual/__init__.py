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
    id="Isaac-Velocity-Flat-FanfanRlCpgResidual-SmallHighFreq-v0",
    **_COMMON,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:FanfanRlCpgResidualSmallHighFreqEnvCfg"
        ),
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

_REFERENCE_DEBUG_TASKS = {
    "ReferenceRaw": "FanfanRlCpgResidualReferenceRawEnvCfg",
    "ReferenceRate": "FanfanRlCpgResidualReferenceRateEnvCfg",
    "ReferenceTorqueMonitor": "FanfanRlCpgResidualReferenceTorqueMonitorEnvCfg",
    "ReferenceTorqueClip": "FanfanRlCpgResidualReferenceTorqueClipEnvCfg",
    "ReferenceDelay": "FanfanRlCpgResidualReferenceDelayEnvCfg",
    "ReferenceFiltered": "FanfanRlCpgResidualReferenceFilteredEnvCfg",
    "JointMapping": "FanfanRlCpgResidualJointMappingEnvCfg",
    "JointMappingDebug": "FanfanRlCpgResidualJointMappingEnvCfg",
    "CsvPlayback": "FanfanRlCpgResidualCsvPlaybackEnvCfg",
}

for suffix, cfg_name in _REFERENCE_DEBUG_TASKS.items():
    gym.register(
        id=f"Isaac-Velocity-Flat-FanfanRlCpgResidual-{suffix}-v0",
        **_COMMON,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FanfanRlCpgResidualPPORunnerCfg",
        },
    )

_SMALL_HIGH_FREQ_REFERENCE_TASKS = {
    "SmallHighFreq-Reference": "FanfanRlCpgResidualSmallHighFreqReferenceEnvCfg",
    "SmallHighFreq-Stage0-Reference": "FanfanRlCpgResidualSmallHighFreqStage0ReferenceEnvCfg",
    "SmallHighFreq-Stage1-Reference": "FanfanRlCpgResidualSmallHighFreqStage1ReferenceEnvCfg",
    "SmallHighFreq-Stage1-Debug-Reference": "FanfanRlCpgResidualSmallHighFreqStage1DebugReferenceEnvCfg",
    "SmallHighFreq-Stage1-Safe-Reference": "FanfanRlCpgResidualSmallHighFreqStage1SafeReferenceEnvCfg",
    "SmallHighFreq-RearLiftTest": "FanfanRlCpgResidualRearLiftTestEnvCfg",
    "SmallHighFreq-RearLiftFixedBaseTest": "FanfanRlCpgResidualRearLiftFixedBaseTestEnvCfg",
    "SmallHighFreq-PressSignTest": "FanfanRlCpgResidualPressSignTestEnvCfg",
    "SmallHighFreq-BodyShiftSweep": "FanfanRlCpgResidualBodyShiftSweepEnvCfg",
    "FastDiagonalTrot-Reference": "FanfanRlCpgResidualFastDiagonalTrotReferenceEnvCfg",
    "FastDiagonalTrot-SafeReference": "FanfanRlCpgResidualFastDiagonalTrotSafeReferenceEnvCfg",
    "SmallHighFreq-Stage2-Reference": "FanfanRlCpgResidualSmallHighFreqStage2ReferenceEnvCfg",
}

for suffix, cfg_name in _SMALL_HIGH_FREQ_REFERENCE_TASKS.items():
    gym.register(
        id=f"Isaac-Velocity-Flat-FanfanRlCpgResidual-{suffix}-v0",
        **_COMMON,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.flat_env_cfg:{cfg_name}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:FanfanRlCpgResidualPPORunnerCfg",
        },
    )
