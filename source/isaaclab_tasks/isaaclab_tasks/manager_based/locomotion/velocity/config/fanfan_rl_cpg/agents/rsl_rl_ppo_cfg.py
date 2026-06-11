# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


# Symmetry is intentionally disabled for the phase-gait training line.  The
# gait clock, rear/front default asymmetry, and rear-specific action scales are
# not yet mirror-consistent, so even a nominally zero-weight symmetry config can
# make debugging harder if a command-line override re-enables it.


@configclass
class FanfanA1CleanRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "fanfan_rl_cpg_rough"
    run_name = "cpg_residual_low_speed"
    # Do not clip in the RSL-RL wrapper.  The action term still clips final
    # joint targets, but leaving raw samples visible to the env lets action_l2
    # penalize an exploding Gaussian std instead of hiding it behind +/-1.
    clip_actions = None
    resume = False
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        # Hardware-oriented training should not rely on very large stochastic
        # samples: deployment uses deterministic ONNX output and the real robot
        # safety layer clips aggressive targets. Keep exploration modest.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.10,
            std_type="log",
        ),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        # Conservative PPO update for hardware-oriented walking.  The previous
        # setting could drive value loss/actor std to NaN after long runs.
        value_loss_coef=0.5,
        use_clipped_value_loss=True,
        clip_param=0.10,
        entropy_coef=0.0,
        num_learning_epochs=2,
        num_mini_batches=8,
        learning_rate=5.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=0.20,
        symmetry_cfg=None,
    )


@configclass
class FanfanA1CleanFlatPPORunnerCfg(FanfanA1CleanRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 300
        self.experiment_name = "fanfan_rl_cpg_flat"
        self.actor.hidden_dims = [128, 128, 128]
        self.critic.hidden_dims = [128, 128, 128]


# Backward-compatible aliases for the existing gym registration names and old
# launch/config references.  Keep these names until every machine has the new
# Fanfan-specific entry points.
UnitreeA1RoughPPORunnerCfg = FanfanA1CleanRoughPPORunnerCfg
UnitreeA1FlatPPORunnerCfg = FanfanA1CleanFlatPPORunnerCfg
FanfanRlCpgRoughPPORunnerCfg = FanfanA1CleanRoughPPORunnerCfg
FanfanRlCpgFlatPPORunnerCfg = FanfanA1CleanFlatPPORunnerCfg
