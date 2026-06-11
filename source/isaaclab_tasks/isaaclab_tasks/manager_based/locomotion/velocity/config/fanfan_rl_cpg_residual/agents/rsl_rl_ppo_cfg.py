from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from isaaclab_tasks.manager_based.locomotion.velocity.config.fanfan_a1_clean.agents.rsl_rl_ppo_cfg import (
    FanfanA1CleanFlatPPORunnerCfg,
)


@configclass
class FanfanRlCpgResidualPPORunnerCfg(FanfanA1CleanFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.max_iterations = 90_000
        self.experiment_name = "fanfan_rl_cpg_residual_flat"
        self.run_name = "wave_reference_residual_7p24kg"
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.actor = RslRlMLPModelCfg(
            hidden_dims=[256, 256, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.08, std_type="log"),
        )
        self.critic = RslRlMLPModelCfg(
            hidden_dims=[256, 256, 128], activation="elu", obs_normalization=False
        )
        self.algorithm.symmetry_cfg = None
