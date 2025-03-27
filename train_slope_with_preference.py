import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import distutils.version
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, task_registry_slope_pref

import torch

from custom_env.llmpref_go2_robot_slope import LLMPrefGo2RobotSlope
from custom_env.llmpref_go2_robot_slope_config import LLMPrefGo2SlopeCfg, LLMPrefGo2SlopeCfgPPO


def main():
    args = get_args()

    env_cfg = LLMPrefGo2SlopeCfg()
    env_cfg.env.reward_module_name = args.reward_module_name

    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.selected = True
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_type = 'curriculum_slope'
    env_cfg.terrain.num_rows = 20
    env_cfg.terrain.num_cols = 5

    task_registry_slope_pref.register(args.task, LLMPrefGo2RobotSlope, env_cfg, LLMPrefGo2SlopeCfgPPO)

    env, env_cfg = task_registry_slope_pref.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry_slope_pref.make_alg_runner(
        env=env, name=args.task, args=args, log_root=args.log_root, flag="pref"
    )

    ppo_runner.curr_learn_w_pred(
        num_learning_iterations=train_cfg.runner.max_iterations,
        main_args=args,
        init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()