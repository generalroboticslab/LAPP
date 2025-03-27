import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import distutils.version
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, task_registry_obstacles_pref

import torch
# from custom_env.pure_go2_robot import PureGo2
# from custom_env.pure_go2_robot_config import PureGo2Cfg, PureGo2CfgPPO
from custom_env.llmpref_go2_robot_obstacles import LLMPrefGo2RobotObstacles
from custom_env.llmpref_go2_robot_obstacles_config import LLMPrefGo2ObstaclesCfg, LLMPrefGo2ObstaclesCfgPPO


def main():
    args = get_args()

    # env_cfg = PureGo2Cfg()
    env_cfg = LLMPrefGo2ObstaclesCfg()
    env_cfg.env.reward_module_name = args.reward_module_name

    env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.selected = True
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.terrain_type = 'curriculum_obs'
    env_cfg.terrain.num_rows = 30
    env_cfg.terrain.num_cols = 10

    train_cfg = LLMPrefGo2ObstaclesCfgPPO()
    train_cfg.runner.save_interval = args.save_interval

    task_registry_obstacles_pref.register(args.task, LLMPrefGo2RobotObstacles, env_cfg, train_cfg)

    env, env_cfg = task_registry_obstacles_pref.make_env(name=args.task, args=args)

    ppo_runner, train_cfg = task_registry_obstacles_pref.make_alg_runner(
        env=env, name=args.task, args=args, log_root=args.log_root, flag="pref"
    )

    ppo_runner.curr_learn_w_pred(
        num_learning_iterations=train_cfg.runner.max_iterations,
        main_args=args,
        init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()