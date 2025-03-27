import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import distutils.version
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, task_registry_bounding_pref

import torch
from custom_env.llmpref_go2_robot_flat import LLMPrefGo2RobotFlat
from custom_env.llmpref_go2_robot_flat_config import LLMPrefGo2RobotFlatCfg, LLMPrefGo2RobotFlatCfgPPO

def main():
    # Create the environment configuration
    args = get_args()
    # args.headless = False

    env_cfg = LLMPrefGo2RobotFlatCfg()
    env_cfg.env.reward_module_name = args.reward_module_name

    # Create the environment
    task_registry_bounding_pref.register(args.task, LLMPrefGo2RobotFlat, env_cfg, LLMPrefGo2RobotFlatCfgPPO)
    # env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env, env_cfg = task_registry_bounding_pref.make_env(name=args.task, args=args)

    # Proceed with training
    ppo_runner, train_cfg = task_registry_bounding_pref.make_alg_runner(
        env=env, name=args.task, args=args, log_root=args.log_root
    )

    print("running ppo_runner.preference_learn_train_pref_predictor3")
    ppo_runner.preference_learn_train_pref_predictor3(
        num_learning_iterations=train_cfg.runner.max_iterations, main_args=args, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()