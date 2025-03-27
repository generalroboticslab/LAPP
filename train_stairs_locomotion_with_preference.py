import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import distutils.version
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, task_registry_stairs_pref

import torch
from custom_env.llmpref_go2_robot_stairs import LLMPrefGo2RobotStairs
from custom_env.llmpref_go2_robot_stairs_config import LLMPrefGo2RobotStairsCfg, LLMPrefGo2RobotStairsCfgPPO

def main():
    # Create the environment configuration
    args = get_args()
    # args.headless = False

    env_cfg = LLMPrefGo2RobotStairsCfg()
    env_cfg.env.reward_module_name = args.reward_module_name

    if args.terrain == 'pyramid_stairs':
        # Modify terrain configuration to use pyramid stairs terrain
        env_cfg.terrain.mesh_type = 'trimesh'  # Use a terrain mesh
        env_cfg.terrain.selected = True  # Select a specific terrain type
        env_cfg.terrain.curriculum = False  # Disable curriculum to use only the selected terrain
        env_cfg.terrain.terrain_type = 'pyramid_stairs'  # Set terrain type
        env_cfg.terrain.num_rows = 35  # 35
        env_cfg.terrain.num_cols = 5  # 35, 10
        env_cfg.terrain.slope_threshold = 0.01  # make it very small, so the stairs are straight without slope
        env_cfg.env.init_difficulty_level = args.init_difficulty_level
        # env_cfg.terrain.terrain_kwargs = {
        #     'step_width': 0.31,  # Set the step width [meters]  0.5
        #     'step_height': 0.00,  # Set the step height [meters] 0.15
        #     'platform_size': 3.0  # Set the platform size [meters]  1.0, 1.5
        # }

    # Create the environment
    task_registry_stairs_pref.register(args.task, LLMPrefGo2RobotStairs, env_cfg, LLMPrefGo2RobotStairsCfgPPO)
    env, env_cfg = task_registry_stairs_pref.make_env(name=args.task, args=args)

    # Proceed with training
    ppo_runner, train_cfg = task_registry_stairs_pref.make_alg_runner(
        env=env, name=args.task, args=args, log_root=args.log_root
    )

    ppo_runner.curriculum_learn_train_pref_predictor(
        num_learning_iterations=train_cfg.runner.max_iterations, main_args=args, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()