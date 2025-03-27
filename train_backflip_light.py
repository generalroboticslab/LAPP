import os
import hydra
from omegaconf import DictConfig, OmegaConf
import sys

import distutils.version
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, task_registry

import torch
from custom_env.llmpref_go2_light_backflip_robot import LLMPrefGo2LightBackflipRobot
from custom_env.llmpref_go2_light_backflip_robot_config import LLMPrefGo2LightBackflipRobotCfg, LLMPref2Go2LightBackflipRobotCfgPPO

def main():
    # Create the environment configuration
    args = get_args()
    # args.headless = False

    env_cfg = LLMPrefGo2LightBackflipRobotCfg()
    env_cfg.env.reward_module_name = args.reward_module_name

    # in the back flip task, we set the velocity commands to 0. Other velocities are already 0.
    env_cfg.commands.ranges.lin_vel_x = [-0.0, 0.0]

    # set the angular range of the initialization state in the air
    env_cfg.domain_rand.random_in_air = args.random_in_air
    print(f"the random_in_air parameter is: {env_cfg.domain_rand.random_in_air}")

    env_cfg.domain_rand.angle_initialization_range = [args.init_angle_low, args.init_angle_high]
    print(f"the range of the random angle initialization range is {env_cfg.domain_rand.angle_initialization_range}")

    # set the height range of the initialization state in the air
    env_cfg.domain_rand.height_initialization_range = [args.init_height_low, args.init_height_high]
    print(f"the range of the random height initialization range is {env_cfg.domain_rand.height_initialization_range}")

    # set the num_steps_per_env to be larger
    env_ppo_cfg = LLMPref2Go2LightBackflipRobotCfgPPO
    env_ppo_cfg.runner.num_steps_per_env = args.num_steps_per_env  # 128 or 72

    # Create the environment
    # task_registry.register(args.task, LLMHydra2Go2JumpRobot, env_cfg, LLMHydra2Go2RobotCfgPPO)
    task_registry.register(args.task, LLMPrefGo2LightBackflipRobot, env_cfg, env_ppo_cfg)
    # env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env, env_cfg = task_registry.make_env(name=args.task, args=args)

    # Proceed with training
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, log_root=args.log_root
    )

    # ppo_runner.curriculum_learn(
    #     num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True
    # )

    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()