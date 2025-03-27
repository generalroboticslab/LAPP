import os

import numpy
import numpy as np
from datetime import datetime
import sys
import distutils.version
import isaacgym
from repo.unitree_rl_gym.legged_gym.envs import *
from repo.unitree_rl_gym.legged_gym.utils import get_args, export_policy_as_jit, task_registry, Logger
import torch
import cv2
from isaacgym import gymapi

from custom_env.llmpref_go2_backflip_robot import LLMPrefGo2BackflipRobot
from custom_env.llmpref_go2_backflip_robot_config import LLMPrefGo2BackflipRobotCfg, LLMPrefGo2BackflipRobotCfgPPO

# Create the environment
task_registry.register('go2_backflip', LLMPrefGo2BackflipRobot, LLMPrefGo2BackflipRobotCfg, LLMPrefGo2BackflipRobotCfgPPO)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.num_rows = 5
    # env_cfg.terrain.num_cols = 5
    # env_cfg.terrain.curriculum = False

    # PJ: set record to get device id
    env_cfg.viewer.record = args.record

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    env_cfg.commands.resampling_time = 1e6  # Don't resample command during the testing, so set it to a huge number.

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

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env_ids = torch.tensor([0], device=env.device)
    # env.reset_idx(env_ids)
    env.reset()

    # print(env.commands)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg, log_root=args.log_root)
    policy = ppo_runner.get_inference_policy(device=env.device)

    env.commands = torch.tensor([0.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

    # Create camera properties
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1280  # Set desired width
    camera_props.height = 720  # Set desired height

    # --- Fixed Camera Setup ---
    # Create camera sensor in the first environment
    fixed_camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    print(fixed_camera_handle)

    # Set camera position and target
    # record_cam_position = env_cfg.viewer.pos
    fixed_record_cam_position = env.base_pos[0].cpu().numpy() + numpy.array([5.0, -25.0, 7.0])
    # record_cam_lookat = env_cfg.viewer.lookat
    fixed_record_cam_lookat = env.base_pos[0].cpu().numpy() + numpy.array([5.0, 5.0, 0.0])
    fixed_camera_position = gymapi.Vec3(fixed_record_cam_position[0], fixed_record_cam_position[1], fixed_record_cam_position[2])
    fixed_camera_target = gymapi.Vec3(fixed_record_cam_lookat[0], fixed_record_cam_lookat[1], fixed_record_cam_lookat[2])
    env.gym.set_camera_location(fixed_camera_handle, env.envs[0], fixed_camera_position, fixed_camera_target)

    # Set up VideoWriter for fixed camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fixed = cv2.VideoWriter(args.log_root + '/' + args.load_run + '/backflip_normal' + '_model' + str(args.checkpoint) +'_fixed.mp4', fourcc, 60.0, (camera_props.width, camera_props.height))
    print("save fixed video to:")
    print(args.log_root + '/' + args.load_run + '/backflip_normal' + '_model' + str(args.checkpoint) + '_fixed.mp4')

    # --- Dynamic Camera Setup ---
    # Create dynamic camera sensor in the first environment
    dynamic_camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    print(dynamic_camera_handle)

    # Set initial dynamic camera position and target
    dynamic_cam_position = env.base_pos[0].cpu().numpy() + np.array([0.0, -2.0, 1.0])  # Initial offset
    dynamic_cam_lookat = env.base_pos[0].cpu().numpy()  # Look at the robot
    dynamic_camera_position = gymapi.Vec3(*dynamic_cam_position)  # PJ: set in a different way as the fixed one, but result is the same
    dynamic_camera_target = gymapi.Vec3(*dynamic_cam_lookat)
    env.gym.set_camera_location(dynamic_camera_handle, env.envs[0], dynamic_camera_position, dynamic_camera_target)

    # Set up VideoWriter for dynamic camera
    out_dynamic = cv2.VideoWriter(args.log_root + '/' + args.load_run + '/backflip_normal' + '_model' + str(args.checkpoint) + '_follow.mp4', fourcc, 60.0, (camera_props.width, camera_props.height))
    print("save dynamic video to:")
    print(args.log_root + '/' + args.load_run + '/backflip_normal' + '_model' + str(args.checkpoint) + '_follow.mp4')

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    env.max_episode_length = 400  # originally 1000
    print('max_episode_length: ', env.max_episode_length)
    for i in range(int(env.max_episode_length)):  # 10 * int(env.max_episode_length)
        print(f"the robot pos is: {env.base_pos[0].cpu().numpy()}")
        print(f"the robot rpy is: {env.rpy[0].cpu().numpy()}")
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # # stumbling penalty
        # stumble = (torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2) > 5.) * (torch.abs(env.contact_forces[:, env.feet_indices, 2]) < 1.)
        # stumble_reward = -5.0 * torch.sum(stumble, dim=1)
        #
        # print(f"stumble reward is: {stumble_reward}")

        # Update dynamic camera position and target to follow the robot
        dynamic_cam_position = env.base_pos[0].cpu().numpy() + np.array([0.0, -2.0, 1.0])  # Offset from robot
        dynamic_cam_lookat = env.base_pos[0].cpu().numpy()  # Look at the robot's current position
        dynamic_camera_position = gymapi.Vec3(*dynamic_cam_position)
        dynamic_camera_target = gymapi.Vec3(*dynamic_cam_lookat)
        env.gym.set_camera_location(dynamic_camera_handle, env.envs[0], dynamic_camera_position, dynamic_camera_target)

        # Render camera sensors
        env.gym.render_all_camera_sensors(env.sim)

        # --- Capture image from fixed camera ---
        image_buffer_fixed = env.gym.get_camera_image(env.sim, env.envs[0], fixed_camera_handle, gymapi.IMAGE_COLOR)

        # Convert image to numpy array and reshape
        image_data_fixed = np.frombuffer(image_buffer_fixed, dtype=np.uint8).reshape((camera_props.height, camera_props.width, 4))

        # Convert from RGBA to BGR format
        image_bgr_fixed = image_data_fixed[:, :, :3]

        # Write frame to fixed camera video
        out_fixed.write(image_bgr_fixed)

        # --- Capture image from dynamic camera ---
        image_buffer_dynamic = env.gym.get_camera_image(env.sim, env.envs[0], dynamic_camera_handle, gymapi.IMAGE_COLOR)

        # Convert image to numpy array and reshape
        image_data_dynamic = np.frombuffer(image_buffer_dynamic, dtype=np.uint8).reshape((camera_props.height, camera_props.width, 4))

        # Convert from RGBA to BGR format
        image_bgr_dynamic = image_data_dynamic[:, :, :3]

        # Write frame to dynamic camera video
        out_dynamic.write(image_bgr_dynamic)

        # *** Save right-camera images every 2 steps if enabled ***
        if args.save_right_camera_figures and (i % 2 == 0) and (i < 200):
            cv2.imwrite(os.path.join("results_fig/right_video_figs", args.video_figures_path, f"right_{i}.png"), image_bgr_dynamic)

    # Release the VideoWriter
    out_fixed.release()
    out_dynamic.release()


if __name__ == '__main__':
    EXPORT_POLICY = False  # oroginally true
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
