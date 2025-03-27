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

from scipy.spatial.transform import Rotation as R

from isaacgym import gymapi

from custom_env.llmpref_go2_robot_wave import LLMPrefGo2RobotWave
from custom_env.llmpref_go2_robot_wave_config import LLMPrefGo2WaveCfg, LLMPrefGo2WaveCfgPPO

task_registry.register('go2_terrain', LLMPrefGo2RobotWave, LLMPrefGo2WaveCfg, LLMPrefGo2WaveCfgPPO)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.reward_module_name = args.reward_module_name
    env_cfg.env.num_envs = 2
    env_cfg.viewer.record = args.record

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    env_cfg.env.test = True
    env_cfg.commands.resampling_time = 1e6  # Don't resample command during the testing, so set it to a huge number.

    env_cfg.terrain.curriculum = False
    if args.terrain == 'plane':
        env_cfg.terrain.mesh_type = 'plane'
        env_cfg.terrain.selected = False
        env_cfg.terrain.terrain_type = 'flat'
    else:
        env_cfg.terrain.mesh_type = 'trimesh'
        env_cfg.terrain.selected = True
        env_cfg.terrain.num_rows = 10
        env_cfg.terrain.num_cols = 5
        if args.terrain == 'pyramid_stairs':
            env_cfg.terrain.terrain_type = 'pyramid_stairs'
        elif args.terrain == 'pyramid_sloped':
            env_cfg.terrain.terrain_type = 'pyramid_sloped'
        elif args.terrain == 'discrete_obstacles':
            env_cfg.terrain.terrain_type = 'discrete_obstacles'
        elif args.terrain == 'wave':
            env_cfg.terrain.terrain_type = 'wave'
        else:
            print("@@@ Terrain type not supported.")

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    # env.difficulty_levels[0] = difficulty_level
    env.difficulty_levels[1] = 4
    env_ids = torch.tensor([1], device=env.device)
    env.reset_idx(env_ids)

    obs = env.get_observations()
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg,
                                                          log_root=args.log_root)
    policy = ppo_runner.get_inference_policy(device=env.device)

    if args.test_direct == "forward":
        env.commands = torch.tensor([1.5, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    elif args.test_direct == "backward":
        env.commands = torch.tensor([-0.9, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    elif args.test_direct == "left":
        env.commands = torch.tensor([0.0, 0.8, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    elif args.test_direct == "left_turn":
        env.commands = torch.tensor([0.7, 0.0, 0.2, 1.57], device=env.device).repeat(env.num_envs, 1)
    else:
        print("command wrong! Set to a very large forward speed")
        env.commands = torch.tensor([8.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)

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
    fixed_record_cam_position = env.base_pos[0].cpu().numpy() + numpy.array([5.0, -5.0, 5.0])
    # record_cam_lookat = env_cfg.viewer.lookat
    fixed_record_cam_lookat = env.base_pos[0].cpu().numpy() + numpy.array([5.0, 5.0, 0.0])
    fixed_camera_position = gymapi.Vec3(fixed_record_cam_position[0], fixed_record_cam_position[1],
                                        fixed_record_cam_position[2])
    fixed_camera_target = gymapi.Vec3(fixed_record_cam_lookat[0], fixed_record_cam_lookat[1],
                                      fixed_record_cam_lookat[2])
    env.gym.set_camera_location(fixed_camera_handle, env.envs[0], fixed_camera_position, fixed_camera_target)

    # Set up VideoWriter for fixed camera
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fixed = cv2.VideoWriter(
        args.log_root + '/' + args.load_run + '/' + args.test_direct + f'_fixed{args.checkpoint}.mp4', fourcc, 60.0,
        (camera_props.width, camera_props.height))
    print("save fixed video to:")
    print(args.log_root + '/' + args.load_run + '/' + args.test_direct + f'_fixed{args.checkpoint}.mp4')

    # --- Dynamic Camera Setup ---
    # Create dynamic camera sensor in the first environment
    dynamic_camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    print(dynamic_camera_handle)

    # Set initial dynamic camera position and target
    LOCAL_CAMERA_OFFSET = np.array([0.0, -2.0, 0.7])
    LOCAL_LOOKAT_OFFSET = np.array([0.0, 0.0, 0.0])
    dynamic_cam_position = env.base_pos[0].cpu().numpy() + LOCAL_CAMERA_OFFSET  # Initial offset
    dynamic_cam_lookat = env.base_pos[0].cpu().numpy()  # Look at the robot
    dynamic_camera_position = gymapi.Vec3(
        *dynamic_cam_position)  # PJ: set in a different way as the fixed one, but result is the same
    dynamic_camera_target = gymapi.Vec3(*dynamic_cam_lookat)
    env.gym.set_camera_location(dynamic_camera_handle, env.envs[0], dynamic_camera_position, dynamic_camera_target)

    # Set up VideoWriter for dynamic camera
    out_dynamic = cv2.VideoWriter(
        args.log_root + '/' + args.load_run + '/' + args.test_direct + f'_right{args.checkpoint}.mp4', fourcc, 60.0,
        (camera_props.width, camera_props.height))
    print("save dynamic video to:")
    print(args.log_root + '/' + args.load_run + '/' + args.test_direct + f'_right{args.checkpoint}.mp4')

    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    # env.max_episode_length = 600  # originally 1000
    env.max_episode_length = 400
    print('max_episode_length: ', env.max_episode_length)
    for i in range(int(env.max_episode_length)):  # 10 * int(env.max_episode_length)
        print(env.base_pos[0].cpu().numpy())

        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())

        # Read base pos/quaternion from env
        base_pos = env.base_pos[0].cpu().numpy()  # shape: (3,)
        base_quat = env.base_quat[0].cpu().numpy()  # shape: (4,) in [x, y, z, w] order

        # Convert to Euler
        r_full = R.from_quat(base_quat)  # if indeed [x, y, z, w]
        roll, pitch, yaw = r_full.as_euler('xyz', degrees=False)

        # Build yaw-only rotation, Keep only yaw
        r_yaw_only = R.from_euler('xyz', [0.0, 0.0, yaw], degrees=False)
        rot_mat_yaw_only = r_yaw_only.as_matrix()

        # Rotate offset, place camera
        world_offset = rot_mat_yaw_only @ LOCAL_CAMERA_OFFSET
        dynamic_cam_position = base_pos + world_offset

        # Rotate look-at offset, place look-at
        world_lookat_offset = rot_mat_yaw_only @ LOCAL_LOOKAT_OFFSET
        dynamic_cam_lookat = base_pos + world_lookat_offset

        # Set camera
        env.gym.set_camera_location(
            dynamic_camera_handle,
            env.envs[0],
            gymapi.Vec3(*dynamic_cam_position),
            gymapi.Vec3(*dynamic_cam_lookat)
        )

        # Render camera sensors
        env.gym.render_all_camera_sensors(env.sim)

        # --- Capture image from fixed camera ---
        image_buffer_fixed = env.gym.get_camera_image(env.sim, env.envs[0], fixed_camera_handle, gymapi.IMAGE_COLOR)

        # Convert image to numpy array and reshape
        image_data_fixed = np.frombuffer(image_buffer_fixed, dtype=np.uint8).reshape(
            (camera_props.height, camera_props.width, 4))

        # Convert from RGBA to BGR format
        image_bgr_fixed = image_data_fixed[:, :, :3]

        # Write frame to fixed camera video
        out_fixed.write(image_bgr_fixed)

        # --- Capture image from dynamic camera ---
        image_buffer_dynamic = env.gym.get_camera_image(env.sim, env.envs[0], dynamic_camera_handle, gymapi.IMAGE_COLOR)

        # Convert image to numpy array and reshape
        image_data_dynamic = np.frombuffer(image_buffer_dynamic, dtype=np.uint8).reshape(
            (camera_props.height, camera_props.width, 4))

        # Convert from RGBA to BGR format
        image_bgr_dynamic = image_data_dynamic[:, :, :3]

        # Write frame to dynamic camera video
        out_dynamic.write(image_bgr_dynamic)

        # *** Save right-camera images every 2 steps if enabled ***
        if args.save_right_camera_figures and (i % 2 == 0) and (i < 100):
            cv2.imwrite(os.path.join("results_fig/right_video_figs", args.video_figures_path, f"right_{i}.png"),
                        image_bgr_dynamic)

    # Release the VideoWriter
    out_fixed.release()
    out_dynamic.release()
    print("release success")


if __name__ == '__main__':
    EXPORT_POLICY = False  # oroginally true
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
    # [XW]
    # ---------------------------------------------------------------------------------
    # Personal reminder
    # export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
    # ---------------------------------------------------------------------------------
