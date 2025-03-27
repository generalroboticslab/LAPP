import time
import os
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from repo.unitree_rl_gym.legged_gym import LEGGED_GYM_ROOT_DIR
from .base_task import BaseTask
from repo.unitree_rl_gym.legged_gym.utils.math import wrap_to_pi
from repo.unitree_rl_gym.legged_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor
from repo.unitree_rl_gym.legged_gym.utils.helpers import class_to_dict
from .llmpref_go2_light_backflip_robot_config import LLMPrefGo2LightBackflipRobotCfg
from .terrain import Terrain


import types
import importlib

# this is an example of using an entire compute_reward without calling reward items functions
# it has the same function as the dreureka_go2_robot.py, but the code structure of computing reward is different
# In this gemini3 version, I put all the reward scales in the compute_reward function,
# modified from the llmhydra2_go2_robot.py file. Deal with the termination condition to allow jump and flip in the air.

class LLMPrefGo2LightBackflipRobot(BaseTask):
    def __init__(self, cfg: LLMPrefGo2LightBackflipRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.reward_module_name = cfg.env.reward_module_name
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)

        # Initialize num_envs and difficulty_levels
        self.num_envs = self.cfg.env.num_envs
        self.difficulty_levels = torch.zeros(self.num_envs, dtype=torch.long, device=sim_device)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.cfg.env.test:
                elapsed_time = self.gym.get_elapsed_time(self.sim)
                sim_time = self.gym.get_sim_time(self.sim)
                if sim_time - elapsed_time > 0:
                    time.sleep(sim_time - elapsed_time)

            if self.device == 'cpu' or self.cfg.viewer.record:
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)

        # step physics and render each frame
        # PJ: I move the render from before to for loop to after the for loop, to ensure step_graphics happens later than refresh_dof_state_tensor
        self.render()
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        # PJ: Reset reset_buf at the start
        # self.reset_buf[:] = 0

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.rpy[:] = get_euler_xyz_in_tensor(self.base_quat[:])
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # Reset pitch_state after completing a 360-degree backflip
        reset_pitch_state_condition = (self.pitch_state == 2.0)
        self.pitch_state[reset_pitch_state_condition] = 0.0
        self.last_pitch_state[reset_pitch_state_condition] = 0.0

        backflip180_condition = (self.rpy[:, 1] < -3.0) & (self.pitch_state == 0.0)  # pi approximately is 3.14
        backflip360_condition = (self.rpy[:, 1] > 0.0) & (self.rpy[:, 1] < 0.1) & (self.pitch_state == 1.0)

        self.pitch_state[backflip360_condition] = 2.0
        self.pitch_state[backflip180_condition] = 1.0

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        # PJ: of the z is higher than previous, put it into the self.height_history_buf
        self.height_history_buf = torch.maximum(self.height_history_buf, self.root_states[:, 2])

        # PJ: I comment out the check_termination() here, so don't stop even if the torso hit the ground
        # self.check_termination()
        # PJ: I only check the time termination of exceeding maximum episode length
        self.check_time_termination()
        self.compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if len(env_ids) != 0:
        #     print(f"the environments {env_ids} are reset in post_physics_step")
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        # PJ: I add it here for calculate rotation for the back flip task
        # This happens after the compute_reward function, so it is correct
        self.last_projected_gravity[:] = self.projected_gravity[:]
        # PJ: I add it here to get last pitch state before the pitch state is updated
        self.last_pitch_state[:] = self.pitch_state[:]
        self.last_rpy[:] = self.rpy[:]

    def check_termination(self):
        """ Check if environments need to be reset
        """
        # PJ: I add the clone here, to avoid RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update
        # self.reset_buf = torch.clone(torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1))
        # PJ: previously this termination says if the "base" part (torso) hit the ground  exceeds a threshold (1.0) force, the episode will terminate
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # self.reset_buf |= torch.clone(torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8))
        # PJ: previously this termination says If the robot's roll (rpy[:, 0]) exceeds 0.8 radians (~45.8 degrees)
        # or its pitch (rpy[:, 1]) exceeds 1.0 radians (~57.3 degrees), the episode is terminated.
        # Now I comment it out, because backflip and many other tasks should allow this
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def check_time_termination(self):
        """ Check if environments need to be reset
        """
        # PJ: I add the clone here, to avoid RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update
        # self.reset_buf = torch.clone(torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1))
        # PJ: previously this termination says if the "base" part (torso) hit the ground  exceeds a threshold (1.0) force, the episode will terminate
        # I also comment this out to only terminate when exceeds the max episode length
        # self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        # PJ: Initialize reset_buf to zeros
        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # self.reset_buf |= torch.clone(torch.logical_or(torch.abs(self.rpy[:, 1]) > 1.0, torch.abs(self.rpy[:, 0]) > 0.8))
        # PJ: previously this termination says If the robot's roll (rpy[:, 0]) exceeds 0.8 radians (~45.8 degrees)
        # or its pitch (rpy[:, 1]) exceeds 1.0 radians (~57.3 degrees), the episode is terminated.
        # Now I comment it out, because backflip and many other tasks should allow this
        # self.reset_buf |= torch.logical_or(torch.abs(self.rpy[:,1])>1.0, torch.abs(self.rpy[:,0])>0.8)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        # print("in check_time_termination, the none 0 time out ids are:")
        # print(self.time_out_buf.nonzero(as_tuple=False).flatten())
        self.reset_buf |= self.time_out_buf
        # print("in check_time_termination, the none 0 reset ids are:")
        # print(self.reset_buf.nonzero(as_tuple=False).flatten())

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return

        # reset robot states
        # print(f"the environments {env_ids} are reset in reset_idx")
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        if self.custom_origins:
            # Update environment origins based on difficulty levels
            terrain_env_origins = torch.tensor(self.terrain.env_origins, device=self.device)
            total_tiles = terrain_env_origins.shape[0]
            total_levels = len(self.cfg.terrain.difficulty_levels)
            tiles_per_level = total_tiles // total_levels

            for env_id in env_ids:
                difficulty_level = self.difficulty_levels[env_id].item()
                level_start_idx = difficulty_level * tiles_per_level
                tile_idx = level_start_idx + torch.randint(0, tiles_per_level, (1,), device=self.device).item()
                tile_idx = min(tile_idx, total_tiles - 1)
                self.env_origins[env_id] = terrain_env_origins[tile_idx]

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        # PJ: also reset the height history to 0.
        self.height_history_buf[env_ids] = 0.
        # PJ: also reset the pitch state to 0.
        self.pitch_state[env_ids] = 0.
        self.last_pitch_state[env_ids] = 0.
        # PJ: I add the detach here, to avoid RuntimeError: Inplace update to inference tensor outside InferenceMode is not allowed.You can make a clone to get a normal tensor before doing inplace update
        self.reset_buf = self.reset_buf.detach()
        self.reset_buf[env_ids] = 1  # 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(
                self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0  # initialize the reward buffer to 0
        rew, rew_components = self.reward_container.compute_reward()
        self.rew_buf += rew
        for name, rew_term in rew_components.items():
            self.episode_sums[name] += rew_term
            # self.command_sums[name] += rew_term

        self.episode_sums["success"] += self.reward_container.compute_success()
        self.episode_sums["total"] += self.rew_buf

        # self.command_sums["lin_vel_raw"] += self.base_lin_vel[:, 0]
        # self.command_sums["ang_vel_raw"] += self.base_ang_vel[:, 2]
        # self.command_sums["lin_vel_residual"] += (self.base_lin_vel[:, 0] - self.commands[:, 0]) ** 2
        # self.command_sums["ang_vel_residual"] += (self.base_ang_vel[:, 2] - self.commands[:, 2]) ** 2
        # self.command_sums["ep_timesteps"] += 1

    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # def create_sim(self):
    #     """ Creates simulation, terrain and evironments
    #     """
    #     self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
    #     self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
    #     self._create_ground_plane()
    #     self._create_envs()

    def create_sim(self):
        """ Creates simulation, terrain, and environments """
        self.up_axis_idx = 2  # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        if self.cfg.terrain.mesh_type == 'plane':
            self._create_ground_plane()
            self.custom_origins = False
        elif self.cfg.terrain.mesh_type == 'trimesh':
            self._create_terrain()
            self.custom_origins = True
        else:
            raise ValueError(f"Unknown terrain mesh type: {self.cfg.terrain.mesh_type}")

        self._create_envs()

    def _create_terrain(self):
        """ Creates a terrain mesh and adds it to the simulation """
        # Create the terrain
        self.terrain = Terrain(self.cfg.terrain, num_robots=self.num_envs)
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = -self.terrain.border_size
        tm_params.transform.p.y = -self.terrain.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution

        self.gym.add_triangle_mesh(
            self.sim,
            self.terrain.vertices.flatten(order='C'),
            self.terrain.triangles.flatten(order='C'),
            tm_params
        )
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)
    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    # ------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        return props

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos) - self.d_gains*self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """Resets root states position and velocities of selected environments"""
        # Base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if not self.cfg.env.test:
            # xy position within 1m of the center. Only do this randomization in training, but not testing.
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device)

        if self.cfg.terrain.terrain_type == 'pyramid_stairs':
            # Adjust the z position based on terrain height
            x = self.root_states[env_ids, 0]
            y = self.root_states[env_ids, 1]
            z = self._get_stairs_terrain_heights(x, y)
            # print(f"at the position ({x[0]}, {y[0]}), the height of the terrain is {z[0]}")
            z += self.cfg.env.robot_height_offset  # Offset to place the robot above the terrain
            self.root_states[env_ids, 2] = z

        # Base velocities
        if not self.cfg.env.test:
            # Randomize the initial base velocities. Only do this randomization in training, but not testing.
            # self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
            self.root_states[env_ids, 7:13] = torch_rand_float(-0.1, 0.1, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel
            if self.cfg.domain_rand.random_in_air == 1:  # random in air for back flip
                # PJ: for the back flip task, we also randomize the height of the robot, and the pitch angle of the robot
                # Randomly reset the initial base heights.
                # Generate a random boolean tensor with 50% True and 50% False.
                prob = torch.rand(len(env_ids), device=self.device) < 0.5  # True with 50% probability
                # print(f"The robot start state is {prob[0]}")
                # Generate random heights between 1.5 and 2.0 meters.
                random_heights = torch_rand_float(self.cfg.domain_rand.height_initialization_range[0], self.cfg.domain_rand.height_initialization_range[1], (len(env_ids), 1), device=self.device)
                random_heights = random_heights.squeeze(-1)  # Remove the singleton dimension if necessary

                # Use torch.where to choose between 0.3 and a random height.
                new_heights = torch.where(prob, random_heights, torch.full_like(random_heights, self.base_init_state[2]))

                # Assign the new heights directly.
                self.root_states[env_ids, 2] = new_heights

                # Identify robots that are in the air
                air_env_ids = env_ids[prob]

                if len(air_env_ids) > 0:
                    # Generate pitch angles in [pi/2, 3pi/2]
                    pitch_angles = torch_rand_float(torch.pi * self.cfg.domain_rand.angle_initialization_range[0], torch.pi * self.cfg.domain_rand.angle_initialization_range[1], (len(air_env_ids), 1), device=self.device)
                    pitch_angles = pitch_angles.squeeze(-1)  # Remove the singleton dimension if necessary

                    # Wrap the angles to [-pi, pi]
                    pitch_angles = (pitch_angles + torch.pi) % (2 * torch.pi) - torch.pi

                    # Generate roll and yaw angles (you can set these to zero or randomize them)
                    roll_angles = torch.zeros(len(air_env_ids), device=self.device)
                    yaw_angles = torch.zeros(len(air_env_ids), device=self.device)

                    # Convert Euler angles to quaternions
                    quaternions = quat_from_euler_xyz(roll_angles, pitch_angles, yaw_angles)

                    # Normalize the quaternions
                    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

                    # Assign the new quaternions to the robots in the air
                    self.root_states[air_env_ids, 3:7] = quaternions
        else:
            self.root_states[env_ids, 7:13] = torch.zeros(len(env_ids), 6, device=self.device)
            if self.cfg.domain_rand.random_in_air == 1:  # random in air for back flip
                print("random in air")
                # PJ: for the back flip task, we also randomize the height of the robot, and the pitch angle of the robot
                # Randomly reset the initial base heights.
                # Generate a random boolean tensor with 50% True and 50% False.
                prob = torch.rand(len(env_ids), device=self.device) < 1.0  # True with 50% probability
                # print(f"The robot start state is {prob[0]}")
                # Generate random heights between 1.5 and 2.0 meters.
                random_heights = torch_rand_float(self.cfg.domain_rand.height_initialization_range[0],
                                                  self.cfg.domain_rand.height_initialization_range[1],
                                                  (len(env_ids), 1), device=self.device)
                random_heights = random_heights.squeeze(-1)  # Remove the singleton dimension if necessary

                # Use torch.where to choose between 0.3 and a random height.
                new_heights = torch.where(prob, random_heights, torch.full_like(random_heights, self.base_init_state[2]))

                # Assign the new heights directly.
                self.root_states[env_ids, 2] = new_heights

                # Identify robots that are in the air
                air_env_ids = env_ids[prob]

                if len(air_env_ids) > 0:
                    # Generate pitch angles in [pi/2, 3pi/2]
                    pitch_angles = torch_rand_float(torch.pi * self.cfg.domain_rand.angle_initialization_range[0],
                                                    torch.pi * self.cfg.domain_rand.angle_initialization_range[1],
                                                    (len(air_env_ids), 1), device=self.device)
                    pitch_angles = pitch_angles.squeeze(-1)  # Remove the singleton dimension if necessary

                    # Wrap the angles to [-pi, pi]
                    pitch_angles = (pitch_angles + torch.pi) % (2 * torch.pi) - torch.pi

                    # Generate roll and yaw angles (you can set these to zero or randomize them)
                    roll_angles = torch.zeros(len(air_env_ids), device=self.device)
                    yaw_angles = torch.zeros(len(air_env_ids), device=self.device)

                    # Convert Euler angles to quaternions
                    quaternions = quat_from_euler_xyz(roll_angles, pitch_angles, yaw_angles)

                    # Normalize the quaternions
                    quaternions = quaternions / torch.norm(quaternions, dim=1, keepdim=True)

                    # Assign the new quaternions to the robots in the air
                    self.root_states[air_env_ids, 3:7] = quaternions

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32)
        )

    def _get_stairs_terrain_heights(self, x, y):
        """Returns the terrain height at the given x, y positions."""
        # Convert x, y positions to numpy arrays
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        # Convert world positions to pixel indices
        px = ((x + self.terrain.border_size) / self.terrain.horizontal_scale).astype(int)
        py = ((y + self.terrain.border_size) / self.terrain.horizontal_scale).astype(int)

        # Clamp indices to be within the height field array bounds
        px = np.clip(px, 0, self.terrain.tot_rows - 1)
        py = np.clip(py, 0, self.terrain.tot_cols - 1)

        # Get the height values from the height field
        height_field = self.terrain.height_field_raw

        # the height field is indexed as [px, py]
        # print(f"the (px, py) pass into the height feild is ({px[0]}, {py[0]})")
        heights = height_field[px, py] * self.terrain.vertical_scale

        # Convert to torch tensor and move to device
        heights = torch.from_numpy(heights).to(self.device, dtype=torch.float)

        return heights

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    def update_difficulty_levels(self, success_score):
        """Updates the difficulty levels of environments based on performance."""
        # Compute the success metric for each environment
        # For example, you might define success as the average reward per episode
        success_threshold = self.cfg.terrain.upgrade_success_threshold
        success_threshold_lower_bound = self.cfg.terrain.upgrade_success_threshold_lower_bound
        downgrade_threshold = self.cfg.terrain.downgrade_success_threshold
        max_level = len(self.cfg.terrain.difficulty_levels) - 1
        min_level = 0
        # print(f"the current mean_success is:{success_score}")

        # Adjust difficulty levels based on the mean success metric
        previous_level = self.difficulty_levels[0].item()  # Assuming all envs have same level
        new_level = previous_level

        # success_threshold -= 1 * new_level  # gradually reduce threshold
        # success_threshold = max(success_threshold, success_threshold_lower_bound)
        # print(f"the success threshold is {success_threshold}")

        if success_score > success_threshold:
            # Upgrade difficulty level
            new_level = min(previous_level + 1, max_level)
            self.difficulty_levels[:] = new_level
            # print(f"Upgrading difficulty level to {new_level}")
        # elif success_score < downgrade_threshold:  # sometimes we can not downgrade for easier curriculum
        #     # Downgrade difficulty level
        #     new_level = max(previous_level - 1, min_level)
        #     self.difficulty_levels[:] = new_level
        #     print(f"Downgrading difficulty level to {new_level}")
        # else:
        #     # Keep the same difficulty level
        #     print(f"Keeping difficulty level at {previous_level}")

        # Optionally, reset the environments to apply the new difficulty levels immediately
        # If the difficulty level has changed, reset all environments
        if new_level != previous_level:
            env_ids = torch.arange(self.num_envs, device=self.device)
            self.reset_idx(env_ids)

        return new_level

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0. # commands
        noise_vec[12:12+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[12+self.num_actions:12+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[12+2*self.num_actions:12+3*self.num_actions] = 0. # previous actions

        return noise_vec

    # ----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rpy = get_euler_xyz_in_tensor(self.base_quat)
        self.base_pos = self.root_states[:self.num_envs, 0:3]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False)  # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False, )  # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        # PJ: I add this for encouraging backwards rotation for back flip
        self.last_projected_gravity = torch.zeros_like(self.projected_gravity)
        # PJ: I add this for encouraging backwards rotation for back flip
        self.pitch_state = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # PJ: I add this for encouraging backwards rotation for back flip
        self.last_pitch_state = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        # Re-initialize buffers to ensure they are regular tensors
        # PJ: added for iteration of training
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # PJ: add a height_history_buf to record the highest history z position of the base torso body
        self.height_history_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        # PJ: I add this for recording the rpy of the last step
        self.last_rpy = torch.zeros_like(self.rpy)

        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(
                self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else:
            self.privileged_obs_buf = None

        # initialize the buffers for storing trajectories pairs for comparing and get preference score.
        # self.num_pairs is the number of trajectories pairs that we use to train preference predictors in each interval of training
        # e.g. self.cfg.env.pref_buf_interval==50, self.cfg.env.num_pref_pairs_per_episode==4
        self.pref_buf_interval = self.cfg.env.pref_buf_interval
        self.num_pref_pairs_per_episode = self.cfg.env.num_pref_pairs_per_episode
        self.num_pairs = self.pref_buf_interval * self.num_pref_pairs_per_episode
        self.base_pos_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env, len(self.base_pos[1]),
                                        device=self.device, dtype=torch.float, requires_grad=False)
        self.rpy_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env, len(self.rpy[1]),
                                   device=self.device, dtype=torch.float, requires_grad=False)
        self.base_lin_vel_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env,
                                            len(self.base_lin_vel[1]), device=self.device, dtype=torch.float,
                                            requires_grad=False)
        self.base_ang_vel_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env,
                                            len(self.base_ang_vel[1]), device=self.device, dtype=torch.float,
                                            requires_grad=False)
        self.feet_contacts_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env,
                                             len(self.feet_indices), dtype=torch.bool, device=self.device,
                                             requires_grad=False)
        self.commands_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env,
                                        self.cfg.commands.num_commands, device=self.device, dtype=torch.float,
                                        requires_grad=False)
        self.actions_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env, self.num_actions,
                                       device=self.device, dtype=torch.float, requires_grad=False)
        # PJ: self.obs_buf is used, and the shape of self.obs_buf is (num_env, dim_obs), so I use self.obs_pairs_buf here as the name
        # PJ: the obs_buf includes the self.action of the last step as a part of the observation, and the actions_buf includes the action of the current step, which is induced by the current step obs
        self.obs_pairs_buf = torch.zeros(self.num_pairs, 2, self.cfg.env.num_steps_per_env, len(self.obs_buf[1]),
                                         device=self.device, dtype=torch.float, requires_grad=False)

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def _prepare_reward_function(self):
        """ Prepares a list of episode sums for all reward items
        """
        import importlib
        print(f"Loading reward module: {self.reward_module_name}")
        reward_module = importlib.import_module(self.reward_module_name)
        importlib.reload(reward_module)
        self.reward_container = reward_module.EurekaReward(self)

        _, reward_components = self.reward_container.compute_reward()
        self.reward_names = list(reward_components.keys())

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_names}
        self.episode_sums["total"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums["success"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment,
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        # asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2, 1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)

            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise, create a grid.
        """
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        if self.custom_origins:
            # Convert terrain env_origins to torch tensor
            terrain_env_origins = torch.tensor(self.terrain.env_origins, device=self.device)
            total_tiles = terrain_env_origins.shape[0]
            total_levels = len(self.cfg.terrain.difficulty_levels)
            tiles_per_level = total_tiles // total_levels

            for env_id in range(self.num_envs):
                difficulty_level = self.difficulty_levels[env_id].item()
                level_start_idx = difficulty_level * tiles_per_level
                # Randomly select a tile within the difficulty level
                tile_idx = level_start_idx + torch.randint(0, tiles_per_level, (1,), device=self.device).item()
                tile_idx = min(tile_idx, total_tiles - 1)
                self.env_origins[env_id] = terrain_env_origins[tile_idx]

            # Optionally add terrain offset
            terrain_center = torch.tensor([
                self.terrain.width_per_env_pixels * self.terrain.horizontal_scale / 2,
                self.terrain.length_per_env_pixels * self.terrain.horizontal_scale / 2,
                0.0
            ], device=self.device)
            self.env_origins += terrain_center
        else:
            # Existing code to create a grid of robots
            num_cols = int(np.floor(np.sqrt(self.num_envs)))
            num_rows = int(np.ceil(self.num_envs / num_cols))
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.0

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)

        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)