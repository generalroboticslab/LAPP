import torch
import numpy as np
from isaacgym.torch_utils import *

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        """ Compute improved rewards
            Compute each reward component first
            Then compute the total reward
            Return the total reward, and the recording of all reward components
        """
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.

        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel_reward = 3.0 * torch.exp(-lin_vel_error / 0.10)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
        tracking_ang_vel_reward = 1.0 * torch.exp(-ang_vel_error / 0.05)

        # # Penalize z axis base linear velocity
        lin_vel_z_reward = -0.00001 * torch.square(env.base_lin_vel[:, 2])

        # Penalize xy axes base angular velocity
        ang_vel_xy_reward = -0.1 * torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        # Penalize torques
        torques_reward = -0.0001 * torch.sum(torch.square(env.torques), dim=1)

        # Penalize dof accelerations
        dof_acc_reward = -5.0e-8 * torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)

        # Reward air time
        contact = env.contact_forces[:, env.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, env.last_contacts)
        env.last_contacts = contact
        first_contact = (env.feet_air_time > 0.) * contact_filt
        env.feet_air_time += env.dt
        rew_airTime = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1)
        rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1
        env.feet_air_time *= ~contact_filt
        feet_air_time_reward = 0.6 * rew_airTime

        # Penalize collisions
        collision_reward = -15.0 * torch.sum(1. * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

        # Penalize changes in actions
        action_rate_reward = -0.015 * torch.sum(torch.square(env.last_actions - env.actions), dim=1)

        # Penalize dofs close to limits
        out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)
        dof_pos_limits_reward = -6.0 * torch.sum(out_of_limits, dim=1)

        # Penalize base height away from target
        target_height_z = 0.34
        base_height = env.root_states[:, 2]
        # get the ground height of the terrain
        ground_x = env.root_states[:, 0]
        ground_y = env.root_states[:, 1]
        ground_z = env._get_stairs_terrain_heights(ground_x, ground_y)
        # calculate the base-to-ground height
        base2ground_height = base_height - ground_z
        height_reward = -0.000001 * torch.square(base2ground_height - target_height_z)

        # stumbling penalty
        stumble = (torch.norm(env.contact_forces[:, env.feet_indices, :2], dim=2) > 5.) * (torch.abs(env.contact_forces[:, env.feet_indices, 2]) < 1.)
        stumble_reward = -4.0 * torch.sum(stumble, dim=1)

        # Combine reward components to compute the total reward in this step
        total_reward = (tracking_lin_vel_reward + tracking_ang_vel_reward + lin_vel_z_reward +
                        ang_vel_xy_reward + torques_reward + dof_acc_reward + feet_air_time_reward +
                        collision_reward + action_rate_reward + dof_pos_limits_reward + height_reward + stumble_reward)

        # Debug information
        reward_components = {"tracking_lin_vel_reward": tracking_lin_vel_reward,
                             "tracking_ang_vel_reward": tracking_ang_vel_reward,
                             "lin_vel_z_reward": lin_vel_z_reward,
                             "ang_vel_xy_reward": ang_vel_xy_reward,
                             "torques_reward": torques_reward,
                             "dof_acc_reward": dof_acc_reward,
                             "feet_air_time_reward": feet_air_time_reward,
                             "collision_reward": collision_reward,
                             "action_rate_reward": action_rate_reward,
                             "dof_pos_limits_reward": dof_pos_limits_reward,
                             "height_reward": height_reward,
                             "stumble_reward": stumble_reward}

        return total_reward, reward_components

    # Success criteria as forward velocity
    def compute_success(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])

        # stumbling penalty
        stumble = (torch.norm(self.env.contact_forces[:, self.env.feet_indices, :2], dim=2) > 5.) * (torch.abs(self.env.contact_forces[:, self.env.feet_indices, 2]) < 1.)
        stumble_success = -2.0 * torch.sum(stumble, dim=1)

        return 1.0 * torch.exp(-lin_vel_error / 0.25) + 0.1 * torch.exp(-ang_vel_error / 0.25) + stumble_success