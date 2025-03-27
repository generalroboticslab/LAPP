import torch
import numpy as np
from isaacgym.torch_utils import *


class CustomReward():
    def __init__(self, env):
        self.env = env
        self.device = env.device  # Ensure correct device

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        """ Compute rewards for wave terrain """
        env = self.env

        # 1. Linear velocity tracking along x-axis
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel_reward = 2.3 * torch.exp(-lin_vel_error / 0.20)

        # 2. Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
        tracking_ang_vel_reward = 0.8 * torch.exp(-ang_vel_error / 0.1)

        # 3. Penalize z-axis velocity
        # lin_vel_z_reward = -0.001 * torch.square(env.base_lin_vel[:, 2])

        # 4. Base height tracking (adjusted for wave terrain)
        target_height_z = 0.34
        base_height = env.root_states[:, 2]
        ground_x = env.root_states[:, 0]
        ground_y = env.root_states[:, 1]
        ground_z = env._get_terrain_heights(ground_x, ground_y)
        base2ground_height = base_height - ground_z
        height_reward = -1.5 * torch.square(base2ground_height - target_height_z)

        # 5. Penalize torques
        torques_reward = -0.00001 * torch.sum(torch.square(env.torques), dim=1)

        # 6. Penalize changes in actions
        action_rate_reward = -0.0045 * torch.sum(torch.square(env.last_actions - env.actions), dim=1)
        # 0.0035 -> 0.0045

        # 7. Encourage smoother joint motions (penalize excessive joint accelerations)
        dof_acc_penalty = -1e-8 * torch.sum(torch.square((env.dof_vel - env.last_dof_vel) / env.dt), dim=1)

        # 8. Air time reward for dynamic gaits
        contact = env.contact_forces[:, env.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, env.last_contacts)
        env.last_contacts = contact
        first_contact = (env.feet_air_time > 0.0) * contact_filt
        env.feet_air_time += env.dt
        rew_airTime = torch.sum((env.feet_air_time - 0.4) * first_contact, dim=1)
        rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1
        env.feet_air_time *= ~contact_filt
        air_time_reward = 0.5 * rew_airTime

        # 9. Collision penalty (avoid collisions with terrain or robot parts)
        collision_penalty = -1. * torch.sum(
            1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.13),
            dim=1
        )

        # 10. Gait pattern reward (encourage trot gait using phase alignment)
        diag_sync = (contact[:, 0] == contact[:, 3]) & (contact[:, 1] == contact[:, 2])
        gait_pattern_reward = 0.0001 * torch.sum(diag_sync.float())

        # 11. Penalize use only two feet
        lack_of_foot_usage = (~contact).float().sum(dim=1)
        lack_of_foot_usage_penalty = -0.01 * lack_of_foot_usage

        # Combine all components into the total reward
        total_reward = (
                tracking_lin_vel_reward +
                tracking_ang_vel_reward +
                # lin_vel_z_reward +
                height_reward +
                torques_reward +
                action_rate_reward +
                dof_acc_penalty +
                air_time_reward +
                collision_penalty +
                gait_pattern_reward +
                lack_of_foot_usage_penalty
        )

        # Debug information for reward components
        reward_components = {
            "tracking_lin_vel_reward": tracking_lin_vel_reward,
            "tracking_ang_vel_reward": tracking_ang_vel_reward,
            # "lin_vel_z_reward": lin_vel_z_reward,
            "height_reward": height_reward,
            "torques_reward": torques_reward,
            "action_rate_reward": action_rate_reward,
            "dof_acc_penalty": dof_acc_penalty,
            "air_time_reward": air_time_reward,
            "collision_penalty": collision_penalty,
            "gait_pattern_reward": gait_pattern_reward,
            "lack_of_foot_usage_penalty": lack_of_foot_usage_penalty
        }

        return total_reward, reward_components

    def compute_success(self):
        """ Compute success metrics """
        env = self.env
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel_reward = 2.3 * torch.exp(-lin_vel_error / 0.20)

        # ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
        # tracking_ang_vel_reward = 1.3 * torch.exp(-ang_vel_error / 0.1)

        action_rate_reward = -0.0025 * torch.sum(torch.square(env.last_actions - env.actions), dim=1)
        # 0.0035 -> 0.0025
        collision_penalty = -0.8 * torch.sum(
            1.0 * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.13),
            dim=1
        )

        return (
                tracking_lin_vel_reward +
                # tracking_ang_vel_reward +
                action_rate_reward +
                collision_penalty
        )
