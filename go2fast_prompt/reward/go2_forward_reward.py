import torch
import numpy as np
from isaacgym.torch_utils import *

# This has been tested to be a good reward for locomotion on flat ground plane

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        """ Compute rewards
            Compute each reward component first
            Then compute the total reward
            Return the total reward, and the recording of all reward components
        """
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.

        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(env.commands[:, :2] - env.base_lin_vel[:, :2]), dim=1)
        tracking_lin_vel_reward = 1.0 * torch.exp(-lin_vel_error / 0.25)

        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(env.commands[:, 2] - env.base_ang_vel[:, 2])
        tracking_ang_vel_reward = 0.5 * torch.exp(-ang_vel_error / 0.25)

        # Penalize z axis base linear velocity
        lin_vel_z_reward = -2.0 * torch.square(env.base_lin_vel[:, 2])

        # Penalize xy axes base angular velocity
        ang_vel_xy_reward = -0.05 * torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        # Penalize torques
        torques_reward = -0.0002 * torch.sum(torch.square(env.torques), dim=1)

        # Penalize dof accelerations
        dof_acc_reward = -2.5e-7 * torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)

        # # Reward long steps, but for fast step, the reward scale is much smaller than the 1.0 in normal locomotion
        # # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        # contact = env.contact_forces[:, env.feet_indices, 2] > 1.
        # contact_filt = torch.logical_or(contact, env.last_contacts)
        # env.last_contacts = contact
        # first_contact = (env.feet_air_time > 0.) * contact_filt
        # env.feet_air_time += env.dt
        # rew_airTime = torch.sum((env.feet_air_time - 0.5) * first_contact, dim=1)  # reward only on first contact with the ground
        # rew_airTime *= torch.norm(env.commands[:, :2], dim=1) > 0.1  # no reward for zero command
        # env.feet_air_time *= ~contact_filt
        # feet_air_time_reward = 0.1 * rew_airTime

        # Penalize collisions on selected bodies
        collision_reward = -1.0 * torch.sum(1. * (torch.norm(env.contact_forces[:, env.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

        # Gait pattern reward (encourage bounding gait)
        # Foot contact forces for all feet
        contact_states = (env.contact_forces[:, env.feet_indices, 2] > 1.0).float()  # Binary: 1 for contact, 0 for no contact
        # Front and rear pairs: front-left (0) & front-right (1), rear-left (2) & rear-right (3)
        pair1_sync = contact_states[:, 0] - contact_states[:, 3]  # Should alternate (closer to 0 over time)
        pair2_sync = contact_states[:, 1] - contact_states[:, 2]  # Should alternate (closer to 0 over time)
        # Penalize deviations from opposite phases
        gait_pattern_reward = -0.1 * (torch.square(pair1_sync) + torch.square(pair2_sync))

        # Gait phase reward (encourage bounding gait)
        # When one diagonal is on the ground, want the other diagonal is in the air
        sum_diag1 = contact_states[:, 0] + contact_states[:, 3]  # front-left + rear-right
        sum_diag2 = contact_states[:, 1] + contact_states[:, 2]  # front-right + rear-left
        # A simple measure of how "in phase" they are:
        sum_diff = sum_diag1 - sum_diag2
        # Penalize small differences => encourage a large difference
        # e.g. a negative reward that grows (more negative) if sum_diff^2 is small
        gait_phase_reward = -0.1 * torch.exp(-2.0 * torch.square(sum_diff))

        # Penalize changes in actions, but for fast step, the penalty scale is much smaller than the 0.01 in normal locomotion
        action_rate_reward = -0.00001 * torch.sum(torch.square(env.last_actions - env.actions), dim=1)

        # Penalize dof positions too close to the limit
        out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)  # lower limit
        out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)
        dof_pos_limits_reward = -10.0 * torch.sum(out_of_limits, dim=1)

        # # Penalize base height away from target
        # target_height_z = 0.34  # Ideal height of the robot’s torso
        # base_height = env.root_states[:, 2]
        # height_reward = -0.05 * torch.square(base_height - target_height_z)  # reward to maintain height

        # Height reward component
        target_height_z = 0.34  # Ideal height of the robot’s torso
        base_height = env.root_states[:, 2]
        height_error = torch.abs(base_height - target_height_z)
        temperature_height = 5.0  # Temperature parameter for the height reward
        height_reward = 1.0 * torch.exp(-temperature_height * height_error)  # More weight to maintain height

        # Combine reward components to compute the total reward in this step
        total_reward = (tracking_lin_vel_reward + tracking_ang_vel_reward + lin_vel_z_reward +
                        ang_vel_xy_reward + torques_reward + dof_acc_reward + collision_reward +
                        gait_pattern_reward + gait_phase_reward + action_rate_reward + dof_pos_limits_reward + height_reward)

        # # Normalizing the total reward to avoid exploding values
        # total_reward = total_reward / (1 + torch.abs(total_reward))  # Additional normalization for stability

        # Debug information
        reward_components = {"tracking_lin_vel_reward": tracking_lin_vel_reward,
                             "tracking_ang_vel_reward": tracking_ang_vel_reward,
                             "lin_vel_z_reward": lin_vel_z_reward,
                             "ang_vel_xy_reward": ang_vel_xy_reward,
                             "torques_reward": torques_reward,
                             "dof_acc_reward": dof_acc_reward,
                             "collision_reward": collision_reward,
                             "gait_pattern_reward": gait_pattern_reward,
                             "gait_phase_reward": gait_phase_reward,
                             "action_rate_reward": action_rate_reward,
                             "dof_pos_limits_reward": dof_pos_limits_reward,
                             "height_reward": height_reward}
        return total_reward, reward_components

    # Success criteria as forward velocity
    def compute_success(self):
        lin_vel_error = torch.sum(torch.square(self.env.commands[:, :2] - self.env.base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.env.commands[:, 2] - self.env.base_ang_vel[:, 2])

        return 1.0 * torch.exp(-lin_vel_error / 0.25) + 0.1 * torch.exp(-ang_vel_error / 0.25)