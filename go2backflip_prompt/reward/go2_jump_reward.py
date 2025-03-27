import torch
import numpy as np
from isaacgym.torch_utils import *

# just encourage the robot to jump as high as it can

class EurekaReward():
    def __init__(self, env):
        self.env = env

    def load_env(self, env):
        self.env = env

    def compute_reward(self):
        """ Compute rewards for the backflip task """
        env = self.env  # Do not skip this line. Afterwards, use env.{parameter_name} to access parameters of the environment.

        # Penalize angular velocity around x-axis (roll) and z-axis (yaw)
        roll_rate = env.root_states[:, 10]  # Angular velocity around x-axis (roll)
        yaw_rate = env.root_states[:, 12]  # Angular velocity around z-axis (yaw)
        roll_rate_reward = -0.05 * torch.square(roll_rate)  # Penalize deviation in roll, org -5.0, 0.5
        yaw_rate_reward = -0.2 * torch.square(yaw_rate)  # Penalize deviation in yaw, org -5.0, -2.0

        # Penalize roll and yaw around x-axis (roll) and z-axis (yaw)
        roll = env.rpy[:, 0]
        pitch = env.rpy[:, 1]
        yaw = env.rpy[:, 2]
        roll_reward = -0.2 * torch.square(roll)  # Penalize deviation in roll
        pitch_reward = -0.2 * torch.square(pitch)  # Penalize deviation in pitch
        yaw_reward = -1.0 * torch.square(yaw)  # Penalize deviation in yaw

        # Encourage z axis base linear velocity
        lin_vel_z_reward = 0.4 * torch.square(env.root_states[:, 9])

        # Penalize x axis base linear velocity forward
        lin_vel_x_reward  = -0.4 * torch.square(torch.relu(env.root_states[:, 7]))

        # Penalize torques
        torques_reward = -0.0005 * torch.sum(torch.square(env.torques), dim=1)

        # Penalize dof accelerations
        dof_acc_reward = -1.0e-7 * torch.sum(torch.square((env.last_dof_vel - env.dof_vel) / env.dt), dim=1)

        # # Penalize dofs close to limits
        # out_of_limits = -(env.dof_pos - env.dof_pos_limits[:, 0]).clip(max=0.)
        # out_of_limits += (env.dof_pos - env.dof_pos_limits[:, 1]).clip(min=0.)
        # dof_pos_limits_reward = -7.0 * torch.sum(out_of_limits, dim=1)

        # Penalize base torso hitting the ground
        base_hit_ground = torch.any(torch.norm(env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 0.1, dim=1)  # >1.0 originally
        base_hit_ground_reward = -20.0 * base_hit_ground  # 10

        # Penalize non flat base orientation
        gravity_reward = -0.01 * torch.sum(torch.square(env.projected_gravity[:, :2]), dim=1)

        # Reward for highest z position reached in an episode
        height_history = env.height_history_buf
        height_history_reward = 1.0 * height_history  # 0.5

        # Combine reward components to compute the total reward in this step
        total_reward = (roll_rate_reward + yaw_rate_reward + roll_reward + pitch_reward + yaw_reward + lin_vel_z_reward + lin_vel_x_reward +
                        torques_reward + dof_acc_reward+ base_hit_ground_reward + gravity_reward + height_history_reward)


        # Debug information
        reward_components = {
            "roll_rate_reward": roll_rate_reward,
            "yaw_rate_reward": yaw_rate_reward,
            "roll_reward": roll_reward,
            "pitch_reward": pitch_reward,
            "yaw_reward": yaw_reward,
            "lin_vel_z_reward": lin_vel_z_reward,
            "lin_vel_x_reward": lin_vel_x_reward,
            "torques_reward": torques_reward,
            "dof_acc_reward": dof_acc_reward,
            "base_hit_ground_reward": base_hit_ground_reward,
            "gravity_reward": gravity_reward,
            "height_history_reward": height_history_reward
        }
        return total_reward, reward_components

    # Success criteria as forward velocity
    def compute_success(self):
        # Reward for highest z position reached in an episode
        height_history = self.env.height_history_buf
        height_history_reward = 5 * height_history  # 0.5

        # Penalize early termination to discourage falling or instability
        early_termination_penalty = -1.0 * self.env.reset_buf * ~self.env.time_out_buf  # penalty if the episode resets early

        return height_history_reward + early_termination_penalty