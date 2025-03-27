import time
import os
from collections import deque
import statistics

from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent, Transformer
from rsl_rl.env import VecEnv
from pathlib import Path
from openai import OpenAI
import ast
import time
import random

from rsl_rl.algorithms import PrefPredTransformerWaveTrain


# class OnPolicyPrefRunner7 is coly from the class OnPolicyPrefRunner4
# The only difference is that we process the collected traj pairs before getting prefenrce labels
# We keep 60% of the original pairs, and select random new pairs in this traj pool to fill in the other 40%.
class OnPolicyPrefWaveRunner:

    def __init__(self,
                 env: VecEnv,
                 #  prompt_init_task: str,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCritic = actor_critic_class(self.env.num_obs,
                                                       num_critic_obs,
                                                       self.env.num_actions,
                                                       **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs],
                              [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        # for preference labels
        self.pref_load_error_value = 4  # use the number 4 to indicate that some error happens and fails to load the preference value
        self.current_file_path = Path(__file__).resolve()
        self.project_root = self.current_file_path.parents[4]
        self.gpt_key_file_path = self.project_root / 'api_key/openai_api_key.txt'  # 'api_key/openai_api_key.txt'
        self.OPENAI_API_KEY = self.get_secret_key()
        # region tmp
        self.client = OpenAI(api_key=self.OPENAI_API_KEY)
        # endregion

        # Initialize LLM parameters
        self.model_name = 'gpt-4o-mini'  # gpt-4o-mini, gpt-4o-2024-11-20
        # self.model_name = 'gpt-3.5-turbo'
        self.temperature = 1.0
        self.n_samples = 15  # 1， 3
        self.prompt_batch_size = 5

        self.track_price = 0

        # # Initialize the conversation with a system message to set up the assistant's behavior
        if env.cfg.terrain.terrain_type == 'curriculum_wave':
            self.init_system_file_path = self.project_root / 'go2wave_prompt/wave_forward_initialize_system.txt'
        else:
            print("@@@ Unexpected: Unsupported terrain type.")

        print(f"$$$ The initial system file path is: {self.init_system_file_path}")

        self.initial_system = self.file_to_string(self.init_system_file_path)
        self.pref_pred_models_root = self.project_root / self.log_dir

        _, _ = self.env.reset()

    def select_random_pairs(self, traj_num, pairs_num):
        """
        Randomly selects unique pairs of trajectory IDs.

        Parameters:
        - traj_num (int): Total number of trajectories.
        - pairs_num (int): Number of unique pairs to select.

        Returns:
        - pairs (np.ndarray): An array of shape (pairs_num, 2) containing the selected trajectory pairs.
        """
        if pairs_num > traj_num * (traj_num - 1) // 2:
            raise ValueError("Number of pairs requested exceeds total possible unique pairs.")

        pairs = set()  # Use a set to ensure uniqueness

        while len(pairs) < pairs_num:
            # Randomly select two distinct trajectories
            i = np.random.randint(0, traj_num)
            j = np.random.randint(0, traj_num)

            # Ensure i < j to avoid duplicates like (1, 2) and (2, 1)
            if i != j:
                pair = (min(i, j), max(i, j))
                pairs.add(pair)  # Set guarantees unique pairs

        # Convert the set of pairs to a numpy array
        pairs_array = np.array(list(pairs))
        return pairs_array

    def get_secret_key(self):
        try:
            with open(self.gpt_key_file_path, 'r') as file:
                secret = file.read().strip()  # Read the file content and remove any surrounding whitespace
            return secret
        except FileNotFoundError:
            return "Error: The file was not found."
        except Exception as e:
            return f"An error occurred: {e}"

    def file_to_string(self, filename):
        with open(filename, 'r', errors="ignore") as file:
            return file.read()

    def pref_value_string2tensor(self, pref_str: str, prompt_batch_size: int, device):
        try:
            # Attempt to parse the GPT response
            pref_values = ast.literal_eval(pref_str)

            # Validate that the result is a list
            if not isinstance(pref_values, list):
                raise ValueError("The response is not a list.")

            # Validate that the list has exactly 5 elements
            if len(pref_values) != prompt_batch_size:
                raise ValueError(f"The response does not contain exactly {prompt_batch_size} elements.")

            # Validate that all elements are integers and in the range 0-3
            valid_numbers = {0, 1, 2, 3}
            for num in pref_values:
                if not isinstance(num, int):
                    raise ValueError(f"Invalid type {type(num)}; all elements must be integers.")
                if num not in valid_numbers:
                    raise ValueError(f"Invalid value {num}; elements must be 0, 1, 2, or 3.")

            # All validations passed; proceed to assign the values
            pref_values_tensor = torch.tensor(pref_values, dtype=torch.int, device=device)
            return pref_values_tensor

        except Exception as e:
            # Handle exceptions by logging and skipping assignment
            print(f"Invalid GPT response at index: {pref_str}")
            print(f"Error: {e}")
            pref_values_tensor = torch.zeros(prompt_batch_size, dtype=torch.int, device=device)
            pref_values_tensor.fill_(self.pref_load_error_value)
            return pref_values_tensor

    def compute_column_mode(self, pref_values_tensor_pool):
        """
        Compute the mode for each column of the input tensor, with random tie-breaking for ties.

        Parameters:
        - pref_values_tensor_pool (torch.Tensor): A 2D tensor of shape (n_samples, prompt_batch_size)
          containing rows of preference values.

        Returns:
        - torch.Tensor: A 1D tensor of shape (prompt_batch_size,) containing the mode of each column,
          with ties broken randomly.
        """
        # Compute the mode using torch.mode
        values, _ = torch.mode(pref_values_tensor_pool, dim=0)

        # Handle ties
        for col in range(pref_values_tensor_pool.size(1)):  # Iterate over columns
            # Get unique values and their counts for the column
            col_values, col_counts = torch.unique(pref_values_tensor_pool[:, col], return_counts=True)

            # Find the maximum count
            max_count = col_counts.max()

            # Check for ties
            candidates = col_values[col_counts == max_count]
            if candidates.numel() > 1:  # Tie detected
                # Randomly select one of the candidates
                rand_idx = torch.randint(0, candidates.numel(), (1,))
                values[col] = candidates[rand_idx]

        return values

    def gpt_generate_pref_labels(self, data: dict):
        # Access the tensors
        cur_start = time.time()
        base_pos_buf = data['base_pos_buf']
        rpy_buf = data['rpy_buf']
        base_lin_vel_buf = data['base_lin_vel_buf']
        base_ang_vel_buf = data['base_ang_vel_buf']
        feet_contacts_buf = data['feet_contacts_buf']
        commands_buf = data['commands_buf']
        pref_label_buf = torch.zeros(len(commands_buf), dtype=torch.int, device=self.env.device, requires_grad=False)
        pref_label_buf.fill_(self.pref_load_error_value)

        base_pos_buf = base_pos_buf.cpu().numpy()
        rpy_buf = rpy_buf.cpu().numpy()
        base_lin_vel_buf = base_lin_vel_buf.cpu().numpy()
        base_ang_vel_buf = base_ang_vel_buf.cpu().numpy()
        feet_contacts_buf = feet_contacts_buf.cpu().numpy()
        commands_buf = commands_buf.cpu().numpy()

        conversation_history = []
        assert len(commands_buf) % self.prompt_batch_size == 0
        batch_prompt_iter_num = len(commands_buf) // self.prompt_batch_size

        float_formatter = lambda x: f"{x:.3f}"
        dummy_large_num = 1000000

        for i in range(batch_prompt_iter_num):  # e.g. batch_prompt_iter_num == 20
            conversation_history = [{"role": "system", "content": self.initial_system}]
            user_chat = ''
            for j in range(self.prompt_batch_size):  # e.g. prompt_batch_size == 5
                user_chat += f'Here is trajectories pair {j}: \n'
                for id_in_pair in range(2):
                    user_chat += f'For trajectory {id_in_pair} in trajectories pair {j}: \n'
                    user_chat += 'The "commands" of forward velocity in this trajectory are: \n'
                    user_chat += np.array2string(commands_buf[i * self.prompt_batch_size + j][id_in_pair][:, 0],
                                                 formatter={'float_kind': float_formatter}, separator=', ',
                                                 threshold=dummy_large_num, max_line_width=dummy_large_num)
                    user_chat += '\n'
                    user_chat += 'The "base linear velocity" in this trajectory is: \n'
                    user_chat += np.array2string(
                        base_lin_vel_buf[i * self.prompt_batch_size + j][id_in_pair],
                        formatter={'float_kind': float_formatter},
                        max_line_width=dummy_large_num,  # Large number to prevent line wrapping
                        threshold=dummy_large_num,  # Prevent abbreviation
                        separator=' '
                    )
                    user_chat += '\n'
                    user_chat += 'The "base angular velocity" in this trajectory is: \n'
                    user_chat += np.array2string(
                        base_ang_vel_buf[i * self.prompt_batch_size + j][id_in_pair],
                        formatter={'float_kind': float_formatter},
                        max_line_width=dummy_large_num,  # Large number to prevent line wrapping
                        threshold=dummy_large_num,  # Prevent abbreviation
                        separator=' '
                    )
                    user_chat += '\n'
                    user_chat += 'The "base height" in this trajectory is: \n'
                    user_chat += np.array2string(base_pos_buf[i * self.prompt_batch_size + j][id_in_pair][:, 2],
                                                 formatter={'float_kind': float_formatter}, separator=', ',
                                                 threshold=dummy_large_num, max_line_width=dummy_large_num)
                    user_chat += '\n'
                    user_chat += 'The "base angular orientation" in this trajectory is: \n'
                    user_chat += np.array2string(
                        rpy_buf[i * self.prompt_batch_size + j][id_in_pair],
                        formatter={'float_kind': float_formatter},
                        max_line_width=dummy_large_num,  # Large number to prevent line wrapping
                        threshold=dummy_large_num,  # Prevent abbreviation
                        separator=' '
                    )
                    user_chat += '\n'
                    user_chat += 'The "feet contacts" in this trajectory is: \n'
                    user_chat += np.array2string(
                        feet_contacts_buf[i * self.prompt_batch_size + j][id_in_pair],
                        formatter={'float_kind': float_formatter},
                        max_line_width=dummy_large_num,  # Large number to prevent line wrapping
                        threshold=dummy_large_num,  # Prevent abbreviation
                        separator=' '
                    )
                    user_chat += '\n'
            # print("######################")
            user_chat += 'Now please provide preference feedback on these 5 pairs of trajectories according to the instructions in the initial system prompt.\n'
            user_chat += 'Please give response with only one list of 5 preference values, e.g., [0,0,1,2,3]. Do not provide any other text such as your comments or thoughts. The preference value number can only be 0, 1, 2, or 3.'
            conversation_history.append({"role": "user", "content": user_chat})
            # print(conversation_history)
            # Make a call to the GPT model with multiple responses
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history,
                temperature=self.temperature,
                n=self.n_samples
            )
            # Extract the responses from the assistant (GPT)
            gpt_replies = [choice.message.content for choice in response.choices]

            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            input_cost = (prompt_tokens * 0.15) / 1_000_000
            output_cost = (completion_tokens * 0.6 * self.n_samples) / 1_000_000
            self.track_price += input_cost + output_cost

            print(f"$$$ Cumulative price is {self.track_price}")

            pref_values_tensor_pool = torch.zeros(self.n_samples, self.prompt_batch_size, dtype=torch.int,
                                                  device=pref_label_buf.device)

            for response_id in range(self.n_samples):
                # print(f"the gpt response {response_id} is:")
                # print(gpt_replies[response_id])
                pref_values_tensor = self.pref_value_string2tensor(pref_str=gpt_replies[response_id],
                                                                   prompt_batch_size=self.prompt_batch_size,
                                                                   device=pref_label_buf.device)
                pref_values_tensor_pool[response_id] = pref_values_tensor

            pref_values_tensor_mode = self.compute_column_mode(pref_values_tensor_pool)
            pref_label_buf[
            i * self.prompt_batch_size: i * self.prompt_batch_size + self.prompt_batch_size] = pref_values_tensor_mode
            # time.sleep(0.4)  # 0.2 is OK for the lab account but too fast for pj account
            time.sleep(0.8)

        data['pref_label_buf'] = pref_label_buf
        print("The preference labels are:")
        print(data['pref_label_buf'])

        cur_end = time.time()
        print(f"Query time is {cur_end - cur_start}s")
        return data

    def predict_preference_rewards(self, pref_pred_models_list, current_state, main_args):
        # pref_pred_models_list is a list that stores the bag of preference prediction networks
        # main_args is the args defined in the training scripts.
        # It comes from the def get_args() function in the helpers.py
        # current_state has size [batch_size, feature_dim]
        # unsqueeze the current_state to have a seq_length dimension, and seq_length == 1
        current_state = current_state.unsqueeze(1)  # [batch_size, 1, feature_dim]

        # Predict from each model
        rewards = []
        with torch.no_grad():
            for model in pref_pred_models_list:
                raw_rewards = model(current_state)  # [traj_length, 1, 1]
                raw_rewards = raw_rewards.squeeze()  # [traj_length, ]
                rewards.append(raw_rewards)

        # Average the three normalized rewards
        final_rewards = torch.stack(rewards).mean(dim=0)
        # main_args.pref_scale: a float to scale the preference rewards (e.g. 1.0).
        final_rewards = main_args.pref_scale * final_rewards
        return final_rewards.to(self.device)

    def prepare_data_dict_for_pref_predictor(self, current_iteration, main_args):
        # e.g. range(4599, 5000, 100)
        # e.g. range(99, 500, 100), 500 - 500 + 100 -1 == 99
        # e.g. range(299, 700, 100), 700 - 500 + 100 -1 == 299. 299, 399, 499, 599, 699
        data_start_eps = current_iteration - main_args.pref_pred_train_data_period_eps + self.env.pref_buf_interval - 1
        assert data_start_eps > 98  # it should start from 99
        file_indices = range(data_start_eps, current_iteration, 100)
        filenames = [f"traj_pairs_{i}.pt" for i in file_indices]

        # Keys in the dataset
        data_keys = [
            'base_pos_buf',
            'rpy_buf',
            'base_lin_vel_buf',
            'base_ang_vel_buf',
            'feet_contacts_buf',
            'commands_buf',
            'actions_buf',
            'obs_pairs_buf',
            'pref_label_buf'
        ]

        # Initialize a dictionary to store lists of tensors for each key
        dataset_dict = {key: [] for key in data_keys}

        # Load each dataset and append the tensors
        for fname in filenames:
            data_path = os.path.join(self.log_dir, fname)
            loaded_data = torch.load(data_path)
            for key in data_keys:
                dataset_dict[key].append(loaded_data[key])

        # Concatenate along dimension 0 for each key
        for key in data_keys:
            dataset_dict[key] = torch.cat(dataset_dict[key], dim=0)

        return dataset_dict

    def update_pref_train_data_queue(self, pref_train_data, new_pref_train_data):
        """
        Update the pref_train_data in a queue manner.
        Push the new_pref_train_data at the end, and pop out the oldest data in the front.
        """
        # Assert that both dictionaries have the same keys
        assert set(pref_train_data.keys()) == set(
            new_pref_train_data.keys()), "Keys in pref_train_data and new_pref_train_data must match."
        # e.g. new_data_length == 100
        new_data_length = len(new_pref_train_data['pref_label_buf'])

        for key in pref_train_data.keys():
            pref_train_data[key] = torch.cat((pref_train_data[key][new_data_length:], new_pref_train_data[key]), dim=0)

        print("the preference predictors training data is updated as a queue.")

        return pref_train_data

    def create_mixed_data_pairs(
            self,
            input_data: dict,
            preserve_fraction: float = 0.6,
            seed: int = None
    ) -> dict:
        """
        Creates a new data-pairs dictionary from an old one:
          (1) preserves 60% of old pairs,
          (2) creates 40% new pairs from old trajectories that did not appear together.

        This version supports different feature_dim for different keys, e.g.:
          - base_pos_buf: shape (pair_num, 2, traj_len, 3)
          - feet_contacts_buf: shape (pair_num, 2, traj_len, 4)
          etc.

        Args:
            input_data (dict): A dictionary whose values are all tensors of shape
              (pair_num, 2, traj_len, feature_dim_k). For example:
                  {
                    'base_pos_buf': (pair_num, 2, traj_len, 3),
                    'feet_contacts_buf': (pair_num, 2, traj_len, 4),
                    ...
                  }
            preserve_fraction (float): Fraction (between 0 and 1) of existing pairs
              to preserve (e.g., 0.6 means keep 60% of original pairs).
            seed (int, optional): Random seed for reproducibility. If None, no seed is set.

        Returns:
            new_data (dict): A new dictionary with the same keys, each of shape
              (pair_num, 2, traj_len, feature_dim_k), where:
                - The first `preserve_fraction * pair_num` come from the original data.
                - The remaining pairs are newly formed from individual
                  trajectories that did not appear together in the old dataset.
        """
        # --------------------------------------
        # 1) Optional seeding for reproducibility
        # --------------------------------------
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

        # --------------------------------------
        # 2) Basic checks and gather shape info
        # --------------------------------------
        # We need to ensure *all* keys have the same (pair_num, 2, traj_len),
        # but we allow different feature_dim for each key.
        all_keys = list(input_data.keys())
        if len(all_keys) == 0:
            raise ValueError("input_data cannot be an empty dictionary.")

        # Check the main shape from the first key
        first_key = all_keys[0]
        # e.g. (pair_num, 2, traj_len, fdim)
        first_shape = input_data[first_key].shape
        if len(first_shape) != 4:
            raise ValueError(f"All tensors must have 4 dimensions, got shape {first_shape} for key {first_key}.")
        pair_num, two_, traj_len, _ = first_shape
        if two_ != 2:
            raise ValueError(f"The second dimension must be 2, but got {two_}.")

        # Verify all other keys share the same (pair_num, 2, traj_len).
        for k in all_keys[1:]:
            shape_k = input_data[k].shape
            if len(shape_k) != 4:
                raise ValueError(f"Key {k} has shape {shape_k}; all must have 4 dims.")
            if shape_k[0] != pair_num or shape_k[1] != 2 or shape_k[2] != traj_len:
                raise ValueError(
                    f"Key {k} has shape {shape_k}, but expected (pair_num={pair_num}, 2, traj_len={traj_len}, ...)."
                )

        # --------------------------------------
        # 3) Determine how many pairs to preserve vs. create
        # --------------------------------------
        num_preserve = int(round(preserve_fraction * pair_num))
        num_new = pair_num - num_preserve

        # --------------------------------------
        # 4) Flatten each tensor from shape:
        #    (pair_num, 2, traj_len, feature_dim_k)
        #    to
        #    (pair_num * 2, traj_len, feature_dim_k).
        # --------------------------------------
        flattened_data = {}
        for k in all_keys:
            # shape: (pair_num, 2, traj_len, feature_dim_k)
            pk, two, tl, fdim = input_data[k].shape
            # Flatten so each trajectory is a single item
            flattened_data[k] = input_data[k].view(pk * 2, tl, fdim)

        # --------------------------------------
        # 5) Prepare the new_data output dictionary
        #    We will fill it with shape (pair_num, 2, traj_len, feature_dim_k)
        # --------------------------------------
        new_data = {}
        for k in all_keys:
            new_data[k] = torch.empty_like(input_data[k])

        # --------------------------------------
        # 6) Randomly pick the preserve_indices
        #    and fill the first part of new_data with them
        # --------------------------------------
        all_pair_indices = list(range(pair_num))  # 0..(pair_num-1)
        preserve_indices = random.sample(all_pair_indices, num_preserve)

        for out_idx, old_pair_idx in enumerate(preserve_indices):
            for k in all_keys:
                new_data[k][out_idx] = input_data[k][old_pair_idx]

        # --------------------------------------
        # 7) Record which pairs (i, j) existed in the old dataset
        #    i.e. (2*pair_idx, 2*pair_idx+1) for pair_idx in [0..pair_num-1]
        # --------------------------------------
        old_pairs = set((2 * i, 2 * i + 1) for i in range(pair_num))

        # --------------------------------------
        # 8) Create brand-new pairs from the pool of flattened trajectories
        #    We want `num_new` such pairs that are not in old_pairs
        # --------------------------------------
        all_traj_indices = list(range(2 * pair_num))  # 0..(2*pair_num - 1)
        new_pairs = []

        while len(new_pairs) < num_new:
            i, j = random.sample(all_traj_indices, 2)
            # Ensure (i, j) is stored in sorted order so that (0,1) == (1,0)
            pair_tuple = tuple(sorted((i, j)))
            if pair_tuple not in old_pairs and pair_tuple not in new_pairs:
                new_pairs.append(pair_tuple)

        # --------------------------------------
        # 9) Fill in the new_data with these new pairs
        #    in the last part of the array
        # --------------------------------------
        for idx, (i, j) in enumerate(new_pairs):
            out_idx = num_preserve + idx
            for k in all_keys:
                # shape for each new pair: (2, traj_len, fdim_k)
                traj_i = flattened_data[k][i]  # shape: (traj_len, fdim_k)
                traj_j = flattened_data[k][j]  # shape: (traj_len, fdim_k)
                new_data[k][out_idx] = torch.stack([traj_i, traj_j], dim=0)

        return new_data

    def curr_learn_w_pred(self, num_learning_iterations, main_args, init_at_random_ep_len=False):
        # the the config parameters to be consistent first
        assert self.env.cfg.env.num_steps_per_env == self.cfg["num_steps_per_env"]

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        difficulty_level = 0

        current_pair_episode_batch_start_id = 0
        current_pair_id = 0
        pref_pred_models_dict = None
        pref_pred_models_list = None
        pref_pred_trainer = None

        init_base_pos_buf = torch.zeros(500, 2, 24, len(self.env.base_pos[1]), device=self.env.device,
                                        dtype=torch.float, requires_grad=False)
        init_rpy_buf = torch.zeros(500, 2, 24, len(self.env.rpy[1]), device=self.env.device, dtype=torch.float,
                                   requires_grad=False)
        init_base_lin_vel_buf = torch.zeros(500, 2, 24, len(self.env.base_lin_vel[1]), device=self.env.device,
                                            dtype=torch.float, requires_grad=False)
        init_base_ang_vel_buf = torch.zeros(500, 2, 24, len(self.env.base_ang_vel[1]), device=self.env.device,
                                            dtype=torch.float, requires_grad=False)
        init_feet_contacts_buf = torch.zeros(500, 2, 24, len(self.env.feet_indices), dtype=torch.bool,
                                             device=self.env.device, requires_grad=False)
        init_commands_buf = torch.zeros(500, 2, 24, self.env.cfg.commands.num_commands, device=self.env.device,
                                        dtype=torch.float, requires_grad=False)
        init_actions_buf = torch.zeros(500, 2, 24, self.env.num_actions, device=self.env.device, dtype=torch.float,
                                       requires_grad=False)
        init_obs_pairs_buf = torch.zeros(500, 2, 24, len(self.env.obs_buf[1]), device=self.env.device,
                                         dtype=torch.float, requires_grad=False)

        for it in range(self.current_learning_iteration, tot_iter):
            # print("$$$")
            # print(f"This is epoch {it}")
            # print("$$$")
            start = time.time()
            # if main_args.save_pairs:
            if it < 100:
                this_episode_selected_pairs_ids = self.select_random_pairs(traj_num=self.env.num_envs, pairs_num=5)
            else:
                this_episode_selected_pairs_ids = self.select_random_pairs(traj_num=self.env.num_envs,
                                                                           pairs_num=self.env.num_pref_pairs_per_episode)

            # e.g. pref_pred_update_period_eps==100
            # update when it = 0, 100, ..., 500, 600, ..., 4900
            if it % main_args.pref_pred_update_period_eps == 0 and it > 0:
                pref_pred_trainer = None  # set this variable back to None first. Could be redundant but make me feel safe.
                # pred_models_0.pt, pred_models_100.pt, ... means the pred_models are trained at episode 0 (100)
                # and it is trained with 'main_args.num_pref_pairs_total_train' pairs of data
                pref_pred_trainer = PrefPredTransformerWaveTrain(org_data=pref_train_data,
                                                              device=self.device,
                                                              # this is the rl_device from the main_args
                                                              save_models_path=self.log_dir + f'/pred_models_{it}.pt',
                                                              save_models=True,
                                                              # save the preference predictor networks
                                                              pool_models_num=main_args.pref_pred_pool_models_num,
                                                              select_models_num=main_args.pref_pred_select_models_num,
                                                              input_mode=main_args.pref_pred_input_mode,
                                                              # mode 0: state(15)
                                                              batch_size=main_args.pref_pred_batch_size,  # 256
                                                              transformer_embed_dim=main_args.pref_pred_transformer_embed_dim,
                                                              seq_length=main_args.pref_pred_seq_length,
                                                              epsilon=0.1,
                                                              lr=9e-4,  # 1e-3
                                                              weight_decay=1e-4,
                                                              epochs=main_args.pref_pred_epoch)
                _, _, _, _ = pref_pred_trainer.train()  # train the three MLP networks

            traj_pref_rewards_buf = torch.zeros(self.num_steps_per_env, dtype=torch.float, device=self.device)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)

                    # region tmp
                    current_x_command = self.env.commands[:, 0].unsqueeze(1)  # torch.Size([num_envs, 1])
                    current_base_height = self.env.base_pos[:, 2].unsqueeze(1)  # torch.Size([num_envs, 1])
                    current_feet_contacts = self.env.contact_forces[:, self.env.feet_indices,
                                            2] > 1.  # torch.Size([num_envs, 4])
                    # the feature_dim of the current_state is 15
                    current_state = torch.cat((current_x_command, self.env.base_lin_vel, self.env.base_ang_vel,
                                               current_base_height, self.env.rpy, current_feet_contacts), dim=1).to(
                        self.device)  # torch.Size([num_envs, feature_dim])
                    # pref_rewards shape (traj_length, )
                    if pref_pred_trainer is not None:
                        pref_rewards = pref_pred_trainer.predict_batch_reward(current_state)
                        # mean_step_pref_reward shape (1, )
                        mean_step_pref_reward = pref_rewards.mean()
                        traj_pref_rewards_buf[i] = mean_step_pref_reward.item()
                    # endregion

                    # if main_args.save_pairs:
                    if pref_pred_trainer is not None:
                        for k in range(len(this_episode_selected_pairs_ids)):
                            current_pair_id = current_pair_episode_batch_start_id + k
                            for first_or_second_in_pair in range(2):  # first_or_second_in_pair will be 0 or 1
                                self.env.base_pos_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                                self.env.base_pos[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                                # Subtract terrain height ==================
                                t_height = self.env._get_terrain_heights(
                                    x=self.env.base_pos_buf[current_pair_id][first_or_second_in_pair][i][0],
                                    y=self.env.base_pos_buf[current_pair_id][first_or_second_in_pair][i][1])
                                self.env.base_pos_buf[current_pair_id][first_or_second_in_pair][i][
                                    2] -= t_height.squeeze()
                                # ==================
                                self.env.rpy_buf[current_pair_id][first_or_second_in_pair][i][:] = self.env.rpy[
                                                                                                       this_episode_selected_pairs_ids[
                                                                                                           k][
                                                                                                           first_or_second_in_pair]][
                                                                                                   :]
                                self.env.base_lin_vel_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                                self.env.base_lin_vel[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                                self.env.base_ang_vel_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                                self.env.base_ang_vel[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                                self.env.feet_contacts_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                                self.env.contact_forces[this_episode_selected_pairs_ids[k][
                                    first_or_second_in_pair], self.env.feet_indices, 2] > 1.
                                self.env.commands_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                                self.env.commands[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                                # also save the actions in the trajectory
                                self.env.actions_buf[current_pair_id][first_or_second_in_pair][i][:] = actions[
                                                                                                           this_episode_selected_pairs_ids[
                                                                                                               k][
                                                                                                               first_or_second_in_pair]][
                                                                                                       :]
                                self.env.obs_pairs_buf[current_pair_id][first_or_second_in_pair][i][:] = obs[
                                                                                                             this_episode_selected_pairs_ids[
                                                                                                                 k][
                                                                                                                 first_or_second_in_pair]][
                                                                                                         :]





                    else:
                        # print("$$$ Enter new logic of collection")
                        for newk, pair_iter in enumerate(this_episode_selected_pairs_ids):
                            current_pair_id = current_pair_episode_batch_start_id + newk
                            for idx_in_pair in range(2):
                                init_base_pos_buf[current_pair_id][idx_in_pair][i][:] = self.env.base_pos[
                                                                                            pair_iter[idx_in_pair]][:]
                                t_height = self.env._get_terrain_heights(
                                    x=init_base_pos_buf[current_pair_id][idx_in_pair][i][0],
                                    y=init_base_pos_buf[current_pair_id][idx_in_pair][i][1])
                                init_base_pos_buf[current_pair_id][idx_in_pair][i][2] -= t_height.squeeze()

                                init_rpy_buf[current_pair_id][idx_in_pair][i][:] = self.env.rpy[pair_iter[idx_in_pair]][
                                                                                   :]
                                init_base_lin_vel_buf[current_pair_id][idx_in_pair][i][:] = self.env.base_lin_vel[
                                                                                                pair_iter[idx_in_pair]][
                                                                                            :]
                                init_base_ang_vel_buf[current_pair_id][idx_in_pair][i][:] = self.env.base_ang_vel[
                                                                                                pair_iter[idx_in_pair]][
                                                                                            :]
                                init_feet_contacts_buf[current_pair_id][idx_in_pair][i][:] = self.env.contact_forces[
                                                                                                 pair_iter[
                                                                                                     idx_in_pair], self.env.feet_indices, 2] > 1.
                                init_commands_buf[current_pair_id][idx_in_pair][i][:] = self.env.commands[
                                                                                            pair_iter[idx_in_pair]][:]
                                init_actions_buf[current_pair_id][idx_in_pair][i][:] = actions[pair_iter[idx_in_pair]][
                                                                                       :]
                                init_obs_pairs_buf[current_pair_id][idx_in_pair][i][:] = obs[pair_iter[idx_in_pair]][:]

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    # region tmp
                    if pref_pred_trainer is not None:
                        # rewards += main_args.pref_scale * pref_rewards
                        rewards = main_args.dense_reward_scale * rewards + main_args.pref_scale * pref_rewards
                    # endregion

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # region essential code
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # if main_args.save_pairs:
                current_pair_episode_batch_start_id += len(
                    this_episode_selected_pairs_ids)  # add up the episode_batch_start_id to next batch

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_episode_pref_reward = traj_pref_rewards_buf.mean().item()
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                success_score = self.log_return_success_w_pref(locals(), difficulty_level)
                # self.log_with_pref(locals())
                if it < 501:  # 19, 39,..., 499
                    if (it + 1) % 20 == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                else:
                    if (it + 1) % self.save_interval == 0:
                        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                self.log_render_only(locals())
            # endregion

            if (it + 1) % self.env.pref_buf_interval == 0:
                if it < 110:
                    print("!!! Enter new logic of making dataset")
                    assert current_pair_id == 499
                    pref_train_data = {
                        'base_pos_buf': init_base_pos_buf,
                        'rpy_buf': init_rpy_buf,
                        'base_lin_vel_buf': init_base_lin_vel_buf,
                        'base_ang_vel_buf': init_base_ang_vel_buf,
                        'feet_contacts_buf': init_feet_contacts_buf.int(),
                        'commands_buf': init_commands_buf,
                        'actions_buf': init_actions_buf,
                        'obs_pairs_buf': init_obs_pairs_buf
                    }
                    print("the initial data states are collected")

                    pref_train_data = self.gpt_generate_pref_labels(pref_train_data)
                    print("the initial data labels are generated")

                    # Save the dictionary to a .pt file
                    torch.save(pref_train_data, os.path.join(self.log_dir, 'traj_pairs_initial500.pt'))

                    current_pair_episode_batch_start_id = 0
                    current_pair_id = 0



                else:
                    # print(f"the current_pair_id is {current_pair_id}")
                    # print(f"the self.env.num_pairs is {self.env.num_pairs}")
                    # if main_args.save_pairs:
                    assert current_pair_id == (self.env.num_pairs - 1)
                    # Create a dictionary of the data
                    new_pref_train_data = {
                        'base_pos_buf': self.env.base_pos_buf,
                        'rpy_buf': self.env.rpy_buf,
                        'base_lin_vel_buf': self.env.base_lin_vel_buf,
                        'base_ang_vel_buf': self.env.base_ang_vel_buf,
                        'feet_contacts_buf': self.env.feet_contacts_buf.int(),
                        # save the int values instead of bool values
                        'commands_buf': self.env.commands_buf,
                        'actions_buf': self.env.actions_buf,
                        'obs_pairs_buf': self.env.obs_pairs_buf
                    }
                    new_pref_train_data = self.create_mixed_data_pairs(input_data=new_pref_train_data)

                    new_pref_train_data = self.gpt_generate_pref_labels(new_pref_train_data)

                    # Save the dictionary to a .pt file
                    torch.save(new_pref_train_data, os.path.join(self.log_dir, 'traj_pairs_{}.pt'.format(it)))
                    # update the pref_train_data dictionary with the newly collected data in a queue manner
                    pref_train_data = self.update_pref_train_data_queue(pref_train_data=pref_train_data,
                                                                        new_pref_train_data=new_pref_train_data)
                    ###
                    # empty the buffers
                    self.env.base_pos_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                        len(self.env.base_pos[1]), device=self.env.device,
                                                        dtype=torch.float,
                                                        requires_grad=False)
                    self.env.rpy_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                   len(self.env.rpy[1]),
                                                   device=self.env.device, dtype=torch.float, requires_grad=False)
                    self.env.base_lin_vel_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                            len(self.env.base_lin_vel[1]), device=self.env.device,
                                                            dtype=torch.float,
                                                            requires_grad=False)
                    self.env.base_ang_vel_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                            len(self.env.base_ang_vel[1]), device=self.env.device,
                                                            dtype=torch.float,
                                                            requires_grad=False)
                    self.env.feet_contacts_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                             len(self.env.feet_indices), dtype=torch.bool,
                                                             device=self.env.device,
                                                             requires_grad=False)
                    self.env.commands_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                        self.env.cfg.commands.num_commands, device=self.env.device,
                                                        dtype=torch.float,
                                                        requires_grad=False)
                    self.env.actions_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                       self.env.num_actions,
                                                       device=self.env.device, dtype=torch.float, requires_grad=False)
                    self.env.obs_pairs_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                         len(self.env.obs_buf[1]), device=self.env.device,
                                                         dtype=torch.float,
                                                         requires_grad=False)
                    current_pair_episode_batch_start_id = 0
                    current_pair_id = 0  # set the current_pair_id back to 0

            difficulty_level = self.env.xw_update_diff_levels(success_score)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations

    def curr_learn_w_fixed_pred(self, num_learning_iterations, main_args, init_at_random_ep_len=False):
        """
        in this function, we collect state data but don't save it.
        Only input it into the pref_pred_models to get the preference rewards.
        Need an arg to tune the scale of the preference rewards.
        Also need to record the preference rewards value to tensorboard.
        Maybe also need to clip the preference rewards within some range (e.g. [-1.0, 1.0])
        main_args: main_args is the args defined in the training scripts.
        """

        # the the config parameters to be consistent first
        assert self.env.cfg.env.num_steps_per_env == self.cfg["num_steps_per_env"]

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations

        difficulty_level = 0

        current_pair_episode_batch_start_id = 0
        current_pair_id = 0
        pref_pred_models_dict = None
        pref_pred_models_list = None

        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            this_episode_selected_pairs_ids = self.select_random_pairs(traj_num=self.env.num_envs,
                                                                       pairs_num=self.env.num_pref_pairs_per_episode)

            # load a corresponding group of preference predictor networks every 500 episodes
            if it % 100 == 0:
                # pref_pred_models_dict: a dictionary that contains several (e.g.) preference prediction models. Keys are model_{i}.
                pref_pred_models_dict = torch.load(self.pref_pred_models_root / f'pred_models_{it}.pt')
                print(f"$$$ The preference predictor networks are loaded from:")
                print(self.pref_pred_models_root / f'pred_models_{it}.pt')

                pref_pred_models_list = []
                for i in range(3):
                    model = Transformer(input_dim=main_args.state_feature_dim,
                                        transformer_embed_dim=main_args.pref_pred_transformer_embed_dim,
                                        transformer_context_length=main_args.pref_pred_transformer_seq_length,
                                        output_dim=1,
                                        transformer_sinusoidal_embedding=True)
                    model.load_state_dict(pref_pred_models_dict[f"model_{i}"])
                    model.to(self.device)  # Ensure the model is moved to the correct device (rl_device)
                    pref_pred_models_list.append(model)

            # Rollout
            # traj_pref_rewards_buf records the mean_step_pref_reward of all steps in a trajectory
            traj_pref_rewards_buf = torch.zeros(self.num_steps_per_env, dtype=torch.float, device=self.device)
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    current_x_command = self.env.commands[:, 0].unsqueeze(1)  # torch.Size([num_envs, 1])
                    current_base_height = self.env.base_pos[:, 2].unsqueeze(1)  # torch.Size([num_envs, 1])
                    current_feet_contacts = self.env.contact_forces[:, self.env.feet_indices,
                                            2] > 1.  # torch.Size([num_envs, 4])
                    # the feature_dim of the current_state is 15
                    current_state = torch.cat((current_x_command, self.env.base_lin_vel, self.env.base_ang_vel,
                                               current_base_height, self.env.rpy, current_feet_contacts), dim=1).to(
                        self.device)  # torch.Size([num_envs, feature_dim])
                    # pref_rewards shape (traj_length, )
                    pref_rewards = self.predict_preference_rewards(pref_pred_models_list=pref_pred_models_list,
                                                                   current_state=current_state,
                                                                   main_args=main_args)
                    # mean_step_pref_reward shape (1, )
                    mean_step_pref_reward = pref_rewards.mean()
                    traj_pref_rewards_buf[i] = mean_step_pref_reward.item()

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    # add the pref_rewards to the original rewards returned from the self.env.step(actions)
                    # print(f"the rewards shape is: {rewards.shape}")
                    # print(f"the pref_rewards shape is: {pref_rewards.shape}")
                    # rewards += pref_rewards
                    rewards = main_args.dense_reward_scale * rewards + main_args.pref_scale * pref_rewards

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # if self.log_dir is not None:
                    # PJ: I remove the if here. Even I don't save the model, I can still calculate and render the reward
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                # if main_args.save_pairs:
                #     current_pair_episode_batch_start_id += len(
                #         this_episode_selected_pairs_ids)  # add up the episode_batch_start_id to next batch

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_episode_pref_reward = traj_pref_rewards_buf.mean().item()
            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                success_score = self.log_return_success_w_pref(locals(), difficulty_level)
                if (it + 1) % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                self.log_render_only(locals())

            difficulty_level = self.env.xw_update_diff_levels(success_score)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations

    # region original pref learn
    def preference_learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # the the config parameters to be consistent first
        assert self.env.cfg.env.num_steps_per_env == self.cfg["num_steps_per_env"]

        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        current_pair_episode_batch_start_id = 0
        current_pair_id = 0
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            this_episode_selected_pairs_ids = self.select_random_pairs(traj_num=self.env.num_envs,
                                                                       pairs_num=self.env.num_pref_pairs_per_episode)

            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    for k in range(len(this_episode_selected_pairs_ids)):
                        current_pair_id = current_pair_episode_batch_start_id + k
                        for first_or_second_in_pair in range(2):  # first_or_second_in_pair will be 0 or 1
                            self.env.base_pos_buf[current_pair_id][first_or_second_in_pair][i][:] = self.env.base_pos[
                                                                                                        this_episode_selected_pairs_ids[
                                                                                                            k][
                                                                                                            first_or_second_in_pair]][
                                                                                                    :]
                            self.env.rpy_buf[current_pair_id][first_or_second_in_pair][i][:] = self.env.rpy[
                                                                                                   this_episode_selected_pairs_ids[
                                                                                                       k][
                                                                                                       first_or_second_in_pair]][
                                                                                               :]
                            self.env.base_lin_vel_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                            self.env.base_lin_vel[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                            self.env.base_ang_vel_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                            self.env.base_ang_vel[this_episode_selected_pairs_ids[k][first_or_second_in_pair]][:]
                            self.env.feet_contacts_buf[current_pair_id][first_or_second_in_pair][i][:] = \
                            self.env.contact_forces[this_episode_selected_pairs_ids[k][
                                first_or_second_in_pair], self.env.feet_indices, 2] > 1.
                            self.env.commands_buf[current_pair_id][first_or_second_in_pair][i][:] = self.env.commands[
                                                                                                        this_episode_selected_pairs_ids[
                                                                                                            k][
                                                                                                            first_or_second_in_pair]][
                                                                                                    :]
                            # also save the actions in the trajectory
                            self.env.actions_buf[current_pair_id][first_or_second_in_pair][i][:] = actions[
                                                                                                       this_episode_selected_pairs_ids[
                                                                                                           k][
                                                                                                           first_or_second_in_pair]][
                                                                                                   :]
                            self.env.obs_pairs_buf[current_pair_id][first_or_second_in_pair][i][:] = obs[
                                                                                                         this_episode_selected_pairs_ids[
                                                                                                             k][
                                                                                                             first_or_second_in_pair]][
                                                                                                     :]

                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # if self.log_dir is not None:
                    # PJ: I remove the if here. Even I don't save the model, I can still calculate and render the reward
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                current_pair_episode_batch_start_id += len(
                    this_episode_selected_pairs_ids)  # add up the episode_batch_start_id to next batch

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                # PJ: I move this if into the not None judgement
                # PJ: I change the it into (it+1), to save model and trajs in the same episode
                if (it + 1) % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                # PJ: only print to terminal, but not write to tensorboard summary writter
                self.log_render_only(locals())

            if (it + 1) % self.env.pref_buf_interval == 0:
                # print(f"the current_pair_id is {current_pair_id}")
                # print(f"the self.env.num_pairs is {self.env.num_pairs}")
                assert current_pair_id == (self.env.num_pairs - 1)
                if self.cfg["save_pairs"]:
                    # Create a dictionary of the data
                    data = {
                        'base_pos_buf': self.env.base_pos_buf,
                        'rpy_buf': self.env.rpy_buf,
                        'base_lin_vel_buf': self.env.base_lin_vel_buf,
                        'base_ang_vel_buf': self.env.base_ang_vel_buf,
                        'feet_contacts_buf': self.env.feet_contacts_buf.int(),
                        # save the int values instead of bool values
                        'commands_buf': self.env.commands_buf,
                        'actions_buf': self.env.actions_buf,
                        'obs_pairs_buf': self.env.obs_pairs_buf
                    }

                    data = self.gpt_generate_pref_labels(data)

                    # Save the dictionary to a .pt file
                    torch.save(data, os.path.join(self.log_dir, 'traj_pairs_{}.pt'.format(it)))

                self.env.base_pos_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                    len(self.env.base_pos[1]), device=self.env.device,
                                                    dtype=torch.float,
                                                    requires_grad=False)
                self.env.rpy_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                               len(self.env.rpy[1]),
                                               device=self.env.device, dtype=torch.float, requires_grad=False)
                self.env.base_lin_vel_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                        len(self.env.base_lin_vel[1]), device=self.env.device,
                                                        dtype=torch.float,
                                                        requires_grad=False)
                self.env.base_ang_vel_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                        len(self.env.base_ang_vel[1]), device=self.env.device,
                                                        dtype=torch.float,
                                                        requires_grad=False)
                self.env.feet_contacts_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                         len(self.env.feet_indices), dtype=torch.bool,
                                                         device=self.env.device,
                                                         requires_grad=False)
                self.env.commands_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                    self.env.cfg.commands.num_commands, device=self.env.device,
                                                    dtype=torch.float,
                                                    requires_grad=False)
                self.env.actions_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                   self.env.num_actions,
                                                   device=self.env.device, dtype=torch.float, requires_grad=False)
                self.env.obs_pairs_buf = torch.zeros(self.env.num_pairs, 2, self.env.cfg.env.num_steps_per_env,
                                                     len(self.env.obs_buf[1]), device=self.env.device,
                                                     dtype=torch.float,
                                                     requires_grad=False)
                current_pair_episode_batch_start_id = 0
                current_pair_id = 0  # set the current_pair_id back to 0

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        # PJ: I comment out this last saving here. Only save the real last episode model with its real id
        # if self.log_dir is not None:
        #     # PJ: Only save the model when self.log_dir is not None
        #     self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    # endregion

    # region original learn
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(
                        self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # if self.log_dir is not None:
                    # PJ: I remove the if here. Even I don't save the model, I can still calculate and render the reward
                    # Book keeping
                    if 'episode' in infos:
                        ep_infos.append(infos['episode'])
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
                # PJ: I move this if into the not None judgement
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                # PJ: only print to terminal, but not write to tensorboard summary writter
                self.log_render_only(locals())

            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        if self.log_dir is not None:
            # PJ: Only save the model when self.log_dir is not None
            self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    # endregion

    def clip(self, value, min_value, max_value):
        return max(min(value, max_value), min_value)

    def log_with_pref(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # add the preferencd reward here
        self.writer.add_scalar('Preference/pref_reward', locs['mean_episode_pref_reward'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        # add the preference reward to the log_string here
        log_string += (f"""{'Preference reward:':>{pad}} {locs['mean_episode_pref_reward']:.4f}\n""")
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if not self.cfg["silence"]:
            # PJ: This args is to stop printing training logs to the terminal. Used for Hydra multi-process training
            print(log_string)

    def log_return_success_w_pref(self, locs, difficulty_level, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        # =================================
        success_value = None
        # =================================
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                # =================================
                if key == "rew_success":
                    success_value = value
                # =================================
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
            # =================================
            self.writer.add_scalar('Episode/rew_difficulty_level', difficulty_level, locs['it'])
            ep_string += f"""{f'Mean episode rew_difficulty_level:':>{pad}} {difficulty_level:.0f}\n"""
            # =================================
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # add the preferencd reward here
        self.writer.add_scalar('Preference/pref_reward', locs['mean_episode_pref_reward'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        # add the preference reward to the log_string here
        log_string += (f"""{'Preference reward:':>{pad}} {locs['mean_episode_pref_reward']:.4f}\n""")
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if not self.cfg["silence"]:
            print(log_string)

        return success_value

    def log_render_only(self, locs, width=80, pad=35):
        # PJ: only render the info to the terminal, but not write it into the tensorboard
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if not self.cfg["silence"]:
            # PJ: This args is to stop printing training logs to the terminal. Used for Hydra multi-process training
            print(log_string)

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if not self.cfg["silence"]:
            # PJ: This args is to stop printing training logs to the terminal. Used for Hydra multi-process training
            print(log_string)

    def log_return_success(self, locs, difficulty_level, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        success_value = None  # PJ: The success sore to be returned

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                if key == "rew_success":
                    # PJ: The success sore to be returned
                    success_value = value
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

            self.writer.add_scalar('Episode/rew_difficulty_level', difficulty_level, locs['it'])
            ep_string += f"""{f'Mean episode rew_difficulty_level:':>{pad}} {difficulty_level:.0f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        if not self.cfg["silence"]:
            # PJ: This args is to stop printing training logs to the terminal. Used for Hydra multi-process training
            print(log_string)

        return success_value

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
        }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path,
                                 map_location=self.device)  # PJ: I modify this place to add map_location=self.device, otherwise it cannot load correctly sometimes
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference