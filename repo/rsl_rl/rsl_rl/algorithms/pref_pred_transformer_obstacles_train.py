import torch
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from rsl_rl.modules import Transformer
import heapq

# modified from PrefPredTransformerTrain6 class
# It can deal with both the seq_length==1 and seq_length>1 situations, and we can handle different type of input with different mode
# mode 0: state(15), 1: obs(48), 2: state(15)+action(12), 3: obs(48)+action(12), 4: state(15)+obs(48)+action(12)
# don't do normalization, and calculate the mean of the outputs from selected models
# The only difference is that: now we train pool_models_num of predictor networks, don't do normalization, and calculate the mean of the 3 raw outputs from them

class PrefPredTransformerObstaclesTrain:
    def __init__(self, org_data: dict, device, save_models_path, save_models=True, pool_models_num=5, select_models_num=3, input_mode=0,
                 batch_size=256, transformer_embed_dim=64, seq_length=10, epsilon=0.1, lr=1e-3, weight_decay=1e-4, epochs=100):
        self.org_data = org_data  # the dictionary of the original dataset
        self.epsilon = epsilon
        self.lr = lr  # I use the lr 1e-4, which is 0.0001. It is 10 times smaller than the mlp lr
        self.init_weight_decay = weight_decay
        self.epochs = epochs
        # mode 0: state(15), 1: obs(48), 2: state(15)+action(12), 3: obs(48)+action(12), 4: state(15)+obs(48)+action(12)
        self.input_mode = input_mode

        self.state_label_data = self.org_data_to_state_label_data(self.org_data)
        self.batch_size = batch_size  # e.g. 32
        self.seq_length = seq_length  # e.g. 10
        self.transformer_embed_dim = transformer_embed_dim
        self.feature_dim = self.state_label_data['states'].size(-1)  # e.g. 15, 48, 27, 60, 75
        self.traj_length = self.state_label_data['states'].size(-2)  # e.g. 24

        self.select_models_num = select_models_num
        self.pool_models_num = pool_models_num
        self.models = []
        self.device = device
        self.save_models_path = save_models_path
        self.save_models = save_models

        self.seq_length = seq_length
        self.states_queue = torch.zeros((self.batch_size, self.seq_length, self.feature_dim)).float()

    def clear_states_queue(self):
        # clear the queues to zeros
        self.states_queue.zero_()

    def org_data_to_state_label_data(self, org_data: dict):
        """
        in this function, the original data dictionary is transferred in to dictionary with state and labels
        the 'pref_label_buf' doesn't change
        other buffers are concatenated. commands_buf only takes id 0, and base_pos_buf only takes id 2, other buffers take all dims
        e.g. base_pos_buf.shape is (100, 2, 24, 3)
        :return: dictionary with state and labels
        """
        commands_tensor = org_data['commands_buf'][:, :, :, 0].unsqueeze(3)  # torch.Size([100, 2, 24, 1])
        base_lin_vel_tensor = org_data['base_lin_vel_buf']  # torch.Size([100, 2, 24, 3])
        base_ang_vel_tensor = org_data['base_ang_vel_buf']  # torch.Size([100, 2, 24, 3])
        base_height_tensor = org_data['base_pos_buf'][:, :, :, 2].unsqueeze(3)  # torch.Size([100, 2, 24, 1])
        rpy_tensor = org_data['rpy_buf']  # torch.Size([100, 2, 24, 3])
        feet_contacts_tensor = org_data['feet_contacts_buf']  # torch.Size([100, 2, 24, 4])
        pref_label_tensor = org_data['pref_label_buf']
        # concatenate the obs tensors to get the states
        # state_tensor shape: [N, 2, 24, features]
        state_tensor = torch.cat((commands_tensor, base_lin_vel_tensor, base_ang_vel_tensor, base_height_tensor, rpy_tensor, feet_contacts_tensor), dim=3)
        obs_tensor = org_data['obs_pairs_buf']  # torch.Size([100, 2, 24, 48])
        action_tensor = org_data['actions_buf']  # torch.Size([100, 2, 24, 15])
        if self.input_mode == 0:
            input_tensor = state_tensor  # torch.Size([100, 2, 24, 15])
        elif self.input_mode == 1:
            input_tensor = obs_tensor  # torch.Size([100, 2, 24, 48])
        elif self.input_mode == 2:
            input_tensor = torch.cat((state_tensor, action_tensor), dim=3)  # torch.Size([100, 2, 24, 27])
        elif self.input_mode == 3:
            input_tensor = torch.cat((obs_tensor, action_tensor), dim=3)  # torch.Size([100, 2, 24, 60])
        elif self.input_mode == 4:
            input_tensor = torch.cat((state_tensor, obs_tensor, action_tensor), dim=3)  # torch.Size([100, 2, 24, 75])
        else:
            print("input_mode is wrong.")

        assert len(input_tensor) == len(pref_label_tensor)
        # Create a mask for values that are not 3 or 4. Only accept 0, or 1, or 2
        # mask = (pref_label_tensor != 3) & (pref_label_tensor != 4)
        mask = (pref_label_tensor == 0) | (pref_label_tensor == 1) | (pref_label_tensor == 2)
        # Filter out incomparable (3) and response error (4)
        filtered_pref_label_tensor = pref_label_tensor[mask]
        filtered_input_tensor = input_tensor[mask]

        # Create a dictionary of filtered states and labels
        state_label_data = {
            'states': filtered_input_tensor,  # [N, 2, 24, features]
            'labels': filtered_pref_label_tensor,
        }

        return state_label_data

    def label_to_y(self, label):
        # map the int label to float y values for the cross entropy loss
        # the label 2 means equally preferable, so map it to y==0.5
        if label == 0:
            return 1.0
        elif label == 1:
            return 0.0
        elif label == 2:
            return 0.5
        else:
            raise ValueError("Invalid label")

    def create_train_val_split(self, states, labels):
        N = states.shape[0]
        # For training, we keep states in the original shape [N, 2, 24, 15] for convenience
        # We'll do indexing for val/train sets directly on N dimension.
        val_indices = []
        block_size = 5
        for start in range(0, N, block_size):
            end = min(start + block_size, N)
            block_indices = list(range(start, end))
            val_idx = np.random.choice(block_indices, 1)[0]
            val_indices.append(val_idx)
        val_indices = sorted(val_indices)
        val_indices = torch.tensor(val_indices, dtype=torch.long)

        all_indices = torch.arange(N)
        mask = torch.ones(N, dtype=torch.bool)
        mask[val_indices] = False
        train_indices = all_indices[mask]

        train_states = states[train_indices]  # [train_N, 2, 24, 15]
        train_labels = labels[train_indices]
        val_states = states[val_indices]  # [val_N, 2, 24, 15]
        val_labels = labels[val_indices]

        return (train_states, train_labels, val_states, val_labels)

    def create_dataloaders(self, train_states, train_labels, val_states, val_labels):
        # Convert to dataset of tuples
        train_dataset = TensorDataset(train_states, train_labels)
        val_dataset = TensorDataset(val_states, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

    def adjust_weight_decay(self, optimizer, train_loss, val_loss):
        # control the validation loss to be 1.1-1.5 times of the training loss
        # use L2 regularization weight to control it
        ratio = val_loss / train_loss
        for param_group in optimizer.param_groups:
            wd = param_group.get('weight_decay', 0.0)
            if ratio < 1.1:
                wd = wd * 0.9
            elif ratio > 1.5:
                wd = wd * 1.1
            param_group['weight_decay'] = wd

    def traj_to_traj_seq(self, traj):
        """
        Create a sequence window of length `window_size` for each timestep in traj.

        Args:
            traj (Tensor): Shape (batch_size, traj_length, feature_dim)
            window_size (int): Length of the history window (current + previous states), which is self.seq_length

        Returns:
            Tensor: shape (batch_size, traj_length, seq_length, feature_dim)
        """
        batch_size, traj_len, feature_dim = traj.shape

        # Pad at the start along the time dimension
        # Padding format: (pad_feature_left, pad_feature_right, pad_time_left, pad_time_right, pad_batch_left, pad_batch_right)
        traj_padded = F.pad(traj, (0, 0, self.seq_length - 1, 0, 0, 0), mode='constant', value=0)
        # traj_padded shape: (batch_size, seq_len + pad_size, feature_dim)
        # print("in def traj_to_traj_seq, the traj_padded is:")
        # print(traj_padded.shape)

        # For each timestep t in [0..traj_len-1], we want the slice traj_padded[:, t:(t+window_size), :]
        # We'll create an index tensor for this:
        time_indices = torch.arange(traj_len).unsqueeze(-1) + torch.arange(self.seq_length)
        # time_indices shape: (seq_len, window_size)
        # time_indices[i] = [i, i+1, ..., i+window_size-1]

        # Use advanced indexing to gather the windows
        # traj_padded: (batch_size, seq_len+pad_size, feature_dim)
        # time_indices: (seq_len, window_size)
        traj_seq = traj_padded[:, time_indices, :]  # shape: (batch_size, seq_len, window_size, feature_dim)
        # print("in def traj_to_traj_seq, the traj_seq to return is:")
        # print(traj_seq.shape)

        return traj_seq

    def compute_trajectory_rewards(self, model, states):
        """
        states: [batch_size, 2, 24, feature_dim]
        model: PrefPredMlp
        We compute sum of rewards per trajectory:
        - Extract trajectory 0: [batch_size, 24, feature_dim]
        - Extract trajectory 1: [batch_size, 24, feature_dim]
        Feed each step into model and sum.
        """
        batch_size = states.shape[0]
        # trajectory 0 shape: (batch_size, 24, feature_dim)
        traj0 = states[:, 0]  # [batch_size, 24, feature_dim]
        traj1 = states[:, 1]  # [batch_size, 24, feature_dim]
        # print(traj0.size())

        # [batch_size, 24, 15] -> # [batch_size, 24, 10, feature_dim]
        traj0_seq = self.traj_to_traj_seq(traj0)
        traj1_seq = self.traj_to_traj_seq(traj1)

        # Flatten steps for batch processing
        traj0_flat = traj0_seq.reshape(batch_size * self.traj_length, self.seq_length, self.feature_dim)  # [batch_size*traj_length, seq_length, feature_dim]
        traj1_flat = traj1_seq.reshape(batch_size * self.traj_length, self.seq_length, self.feature_dim)  # [batch_size*traj_length, seq_length, feature_dim]

        # Move to device
        traj0_flat = traj0_flat.float().to(self.device)
        traj1_flat = traj1_flat.float().to(self.device)

        rewards0 = model(traj0_flat)  # [batch_size*traj_length, seq_length, 1]
        rewards1 = model(traj1_flat)  # [batch_size*traj_length, seq_length, 1]

        rewards0 = rewards0.reshape(batch_size, self.traj_length, self.seq_length)  # [batch_size, traj_length, seq_length]
        rewards1 = rewards1.reshape(batch_size, self.traj_length, self.seq_length)  # [batch_size, traj_length, seq_length]

        # only take the reward of the last state in a sequence
        rewards0 = rewards0[:, :, -1]  # [batch_size, traj_length]
        rewards1 = rewards1[:, :, -1]  # [batch_size, traj_length]

        return rewards0.sum(dim=1), rewards1.sum(dim=1)  # [batch_size]

    def preference_loss(self, reward_0, reward_1, labels):
        # use the cross-entropy loss
        # consider self.epsilon==0.1 which means there is 10% of chance that the label is wrong
        delta_R = reward_0 - reward_1
        sigma_delta_R = torch.sigmoid(delta_R)
        P_adjusted = (1 - self.epsilon) * sigma_delta_R + self.epsilon * 0.5
        y_values = torch.tensor([self.label_to_y(int(l.item())) for l in labels], dtype=torch.float, device=labels.device)
        # add this small eps to prevent log(0) from happening
        eps = 1e-12
        loss = - (y_values * torch.log(P_adjusted + eps) + (1 - y_values) * torch.log(1 - P_adjusted + eps))
        return loss.mean()

    def evaluate(self, model, loader):
        model.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for states, labels in loader:
                labels = labels.long().to(self.device)
                rewards0, rewards1 = self.compute_trajectory_rewards(model, states)
                loss = self.preference_loss(rewards0, rewards1, labels)
                # this is because the batch size of the last batch can be different,
                # so we use  * states.size(0) as an weight to calculate the correct mean loss by total_loss / count
                total_loss += loss.item() * states.size(0)
                count += states.size(0)
        return total_loss / count

    def train_single_model(self, train_loader, val_loader):
        # Input dimension is the per-step feature count = 15 in your case
        model = Transformer(input_dim=self.feature_dim,
                            transformer_embed_dim=self.transformer_embed_dim,
                            transformer_context_length=self.seq_length,
                            output_dim=1,
                            transformer_sinusoidal_embedding=True).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.init_weight_decay)

        final_train_loss = 0.0
        final_val_loss = 0.0
        for epoch in range(self.epochs):
            model.train()
            total_train_loss = 0.0
            count = 0
            for states, labels in train_loader:
                labels = labels.long().to(self.device)
                # compute trajectory rewards
                rewards0, rewards1 = self.compute_trajectory_rewards(model, states)
                loss = self.preference_loss(rewards0, rewards1, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * states.size(0)
                count += states.size(0)

            train_loss = total_train_loss / count
            val_loss = self.evaluate(model, val_loader)
            print(f"Epoch {epoch}: the pref predictor training loss is: {train_loss}, validation loss is: {val_loss}")
            # self.adjust_weight_decay(optimizer, train_loss, val_loss)
            if epoch > 30 and val_loss > train_loss * 1.25:
                final_train_loss = train_loss
                final_val_loss = val_loss
                break

            if epoch == self.epochs-1:
                # at the last epoch
                final_train_loss = train_loss
                final_val_loss = val_loss

        return model, final_train_loss, final_val_loss

    def train(self):
        self.models = []
        final_train_loss_list = []
        final_val_loss_list = []
        org_data_dict = self.state_label_data
        org_states = org_data_dict['states']
        org_labels = org_data_dict['labels']
        train_states, train_labels, val_states, val_labels = self.create_train_val_split(org_states, org_labels)

        train_loader, val_loader = self.create_dataloaders(train_states, train_labels, val_states, val_labels)
        # train a bunch of models to calculate the mean outputs in future
        for i in range(self.pool_models_num):
            model, final_train_loss, final_val_loss = self.train_single_model(train_loader, val_loader)
            self.models.append(model)
            final_train_loss_list.append(final_train_loss)
            final_val_loss_list.append(final_val_loss)

        # Find the indices of the self.select_models_num smallest losses
        smallest_loss_indices = heapq.nsmallest(self.select_models_num, range(len(final_val_loss_list)), key=final_val_loss_list.__getitem__)
        # Keep only the corresponding models
        filtered_model_list = [self.models[i] for i in smallest_loss_indices]
        self.models = filtered_model_list
        if self.save_models:
            # Create a dictionary to store the models
            save_models_dict = {f"model_{i}": model.state_dict() for i, model in enumerate(self.models)}

            # Save the dictionary to a .pt file at the path self.save_models_path
            torch.save(save_models_dict, self.save_models_path)
            print(f"Models saved successfully at {self.save_models_path}")

        return train_states, train_labels, val_states, val_labels

    def predict_batch_reward(self, current_state):
        """
        Given a batch of current_state (shape [batch_size, features]), predict the reward.
        This function only works when seq_length==1. Not solving the seq_length>1 situation yet.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")

        # current_state has size [batch_size, feature_dim]
        # unsqueeze the current_state to have a seq_length dimension, and seq_length == 1
        current_state = current_state.unsqueeze(1).to(self.device)  # [batch_size, 1, feature_dim]

        # Predict from each model
        rewards = []
        with torch.no_grad():
            for model in self.models:
                raw_rewards = model(current_state)  # [traj_length, 1, 1]
                raw_rewards = raw_rewards.squeeze()  # [traj_length, ]
                rewards.append(raw_rewards)

        # Average the three normalized rewards
        final_rewards = torch.stack(rewards).mean(dim=0)
        return final_rewards.to(self.device)

    def predict_single_traj_seq_reward(self, single_traj_seq):
        """
        Given a single states_seq (shape [traj_length, seq_length, features]), predict the reward.
        """
        if not self.models:
            raise ValueError("Models not trained. Call train() first.")

        # for the transformer input, need to add one more dim for batch size
        single_traj_seq = single_traj_seq.to(self.device)  # [traj_length, seq_length, features]

        # Predict from each model
        rewards = []
        with torch.no_grad():
            for model in self.models:
                raw_rewards = model(single_traj_seq)  # [traj_length, seq_length, 1]
                raw_rewards = raw_rewards[:, -1, :]  # [traj_length, 1, 1]
                raw_rewards = raw_rewards.squeeze()  # [traj_length, ]
                rewards.append(raw_rewards)

        # Average the three normalized rewards
        final_rewards = torch.stack(rewards).mean(dim=0)
        return final_rewards

    def predict_traj_reward(self, single_traj):
        """
        Given a single trajectory: shape [traj_length, features] (e.g. traj_length=24)
        Predict and sum up normalized rewards for each state in the trajectory.
        """
        # traj_states: [T, features]
        # Predict each step's normalized reward and sum
        total_reward = 0.0
        batch_single_traj = single_traj.unsqueeze(0)  # shape [1, traj_length, features]
        batch_single_traj_seq = self.traj_to_traj_seq(batch_single_traj)  # shape [1, traj_length, seq_length, features]
        single_traj_seq = batch_single_traj_seq.squeeze(0)   # shape [traj_length, seq_length, features]
        raw_rewards = self.predict_single_traj_seq_reward(single_traj_seq)  # [traj_length, ]
        total_reward += raw_rewards.sum().cpu().item()
        return total_reward

    def predict_compare_pairs(self, pairs):
        """
        Given pairs: shape [N, 2, 24, features]
        For each pair, compute traj_0 reward, traj_1 reward, and output predicted label:
        0 if traj_0 > traj_1, else 1.
        """
        # pairs: [N, 2, 24, features]
        predicted_labels = []
        for i in range(pairs.shape[0]):
            traj0_states = pairs[i, 0]  # [24, features]
            traj1_states = pairs[i, 1]  # [24, features]
            traj0_reward = self.predict_traj_reward(traj0_states)
            traj1_reward = self.predict_traj_reward(traj1_states)
            label = 0 if traj0_reward > traj1_reward else 1
            predicted_labels.append(label)
        return torch.tensor(predicted_labels, dtype=torch.long)
