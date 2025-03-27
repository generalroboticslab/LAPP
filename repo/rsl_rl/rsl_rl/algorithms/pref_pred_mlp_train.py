import torch
# from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from rsl_rl.modules import PrefPredMlp

class PrefPredMlpTrain:
    def __init__(self, org_data: dict, device, epsilon=0.1, hidden_dims=[512,512,512], lr=1e-3, weight_decay=1e-4, epochs=50):
        self.org_data = org_data  # the dictionary of the original dataset
        self.epsilon = epsilon
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.init_weight_decay = weight_decay
        self.epochs = epochs

        self.state_label_data = self.org_data_to_state_label_data(self.org_data)
        self.feature_dim = self.state_label_data['states'].size(-1)

        self.models = []
        self.norms = []
        self.device = device

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

        assert len(state_tensor) == len(pref_label_tensor)
        # Create a mask for values that are not 3 or 4. Only accept 0, or 1, or 2
        # mask = (pref_label_tensor != 3) & (pref_label_tensor != 4)
        mask = (pref_label_tensor == 0) | (pref_label_tensor == 1) | (pref_label_tensor == 2)
        # Filter out incomparable (3) and response error (4)
        filtered_pref_label_tensor = pref_label_tensor[mask]
        filtered_state_tensor = state_tensor[mask]

        # Create a dictionary of filtered states and labels
        state_label_data = {
            'states': filtered_state_tensor,  # [N, 2, 24, features]
            'labels': filtered_pref_label_tensor,
        }

        return state_label_data

    def sample_with_replacement(self, data_dict: dict, size: int):
        # Generate random indices with replacement
        indices = torch.randint(low=0, high=size, size=(size,))

        # Sample states and labels using the generated indices
        sampled_states = data_dict['states'][indices]
        sampled_labels = data_dict['labels'][indices]

        return {
            'states': sampled_states,
            'labels': sampled_labels
        }

    def state_label_data_bootstrap(self, state_label_data: dict):
        dataset_size = state_label_data['states'].shape[0]
        # Create three sampled datasets
        state_label_data1 = self.sample_with_replacement(state_label_data, dataset_size)
        state_label_data2 = self.sample_with_replacement(state_label_data, dataset_size)
        state_label_data3 = self.sample_with_replacement(state_label_data, dataset_size)
        # a bag of data with 3 datasets, sampled from the original state_label_data with replacement
        state_label_data_bag = {
            'state_label_data1': state_label_data1,
            'state_label_data2': state_label_data2,
            'state_label_data3': state_label_data3
        }
        return state_label_data_bag

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

    def create_dataloaders(self, train_states, train_labels, val_states, val_labels, batch_size=32):
        # Convert to dataset of tuples
        train_dataset = TensorDataset(train_states, train_labels)
        val_dataset = TensorDataset(val_states, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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

    def compute_trajectory_rewards(self, model, states):
        """
        states: [batch_size, 2, 24, 15]
        model: PrefPredMlp
        We compute sum of rewards per trajectory:
        - Extract trajectory 0: [batch_size, 24, 15]
        - Extract trajectory 1: [batch_size, 24, 15]
        Feed each step into model and sum.
        """
        batch_size = states.shape[0]
        # trajectory 0 shape: (batch_size, 24, 15)
        traj0 = states[:, 0]  # [batch_size, 24, 15]
        traj1 = states[:, 1]  # [batch_size, 24, 15]
        # print(traj0.size())

        # Flatten steps for batch processing
        traj0_flat = traj0.reshape(batch_size * 24, self.feature_dim)
        traj1_flat = traj1.reshape(batch_size * 24, self.feature_dim)

        # Move to device
        traj0_flat = traj0_flat.float().to(self.device)
        traj1_flat = traj1_flat.float().to(self.device)

        # model returns [batch*24, 1]
        rewards0 = model(traj0_flat).reshape(batch_size, 24)  # sum over steps
        rewards1 = model(traj1_flat).reshape(batch_size, 24)
        # with torch.no_grad():
        #     # model returns [batch*24, 1]
        #     # For training/val, we don't need detach here, but in eval we have no_grad anyway.
        #     rewards0 = model(traj0_flat).reshape(batch_size, 24)  # sum over steps
        #     rewards1 = model(traj1_flat).reshape(batch_size, 24)

        return rewards0.sum(dim=1), rewards1.sum(dim=1)  # [batch_size], [batch_size]

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
                total_loss += loss.item() * states.size(0)
                count += states.size(0)
        return total_loss / count

    def normalize_model(self, model, val_loader):
        """
        Compute mean and std of raw rewards predicted by the model for individual states.
        We consider each state in isolation.
        """
        model.eval()
        predictions = []
        with torch.no_grad():
            for states, labels in val_loader:
                # states: [batch_size, 2, 24, features]
                batch_size = states.shape[0]
                # Flatten to get all states from both trajectories and steps individually
                # shape: [batch_size*2*24, features]
                all_states = states.reshape(batch_size * 2 * 24, self.feature_dim).float().to(self.device)

                # Predict raw rewards for each state
                raw_rewards = model(all_states)  # shape: [batch_size*2*24, 1]
                predictions.append(raw_rewards.cpu().squeeze())

        predictions = torch.cat(predictions)  # 1D tensor of raw rewards
        mean = predictions.mean()
        std = predictions.std() if predictions.std() > 0 else torch.tensor(1.0)
        return mean.item(), std.item()

    def train_single_model(self, train_loader, val_loader):
        # Input dimension is the per-step feature count = 15 in your case
        # The MLP input_dim is 15 since it processes one step at a time.
        # input_dim == 15
        model = PrefPredMlp(input_dim=self.feature_dim, output_dim=1, hidden_dims=self.hidden_dims).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.init_weight_decay)

        for epoch in range(self.epochs):
            print(epoch)
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
            print(f"the training loss is {train_loss}")
            print(f"the validation loss is {val_loss}")
            self.adjust_weight_decay(optimizer, train_loss, val_loss)

        mean, std = self.normalize_model(model, val_loader)
        return model, mean, std

    def train(self):
        self.models = []
        self.norms = []
        org_data_dict = self.state_label_data
        org_states = org_data_dict['states']
        org_labels = org_data_dict['labels']
        org_train_states, org_train_labels, org_val_states, org_val_labels = self.create_train_val_split(org_states, org_labels)
        train_data_dict = {
            'states': org_train_states,
            'labels': org_train_labels
        }

        train_state_label_data_bag = self.state_label_data_bootstrap(train_data_dict)

        data_keys = ['state_label_data1', 'state_label_data2', 'state_label_data3']

        for key in data_keys:
            data_dict = train_state_label_data_bag[key]
            train_states = data_dict['states']
            train_labels = data_dict['labels']

            train_loader, val_loader = self.create_dataloaders(train_states, train_labels, org_val_states, org_val_labels, batch_size=256)
            model, mean, std = self.train_single_model(train_loader, val_loader)
            self.models.append(model)
            self.norms.append((mean, std))

        return org_train_states, org_train_labels, org_val_states, org_val_labels

    def predict_state_reward(self, state):
        """
        Given a single state (shape [features]), predict the normalized reward.
        state: torch.Tensor of shape [features]
        """
        if not self.models or not self.norms:
            raise ValueError("Models not trained. Call train() first.")

        state = state.unsqueeze(0).to(self.device)  # [1, features]

        # Predict from each model
        rewards = []
        with torch.no_grad():
            for (model, (mean, std)) in zip(self.models, self.norms):
                raw_reward = model(state)  # [1, 1]
                normalized_reward = (raw_reward.squeeze() - mean) / std
                rewards.append(normalized_reward)
        # Average the three normalized rewards
        final_reward = torch.stack(rewards).mean()
        return final_reward.cpu().item()

    def predict_traj_reward(self, traj_states):
        """
        Given a trajectory: shape [T, features] (e.g. T=24)
        Predict and sum up normalized rewards for each state in the trajectory.
        """
        # traj_states: [T, features]
        # Predict each step's normalized reward and sum
        total_reward = 0.0
        for t in range(traj_states.shape[0]):
            step_state = traj_states[t]  # [features]
            step_reward = self.predict_state_reward(step_state)
            total_reward += step_reward
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
