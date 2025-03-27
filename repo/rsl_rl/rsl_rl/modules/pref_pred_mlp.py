import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributions as D

class PrefPredMlp(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dims=[512, 512, 512]):
        super(PrefPredMlp, self).__init__()
        # Define the MLP layers
        layers = []
        dims = [input_dim] + hidden_dims  # 3 for mean, scale, and logits
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, inputs):
        # inputs: [batch, input_dim]
        x = self.mlp(inputs)
        pref_reward = self.output_layer(x)  # [batch, 1]
        return pref_reward
