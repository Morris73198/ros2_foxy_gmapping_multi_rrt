import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GNNLayer, self).__init__()
        self.hidden_size = hidden_size
        
        self.w1 = nn.Linear(hidden_size, hidden_size)
        self.w2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        h1 = self.w1(x)
        h2 = self.w2(x)
        return F.relu(h1 + h2)

class GNN(nn.Module):
    def __init__(self, input_shape, gnn_layers, use_history=True, ablation=0):
        super(GNN, self).__init__()

        self.is_recurrent = True
        self.rec_state_size = 512
        self.output_size = 512

        hidden_size = 256
        self.gnn_layers = gnn_layers
        self.use_history = use_history
        self.ablation = ablation

        # Actor
        self.actor = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU()
        )
        
        self.downscaling = 8

        # GNN layers
        self.gnn = nn.ModuleList(
            [GNNLayer(hidden_size) for _ in range(len(gnn_layers))]
        )

        # Critic
        self.critic = nn.ModuleList([
            nn.Linear(self.output_size, 1)
        ])

    def forward(self, inputs, rnn_hxs, masks, extras=None):
        x = self.encoder(inputs)
        N = x.size(0)

        if self.use_history and hasattr(self, 'last_x'):
            x = torch.cat([x, self.last_x], dim=1)

        self.last_x = x

        x = x.view(N, -1)
        x = F.relu(x)

        for layer in self.gnn:
            x = layer(x)

        return self.critic[0](x), x, rnn_hxs