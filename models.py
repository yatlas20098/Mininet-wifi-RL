import gymnasium as gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import struct
import time
import threading
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import GATConv


from torch.distributions.categorical import Categorical 

import os

from WSN_env import WSNEnvironment

class DDQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, num_sensors, device):
        super(DDQN, self).__init__()
        self._device = device

        w = 128 # number of nodes in a hidden layer
        num_hidden_layers = 12  

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())

        self._layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        self._fc1 = nn.Linear(w, w)
        std = math.sqrt(2.0 / (64))
        nn.init.normal_(self._fc1.weight, mean=0.0, std=std)                                                 
        self._fc1.bias.data.fill_(0.0)
        self._V = nn.Linear(w, 1)
        self._A = nn.Linear(w, sampling_freq)

     # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        x = self._layers(x).to(self._device)
        x = F.relu(self._fc1(x))

        V = self._V(x)
        A = self._A(x)
        Q = V + (A - A.mean(dim=0, keepdim=True))

        return Q

# (s,a) -> Q
class DQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, num_sensors, device):
        super(DQN, self).__init__()
        self._device = device

        w = 512 # number of nodes in a hidden layer
        num_hidden_layers = 64 

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, sampling_freq))
        self._layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        return self._layers(x).to(self._device)

class GraphEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        self.device = device
        self.gat1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=True)

        self.to(device)

    def forward(self, x, edge_index):
        x = F.relu(self.gat1(x.float(), edge_index))
        x = F.relu(self.gat2(x, edge_index))
        
        return x

# (s) -> a
class ActorNetwork(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def __init__(self, n_agents, n_actions, input_dim, lr, device, chkpt_dir=".\actor.chkpt", hidden_dim=16):
        super(ActorNetwork, self).__init__()
        self.device = device
        id_embedding_dim = 16 
        self._id_embeddings = nn.Embedding(n_agents, id_embedding_dim)
        self._checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self._graph_encoder = GraphEncoder(input_dim + id_embedding_dim, hidden_dim, device)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self._initialize_weights()
        self.to(device)
 
    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state, agent_ids):
        # Add id embeddings as node features
        num_graphs = getattr(state, 'num_graphs', 1) 
        agent_ids = agent_ids.unsqueeze(0).repeat(num_graphs, 1).view(-1)
        id_vecs = self._id_embeddings(agent_ids)
        state_w_ids = torch.cat([state.x, id_vecs], dim=-1)

        node_embedding = self._graph_encoder(state_w_ids, state.edge_index)
        logits = self._mlp(node_embedding)

        #logits = self._layers(state)
        return Categorical(logits=logits) # switch to categorical distrbuition

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))

# (s, a) -> Q
class CriticNetwork(nn.Module):
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.orthogonal_(m.weight)

               # init.kaiming_uniform_(m.weight, nonlinearity='relu')  # or use xavier_uniform_
                if m.bias is not None:
                    init.zeros_(m.bias)


    def __init__(self, input_dim, output_dim, lr, device, chkpt_dir=".\critic.chkpt", hidden_dim=32):
        super(CriticNetwork, self).__init__()
        self.device = device

        self._checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self._graph_encoder = GraphEncoder(input_dim, hidden_dim, device)
        self._mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._initialize_weights()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        graph_embedding = self._graph_encoder(state.x, state.edge_index)
        x = global_mean_pool(graph_embedding, state.batch)
        value = self._mlp(x)
        
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))
