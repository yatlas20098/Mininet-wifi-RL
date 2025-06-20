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

# (s) -> a
class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, lr, device, chkpt_dir=".\actor.chkpt", fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.device = device
 
        self._checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self._layers = nn.Sequential(
            nn.Linear(input_dims, fc1_dims), 
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
 
    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        dist = self._layers(state)
        dist = Categorical(dist) # switch to categorical distrbuition
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))

# (s, a) -> Q
class CriticNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, lr, device, chkpt_dir=".\critic.chkpt", fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.device = device

        self._checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self._layers = nn.Sequential(
            nn.Linear(input_dims, fc1_dims), 
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, state):
        value = self._layers(state)
        
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self._checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self._checkpoint_file))
