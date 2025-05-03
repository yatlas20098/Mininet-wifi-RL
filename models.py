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
from torch_geometric.nn import GCNConv 

from WSN_env import WSNEnvironment

class DDQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, num_sensors, device):
        super(DDQN, self).__init__()
        self._device = device

        w = 130 # number of nodes in a hidden layer
        num_hidden_layers = 128 

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
class Actor(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors, device, min_freq, max_freq):
        super(Actor, self).__init__()
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._device = device
        w = 100 # number of nodes in a hidden layer
        num_hidden_layers = 100 
 
        layers = [nn.Linear(n_observations, w), nn.LeakyReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(w, 1))
 
        self.layers = nn.Sequential(*layers)
 
        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)
                                                                                
    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        action = self.layers(x).to(self._device)
        action = torch.tanh(action)
        action = (action + 1) / 2 * (self._max_freq - self._min_freq) + self._min_freq
        return action

# (s, a) -> Q
class Critic(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors, device):
        super(Critic, self).__init__()
        self._device = device
        w = 100 # number of nodes in a hidden layer
        num_hidden_layers = 100 
 
        layers = [nn.Linear(n_observations + 1, w), nn.LeakyReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Linear(w, 1))
 
        self.layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)
                                                                                
    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        return self.layers(x).to(self._device)

class ActorGNN(nn.Module):
    def __init__(self, input_dimension, num_hidden_layers, hidden_layer_dimension, sampling_freq, device):
        super(ActorGNN, self).__init__()
        self._device = device

        layers = [GCNConv(input_dimension, hidden_layer_dimension), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(GCNConv(input_dimension, hidden_layer_dimension))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_dimension, sampling_freq))
        self._layers = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        return self.layers(x, edge_index).to(self._device)

class CriticGNN(nn.Module):
    def __init__(self, input_dimension, num_hidden_layers, hidden_layer_dimension, device):
        super(CriticGNN, self).__init__()
        self._device = device

        layers = [GCNConv(input_dimension + 1, hidden_layer_dimension), nn.ReLU()] # +1 accounts for action
        for _ in range(num_hidden_layers):
            layers.append(GCNConv(hidden_layer_dimension, hidden_layer_dimension))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layer_dimension, 1))
        self._layers = nn.Sequential(*layers)

    def forward(self, x, edge_index):
        return self._layers(x, edge_index)

