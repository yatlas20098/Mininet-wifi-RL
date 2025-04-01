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

from WSN_env import WSNEnvironment

# (s) -> a
class Actor(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors, device, min_freq, max_freq):
        super(Actor, self).__init__()
        self._min_freq = min_freq
        self._max_freq = max_freq
        self._device = device
        w = 64 # number of nodes in a hidden layer
        num_hidden_layers = 6 
 
        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, 1))
        #layers.append(nn.ReLU())
 
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
        w = 64 # number of nodes in a hidden layer
        num_hidden_layers = 6 
 
        layers = [nn.Linear(n_observations + 1, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, 1))
        layers.append(nn.ReLU())
 
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
