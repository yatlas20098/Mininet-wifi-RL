import os
import gymnasium as gym
from gymnasium import spaces
from torch import nn
import torch
from collections import deque, defaultdict
import itertools
import numpy as np
import random
import time
import struct
from scipy.stats import poisson
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import threading
import multiprocessing
import matplotlib.pyplot as plt
import pickle
import dill

try:
    from mininet_simulation import sensor_cluster, Mininet_Simulation_Parameters
except ImportError:
    pass

from server import Mininet_Remote_Session 


class WSNEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, mininet_simulation_parameters, max_steps, device):
        super(WSNEnvironment, self).__init__()

        # Environment parameters
        self._num_sensors = len(mininet_simulation_parameters.sensor_ids) 
        if mininet_simulation_parameters.local_simulation:
            self._cluster = sensor_cluster(mininet_simulation_parameters)
            print("Starting cluster\n")
            cluster_process = multiprocessing.Process(target=self._cluster.start, args=())
            cluster_process.start()
            
            # Give cluster thread time to start
            time.sleep(10)

        else:
            self._cluster = mininet_server(self._num_sensors, server_ip, server_port)
            time.sleep(10)

        self._device = device
        self._max_steps = max_steps

        self.step_log = []

        print("Done waiting for cluster to start")

        # Define observation space
        # self.num_sensor*self.num_sensor similarity features, self.num_sensors previous throughput features, and self.num_sensors previous transmission rate features 
        #self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_sensors*self.num_sensors + 2*self.num_sensors,), dtype=np.float32)
        # Each sensor has num_sensor similarity features, a feature for its previous throughput, and a feature for its previous transmission rate
        #self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_sensors, self.num_sensors + 2,), dtype=np.float32)

        #self.sampling_freq = sampling_freq

        ## Define the action_space
        #self.action_space = spaces.MultiDiscrete([self.sampling_freq] * self.num_sensors)

        # Internal state variables
        self.step_count = 0

        # information dictionary
        self.info = {}

        # the generated event
        self.event = None

        # Number of generated events
        self.generated_events = 0
        self.similarity = 0
        self.similarity_penalty = 0

    def reset(self, seed=0):
        # Initialize the environment at the start of each episode
        self.step_count = 0
        self.generated_events = 0
        self.info = {'captured': 0, 'non-captured': 0}

        for i in range(self._num_sensors):
            self.info['sensor ' + str(i)] = set()

        similarity, throughputs, throughput_reward, clique_reward, rates = self._cluster.get_observation([2]*self._num_sensors)

        self._state = np.column_stack((similarity, rates, throughputs))
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device).squeeze()
        self._state = torch.flatten(self._state)

        return self._state, self.info

    def step(self, new_rates_id, action):
        print(f"Step action: {action}")

        # Execute one step in the environment
        truncated = bool(self.step_count > self._max_steps)
        terminated = False
        throughput_reward = [0] * self._num_sensors
        clique_reward = [0] * self._num_sensors

        # Check termination condition
        if truncated:
            terminated = True
            return self._state, throughput_reward, clique_reward, terminated, truncated, self.info

        print('Returning reward')
        self.step_count += 1
        
        similarity, throughputs, throughput_reward, clique_reward, rates = self._cluster.get_observation(action.cpu().detach().numpy())
        self._state = np.column_stack((similarity, rates, throughputs))
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device).squeeze()
        self._state = torch.flatten(self._state)

        return self._state, throughput_reward, clique_reward, terminated, truncated, self.info
