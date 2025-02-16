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
#import msgpack
import matplotlib.pyplot as plt
#from msgpack_numpy import patch as msgpack_numpy_patch
#msgpack_numpy_patch()
import pickle
import dill
from local_simulation import sensor_cluster 

SAMPLING_FREQ = [1,2,3]
# TOTAL_ENERGY = 100
log_directory = "log"

class WSNEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, cluster_id, sensor_ids, device, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, recharge_thresh=0.2, sensor_coverage=0.4, sampling_freq=4, max_steps=100, threshold_prob=0.3):
        super(WSNEnvironment, self).__init__()

        # Environment parameters
        self.num_sensors = len(sensor_ids) 
        self.sensor_ids = sensor_ids
        self.sensor_coverage = sensor_coverage
        self.max_steps = max_steps
        self.threshold_prob = threshold_prob
        self.recharge_thresh=recharge_thresh
        self.alpha=0.6
        self.beta=0.3
        self.observation_time = observation_time 
        self._cluster = sensor_cluster(cluster_id, sensor_ids, log_directory=f'data/c{cluster_id}/log', data_offset=cluster_id, observation_time=self.observation_time, transmission_size=transmission_size, file_lines_per_chunk=file_lines_per_chunk)
        self._device = device

        print("Starting cluster\n")
        cluster_thread = threading.Thread(target=self._cluster.start, args=())
        cluster_thread.start()
        # Give cluster thread time to start
        time.sleep(10)
        print("Done waiting for cluster to start")

        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_sensors*self.num_sensors + 2*self.num_sensors,), dtype=np.float32)

        self.sampling_freq = sampling_freq

        ## Define the action_space
        self.action_space = spaces.MultiDiscrete([self.sampling_freq] * self.num_sensors)

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

        for i in range(self.num_sensors):
            self.info['sensor ' + str(i)] = set()

        similarity, energy_data, throughputs, reward = self._cluster.get_obs()
        time.sleep(self.observation_time)
        similarity, energy_data, throughputs, reward = self._cluster.get_obs()
        self._state = np.concatenate([similarity.flatten(), energy_data, throughputs])
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device)

        return self._state, self.info
    
    def step(self, new_rates_id, action):
        print(f"Step action: {action}")

        # Execute one step in the environment
        truncated = bool(self.step_count > self.max_steps)
        terminated = False
        reward = 0

        # Check termination condition
        if truncated:
            terminated = True
            print(self.sensor_information, reward, terminated, truncated, self.info)
            self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device)
            return self._state, reward, terminated, truncated, self.info
       
        for i in range(len(action)):
            self._cluster.transmission_freq_idxs[i] = action[i]

        print('Waiting for observation from cluster head')
        time.sleep(self.observation_time)

        print('Returning reward')
        self.step_count += 1
       
        similarity, energy_data, throughputs, reward = self._cluster.get_obs()
        self._state = np.concatenate([similarity.flatten(), energy_data, throughputs])
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device)

        return self._state, reward, terminated, truncated, self.info
