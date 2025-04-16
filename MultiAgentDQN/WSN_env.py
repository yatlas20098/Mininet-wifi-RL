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
#import msgpack
import matplotlib.pyplot as plt
#from msgpack_numpy import patch as msgpack_numpy_patch
#msgpack_numpy_patch()
import pickle
import dill

try:
    from mininet_simulation import sensor_cluster
except ImportError:
    pass

from server import mininet_server
#SAMPLING_FREQ = [1,2,3]
# TOTAL_ENERGY = 100
#log_directory = "log"

class WSNEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}
    
    def __init__(self, sensor_ids, device, observation_time=10, transmission_size=4*1024, transmission_frame_duration=1, file_lines_per_chunk=5, recharge_thresh=0.2, sensor_coverage=0.4, sampling_freq=4, max_steps=100, num_episodes=10, local_mininet_simulation=True, server_ip="", server_port=""):
        super(WSNEnvironment, self).__init__()

        # Environment parameters
        self.num_sensors = len(sensor_ids) 
        self.sensor_ids = sensor_ids
        self.sensor_coverage = sensor_coverage
        self.max_steps = max_steps
        self.recharge_thresh=recharge_thresh
        self.alpha=0.6
        self.beta=0.3
        self.observation_time = observation_time
        num_transmission_frames = 1*((max_steps * num_episodes * observation_time) // transmission_frame_duration) + 1000 # include extra frames as buffer
        self._local_mininet_simulation = local_mininet_simulation
        if local_mininet_simulation:
            self._cluster = sensor_cluster(sensor_ids, log_directory=f'data/log', observation_time=self.observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, num_transmission_frames=num_transmission_frames)
            print("Starting cluster\n")
            cluster_process = multiprocessing.Process(target=self._cluster.start, args=())
            cluster_process.start()
            
            # Give cluster thread time to start
            time.sleep(10)

        else:
            self._cluster = mininet_server(self.num_sensors, server_ip, server_port)
            time.sleep(10)

        self._device = device

        self.step_log = []

        print("Done waiting for cluster to start")

        # Define observation space
        # self.num_sensor*self.num_sensor similarity features, self.num_sensors previous throughput features, and self.num_sensors previous transmission rate features 
        #self.observation_space = spaces.Box(low=0, high=1, shape=(self.num_sensors*self.num_sensors + 2*self.num_sensors,), dtype=np.float32)
        # Each sensor has num_sensor similarity features, a feature for its previous throughput, and a feature for its previous transmission rate
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_sensors, self.num_sensors + 2,), dtype=np.float32)

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

        similarity, throughputs, throughput_reward, clique_reward, rates = self._cluster.get_observation([2]*self.num_sensors)

        self._state = np.column_stack((similarity, rates, throughputs))
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device).squeeze()
        self._state = torch.flatten(self._state)

        return self._state, self.info

    def step(self, new_rates_id, action):
        print(f"Step action: {action}")

        # Execute one step in the environment
        truncated = bool(self.step_count > self.max_steps)
        terminated = False
        throughput_reward = [0] * self.num_sensors
        clique_reward = [0] * self.num_sensors

        # Check termination condition
        if truncated:
            terminated = True
            #self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device)
            return self._state, throughput_reward, clique_reward, terminated, truncated, self.info

        print('Returning reward')
        self.step_count += 1
        
        similarity, throughputs, throughput_reward, clique_reward, rates = self._cluster.get_observation(action.cpu().detach().numpy())
        self._state = np.column_stack((similarity, rates, throughputs))
        self._state = torch.tensor(self._state, dtype=torch.float32, device=self._device).squeeze()
        self._state = torch.flatten(self._state)

        return self._state, throughput_reward, clique_reward, terminated, truncated, self.info
