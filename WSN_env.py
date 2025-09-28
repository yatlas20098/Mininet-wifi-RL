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
from configs import Mininet_Simulation_Config

class WSNEnvironment(gym.Env):
    metadata = {"render_modes": ["console"]}
    
    def __establish_virtual_WSN_cluster(self, config):
        if config.local_simulation:
            from network import Network
            # Create a local virtual cluster
            self._network = Network(config)
            
            # Start the cluster thread
            print("Local simulation requested. Starting network simulation.\n")
            network_thread = threading.Thread(target=self._network.start, args=())
            network_thread.start()
            
            # Give the cluster thread time to start up
            print("Giving cluster thread time to start")
            time.sleep(20)
        else:
            from remote_session_utils import Mininet_Remote_Session 

            # Connect to a remote virtual cluster
            print("Remote simulation requested. Attempting connect to remote virtual vluster\n")
            self._cluster = Mininet_Remote_Session(self._num_sensors, config.server_ip, config.port)

            # Give the cluster time to start up
            time.sleep(10)

    def __init__(self, sim_config, max_steps, device):
        super(WSNEnvironment, self).__init__()
        
        # Environment parameters
        self._num_sensors = len(sim_config.sensor_ids)
        self._device = device
        self._max_steps = max_steps
        self._num_clusters = sim_config.num_clusters
        
        self.__establish_virtual_WSN_cluster(sim_config)

        ######################### Observation Space #########################
        # The observation space at time t is an (n+2) x n matrix. The ith row
        # in the matrix consists of the recorded temperature (temp_i), 
        # throughput (th_i), transmission rate (tr_i), and n similarity 
        # values (sim_{i,j}) for sensor i. 
        #
        # Temperature:
        #   Unit: Celsuius
        #   Range: -10 to 25
        # Throughput:
        #   Unit: Packets / Second
        #   Range: 0 to +inf
        # Transmission Rate:
        #   Unit: Packets / Second
        #   Range: 0 to +inf
        # Similarity:
        #   Range: 0 or 1 (binary)
        #
        # Note that similarity is computed using the isSimlar function.
        #
        #                       Observation Space Matrix
        # |temp_1| |th_i| |tr_1| |sim_{1,1}| |sim_{1,2}| ... |sim_{1,n}|
        # |temp_2| |th_2| |tr_2} |sim_{2,1}| |sim_{2,2}| ... |sim_{2,n}|
        #                                ... 
        # |temp_n| |th_n| |tr_n| |sim_{n,1}| |sim_{n,2}| ... |sim_{n,n}|
        #####################################################################
        self.observation_space = spaces.Box(low=-100, high=1000, shape=(self._num_sensors, self._num_sensors + 3,), dtype=np.float32)

        ########################### Action Space ###########################
        # The action space at time t is an n+1 vector. The first n entries
        # correspond to transmisison rates (tr) for sensors. And the n+1th 
        # entry is t' if the enviornment at time t' should be replayed
        # and -1 otherwise. 
        #
        # Transmission Rate:
        #   Unit: Packets / Second
        #   Range: 0 to 250
        #
        # |tr_1| |tr_2| ... |tr_n| |replay|
        ####################################################################
        self.action_space = spaces.MultiDiscrete(np.array([250] * (self._num_sensors + 1)))

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
        # Reset step count
        self.step_count = 0
    
        # Set sensor transmission rates to 0 and observe the inital state 
        observations = self._network.get_observation(np.zeros((self._num_clusters, self._num_sensors)))

        # state = |similarity, rates, throughput|
        # self._states = (np.column_stack(tuple((obs[0], obs[3], obs[1]))) for obs in observations)
        # self._states = (torch.tensor(state, dtype=torch.float32, device=self._device).squeeze() for state in self._states)
        # self._states = tuple(torch.flatten(state).detach() for state in self._states)
        #_, self._states, _ = zip(*observations)
        _, connectivity_graphs, redundancy_graphs, _ = zip(*observations)
        self._states = list(zip(connectivity_graphs, redundancy_graphs))

        return self._states, self.info

    def step(self, new_rates_id, actions):
        # Execute one step in the environment
        truncated = [bool(self.step_count > self._max_steps)]
        terminated = [False]
        rewards = [{
                "throughput": [0] * self._num_sensors,
                "similarity": [0] * self._num_sensors
                }] * self._num_clusters
        
        # Check termination condition
        if truncated[0]:
            print("Episode terminated")
            terminated = [True]
            return self._states, rewards, terminated, truncated, self.info

        self.step_count += 1
        actions = (action.cpu().detach().numpy() for action in actions)
        observations = self._network.get_observation(actions)
        
        # state = |similarity, rates, throughput|
        #self._states = (np.column_stack(tuple((obs[0], obs[3], obs[1]))) for obs in observations)
        #self._states = (torch.tensor(state, dtype=torch.float32, device=self._device).squeeze() for state in self._states)
        #self._states = tuple((torch.flatten(state) for state in self._states))
        #rewards = tuple((obs[2] for obs in observations))
        terminated, connectivity_graphs, redundancy_graphs, rewards = zip(*observations)
        self._states = list(zip(connectivity_graphs, redundancy_graphs))


        #print("Sim: ", [obs[0] for obs in observations])
        #print("Rates: ", [obs[3] for obs in observations])
        #print("Throughput: ", [obs[1] for obs in observations])
        print("Rewards: ", [r["throughput"] for r in rewards])

        return self._states, rewards, terminated, truncated, self.info
