import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

import os
import gym
from gym import spaces
from torch import nn
import torch
from collections import deque, defaultdict
import itertools
import numpy as np
import random
from scipy.stats import poisson
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import msgpack
import matplotlib.pyplot as plt
from msgpack_numpy import patch as msgpack_numpy_patch
msgpack_numpy_patch()
import pickle
import dill
SAMPLING_FREQ = [1,2,3]
# TOTAL_ENERGY = 100
log_directory = "data/log"

# def read_temperature_data(file_path):
#     with open(file_path, 'r') as file:
#         # Skip the header line
#         next(file)
#         # Read only the temperature data (9th column, index 8)
#         data = [float(line.strip().split(',')[8]) for line in file]
#     return np.array(data)
def read_temperature_data(file_path):
    with open(file_path, 'r') as file:
        # Skip the first 4 lines

        data = []
        for line in file:
            try:
                # Split the line and try to convert the temperature value (9th column, index 8) to float
                temperature = float(line.strip().split(',')[8])
                data.append(temperature)
            except (ValueError, IndexError):
                # If conversion fails or the line doesn't have enough columns, skip this line
                continue

    return np.array(data)

class IoBTEnv(py_environment.PyEnvironment):
    metadata = {"render_modes": ["console"]}
    def _load_temperature_data(self, data_directory):
        temperature_data = []

        for i in range(self.num_sensors):
            file_path = os.path.join(data_directory, f'ch_received_from_sensor_{i}.txt')
            if os.path.exists(file_path):
                sensor_data = read_temperature_data(file_path)
                temperature_data.append(sensor_data)
            else:
                print(f"Warning: Temperature data file not found: {file_path}")

        if not temperature_data:
            print("Warning: No temperature data loaded")
            return np.array([[]])  # Return an empty 2D array

        # Find the length of the shortest data array
        min_length = min(len(data) for data in temperature_data)

        # Truncate all arrays to the shortest length
        truncated_data = [data[:min_length] for data in temperature_data]

        result = np.array(truncated_data)
        print(f"Final temperature data shape: {result.shape}")

        return result

    def __init__(self, num_sensors=10, sensor_coverage=0.4, sampling_freq=2, max_steps=50, threshold_prob=0.3):
        #super(WSNEnvironment, self).__init__()

        # Environment parameters
        self.num_sensors = num_sensors
        self.sensor_coverage = sensor_coverage
        self.max_steps = max_steps
        self.threshold_prob = threshold_prob
        self.alpha=0.6
        self.beta=0.3
        self.sampling_freq = sampling_freq

        self._state = []
        self._epsiode_ended = False

        # Load the temperature data
        self.temperature = self._load_temperature_data(log_directory)

        # Internal state variables
        self.step_count = 0

        # information dictionary
        self.info = {}

        # the generated event
        self.event = None

        # Number of generated events
        self.generated_events = 0
        self.similarity=0
        self.similarity_penalty=0

        
        # Initialize the environment
        self.sensor_information, self.num_points = self._generate_sensor_positions(self.temperature[:,0])

        """
        Define the observation space

        The observation space is a matrix with rows as sensors and columns as sensor data 
        """
        self._observation_spec = array_spec.BoundedArraySpec(shape=(self.num_points, 3,), dtype=np.float32, minimum=0, maximum=1000, name='observation')
        #self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_points, 3), dtype=np.float32)

        """
        Define the action space
        
        At each step a sensor can be assigned a frequency of 0, 1, or 2
        """
        self._action_spec = array_spec.BoundedArraySpec(shape=(), dtype=np.float32, minimum=0, name='action')
        #self.action_space = spaces.MultiDiscrete([self.sampling_freq] * self.num_points)
        
    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = 0
        self._episode_ended = False

        self.step_count = 0
        self.generated_events = 0
        self.info = {'captured': 0, 'non-captured': 0}

        for i in range(self.num_points):
            self.info['sensor '+str(i)] = set()
            # initialize the energy and tempreture
            # self.sensor_information[i,1] = self.remaining_energy[i]
            self.sensor_information[i,0] = self.temperature[i,0]
        return self.sensor_information, self.info

        #return ts.restart(np.array([self._state], dtype=np.int32))

    def _step(self, action):
        reward = 0

        if self.step_count > self.max_steps:
            self._epsiode_ended = True
            return ts.termination(np.array([self._state], dtype=np.float32), reward)

        for i in range(len(self.sensor_information)):
            for j in range(i + 1, len(self.sensor_information)):
                self.similarity = rmse(self.sensor_information[i,0], self.sensor_information[j,0])
                print(f"Similarity between sensor {i} and sensor {j}: {self.similarity}")

                # Penalize pairs with high similarity and high frequency rates
                self.similarity_penalty += self.similarity * (action[i] + action[j])

        self.similarity_penalty /= (len(self.sensor_information) * (len(self.sensor_information) - 1) / 2)

        ### Calculate energy efficiency
        # Utility combines penalties for redundancy and rewards for high average energy
        reward = -self.alpha * (self.similarity_penalty)

        for i in range(len(self.sensor_information)):
            # self.sensor_information[i,0] = self.temprature[i,self.step_count+1]
            if self.step_count + 1 < self.temprature.shape[1]:
                self.sensor_information[i,0] = self.temprature[i, self.step_count + 1]
        
        self.step_count += 1

        #return self.sensor_information, reward, self.epsiode_ended, self.info
        return ts.transition(
                np.array([self._state], dtype=np.float32), reward=reward, discount=1.0)

    def _generate_sensor_positions(self,temp):
        # Set a fixed random seed for reproducibility
        np.random.seed(42)

        # # Define the intensity (lambda) for the Poisson process
        # intensity = self.num_sensors / (1 * 1)  # Adjust as needed based on the size of the space

        # Generate the number of points based on a Poisson distribution
        num_points = 10

        # Generate the sensor positions uniformly at random within the space
        sensor_information = np.zeros((num_points, 1))

        sensor_information[:, 0] = temp[0]

        # Fill the second column with 100
        # sensor_information[:, 1] = energy

        # Create a new column of zeros with the same number of sensors to calculate the number of pulling for each one
        #zeros_column = np.zeros((num_points, 1))

        # Concatenate the original array with the new column of zeros
        # sensor_positions_with_zeros = np.hstack((sensor_positions, zeros_column))
        # if not temp:
        #     print("Warning: No temperature data loaded")
        #     return np.array([[]]),10

        return sensor_information.astype(np.float32), num_points

