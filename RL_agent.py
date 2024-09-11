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
log_directory = "/mydata/mydata/RL_agent/output"



def load_q_table(filename):
    with open(filename, 'rb') as f:
        q_values = dill.load(f)
    return q_values
def rmse(e1,e2):
    return np.sqrt(np.mean((e1 - e2) ** 2))

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



class WSNEnvironment(gym.Env):

    metadata = {"render_modes": ["console"]}
    # def _load_temperature_data(self, data_directory):
    #     temperature_data = []
    #     for i in range(self.num_sensors):
    #         file_path = os.path.join(data_directory, f'ch_received_from_sensor_{i}.txt')
    #         sensor_data = read_temperature_data(file_path)
    #         temperature_data.append(sensor_data)

    #     return np.array(temperature_data)
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
    
    def __init__(self, num_sensors=10, sensor_coverage=0.4, sampling_freq = 2, max_steps=50, threshold_prob=0.3):
        super(WSNEnvironment, self).__init__()

        # Environment parameters
        self.num_sensors = num_sensors
        self.sensor_coverage = sensor_coverage
        self.max_steps = max_steps
        self.threshold_prob = threshold_prob
        self.alpha=0.6
        self.beta=0.3


        ## this is where we should load the temprature data
        # self.temprature = np.random.randint(10, 12, size=(num_sensors-1, 100))
        self.temprature = self._load_temperature_data(log_directory)
        # self.remaining_energy = [energy] * self.num_sensors

        # Initialize the environment
        self.sensor_information, self.num_points = self._generate_sensor_positions(self.temprature[:,0])

        # Define observation space
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_points, 3), dtype=np.float32)

        self.sampling_freq = sampling_freq

        ## Define the action_space
        self.action_space = spaces.MultiDiscrete([self.sampling_freq] * self.num_points)



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






    def reset(self, seed=0):
        # Initialize the environment at the start of each episode
        self.step_count = 0
        self.generated_events = 0
        self.info = {'captured': 0, 'non-captured': 0}

        for i in range(self.num_points):
            self.info['sensor '+str(i)] = set()
            # initialize the energy and temprature
            # self.sensor_information[i,1] = self.remaining_energy[i]
            self.sensor_information[i,0] = self.temprature[i,0]
        return self.sensor_information, self.info

    def step(self, action):
        # Execute one step in the environment
        #print(self.step_count)
        truncated = bool(self.step_count > self.max_steps)
        terminated = False

        reward = 0

        # Check termination condition
        if truncated:
            terminated = True
            return self.sensor_information, reward, terminated, truncated, self.info

        ## energy consumption
        # energy_consumption = [0] * len(action)
        # for i in range(len(action)):
        #    if action[i] == 0:
        #       energy_consumption[i] = action[i] * 0.3  ## energy consumption for sensing
        #    else:
        #       energy_consumption[i] = action[i] * 4 ## energy consumption for sensing and data communication

        # ### update the state information


        for i in range(len(self.sensor_information)):
            for j in range(i + 1, len(self.sensor_information)):
                self.similarity = rmse(self.sensor_information[i,0], self.sensor_information[j,0])
                # print(f"Similarity between sensor {i} and sensor {j}: {self.similarity}")
            # Penalize pairs with high similarity and high frequency rates
                self.similarity_penalty += self.similarity * (action[i] + action[j])

        self.similarity_penalty /= (len(self.sensor_information) * (len(self.sensor_information) - 1) / 2)
        # print(f"similarity_penalty {self.similarity_penalty}")
        # print()
    # Calculate energy efficiency
        # average_energy =(np.mean(energy_consumption))
        # Utility combines penalties for redundancy and rewards for high average energy
        # reward = -self.alpha * (self.similarity_penalty) - self.beta * average_energy
        reward = -self.alpha * (self.similarity_penalty)
        reward
        




        for i in range(len(self.sensor_information)):
            # self.sensor_information[i,0] = self.temprature[i,self.step_count+1]
            if self.step_count + 1 < self.temprature.shape[1]:
                self.sensor_information[i,0] = self.temprature[i, self.step_count + 1]
            # self.sensor_information[i,1] = self.sensor_information[i,1] - energy_consumption[i]


        ## something should be considered 1) lower remaining energy, higher penalty for sensing and collecting data
        ## 2) how to measure the redundancy, for example, if the temprature of some devices have the similar values, but those devices all take the action for sending the data, then it should receive the penalty?
        ## 3) higher sampling frequency, higher energy consumption
        ## 4) additional terminate conditions: if all sensors die out



        self.step_count += 1

        # Return the next observation, reward, termination signal, and additional information
        return self.sensor_information, reward, terminated, truncated, self.info

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
    # def reset(self, seed=0):
    #     print("\n--- Resetting Environment ---")
    #     self.step_count = 0
    #     self.generated_events = 0
    #     self.info = {'captured': 0, 'non-captured': 0}

    #     for i in range(self.num_points):
    #         self.info['sensor '+str(i)] = set()
    #         self.sensor_information[i,0] = self.temprature[i,0]
    #         print(f"Sensor {i} initial temperature: {self.sensor_information[i,0]}")

    #     print(f"Initial sensor information:\n{self.sensor_information}")
    #     return self.sensor_information, self.info

    # def step(self, action):
    #     print(f"\n--- Step {self.step_count + 1} ---")
    #     print(f"Action taken: {action}")
        
    #     truncated = bool(self.step_count > self.max_steps)
    #     terminated = False
    #     reward = 0

    #     if truncated:
    #         terminated = True
    #         print("Episode terminated due to max steps reached.")
    #         return self.sensor_information, reward, terminated, truncated, self.info

    #     self.similarity_penalty = 0
    #     for i in range(len(self.sensor_information)):
    #         for j in range(i + 1, len(self.sensor_information)):
    #             self.similarity = rmse(self.sensor_information[i,0], self.sensor_information[j,0])
    #             print(f"Similarity between sensor {i} and {j}: {self.similarity}")

    #             penalty = self.similarity * (action[i] + action[j])
    #             self.similarity_penalty += penalty
    #             print(f"Penalty for sensors {i} and {j}: {penalty}")

    #     self.similarity_penalty /= (len(self.sensor_information) * (len(self.sensor_information) - 1) / 2)
    #     print(f"Overall similarity penalty: {self.similarity_penalty}")

    #     reward = -self.alpha * self.similarity_penalty
    #     print(f"Calculated reward: {reward}")

    #     print("Updating sensor information:")
    #     for i in range(len(self.sensor_information)):
    #         if self.step_count + 1 < self.temprature.shape[1]:
    #             old_temp = self.sensor_information[i,0]
    #             self.sensor_information[i,0] = self.temprature[i, self.step_count + 1]
    #             print(f"Sensor {i}: {old_temp} -> {self.sensor_information[i,0]}")
    #         else:
    #             print(f"Warning: Reached end of temperature data for sensor {i}")

    #     self.step_count += 1
    #     print(f"Updated sensor information:\n{self.sensor_information}")

    #     return self.sensor_information, reward, terminated, truncated, self.info

    # def _generate_sensor_positions(self,temp):
    #     print("\n--- Generating Sensor Positions ---")
    #     np.random.seed(42)
    #     num_points = 10
    #     sensor_information = np.zeros((num_points, 1))
    #     sensor_information[:, 0] = temp[0]
    #     print(f"Generated sensor information:\n{sensor_information}")
    #     return sensor_information.astype(np.float32), num_points

    # def render(self):
    #     pass

    # def close(self):
    #     pass


class WSNEnvironmentAgent:
    def __init__(
        self,
        env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """

        ### initialize the q-table
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.nvec))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: np.ndarray) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # Convert numpy array to tuple
        obs_key = tuple(tuple(row) for row in obs)
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # with probability (1 - epsilon) act greedily (exploit) instead of one action, we generate an action vector
        else:
            return np.unravel_index(np.argmax(self.q_values[obs_key]), self.q_values[obs_key].shape)

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Updates the Q-value of an action."""
        next_obs_key = tuple(tuple(row) for row in next_obs)
        m_obs = tuple(tuple(row) for row in obs)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[m_obs][action]
        )

        self.q_values[m_obs][action] = (
            self.q_values[m_obs][action] + self.lr * temporal_difference
        )


        self.training_error.append(temporal_difference)
    def save_q_table(self, filename):
      with open(filename, 'wb') as f:
          dill.dump(self.q_values, f)


    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    # def get_action(self, obs: np.ndarray) -> int:
    #     obs_key = tuple(tuple(row) for row in obs)
    #     if np.random.random() < self.epsilon:
    #         action = self.env.action_space.sample()
    #         print(f"Exploring: Random action chosen: {action}")
    #     else:
    #         action = np.unravel_index(np.argmax(self.q_values[obs_key]), self.q_values[obs_key].shape)
    #         print(f"Exploiting: Best action chosen: {action}")
    #     return action

    # def update(self, obs: np.ndarray, action: int, reward: float, terminated: bool, next_obs: np.ndarray):
    #     print("\n--- Updating Q-values ---")
    #     next_obs_key = tuple(tuple(row) for row in next_obs)
    #     m_obs = tuple(tuple(row) for row in obs)
    #     future_q_value = (not terminated) * np.max(self.q_values[next_obs_key])
    #     temporal_difference = (
    #         reward + self.discount_factor * future_q_value - self.q_values[m_obs][action]
    #     )
    #     print(f"Temporal Difference: {temporal_difference}")

    #     old_q_value = self.q_values[m_obs][action]
    #     self.q_values[m_obs][action] = old_q_value + self.lr * temporal_difference
    #     print(f"Updated Q-value: {old_q_value} -> {self.q_values[m_obs][action]}")

    #     self.training_error.append(temporal_difference)

    # def decay_epsilon(self):
    #     old_epsilon = self.epsilon
    #     self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
    #     print(f"Epsilon decayed: {old_epsilon} -> {self.epsilon}")
        
# env = WSNEnvironment()

# # hyperparameters
# learning_rate = 0.01
# n_episodes = 10_000
# start_epsilon = 1.0
# epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
# final_epsilon = 0.1

# agent = WSNEnvironmentAgent(
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

# episodes = 20
# scores = []

# for episode in tqdm(range(n_episodes)):
#     obs, info = env.reset()
#     m_obs = tuple(tuple(row) for row in obs)
#     done = False

#     # play one episode
#     while not done:
#         action = agent.get_action(m_obs)
#         next_obs, reward, terminated, truncated, info = env.step(action)

#         m_next_obs = tuple(tuple(row) for row in next_obs)
#         # update the agent
#         agent.update(m_obs, action, reward, terminated, next_obs)

#         # update if the environment is done and the current obs
#         done = terminated or truncated
#         m_obs = m_next_obs

#     agent.decay_epsilon()


# for episode in range(1, episodes+1):
#     state, info = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = agent.get_action(state)
#         n_state, reward, terminated, truncated, info = env.step(action)
#         score += reward
#         done = terminated or truncated
#         state = n_state


#     scores.append(score)
#     if (episode==n_episodes-1):
#       filename = 'q_table.pkl'
#       agent.save_q_table(filename)





#     print('Episode:{}\t Score:{:.2f} \t{}'.format(episode, score, info))



# scoresW=scores
# # Print average score
# print('Average Score:', np.mean(scores))
# Q_Learning_Scheduler = scoresW