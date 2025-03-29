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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )
#device = torch.device("cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'throughput_reward', 'clique_reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen = capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors):
        super(DQN, self).__init__()
        w = 512 # number of nodes in a hidden layer
        num_hidden_layers = 64 

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, n_actions))
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
        return self.layers(x).to(device)

# Dueling DQN
# Some code taken from github.com/markelsanz14/indepdent-rl-agents
class DDQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors):
        super(DDQN, self).__init__()

        w = 1024 # number of nodes in a hidden layer
        num_hidden_layers = 64 

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        #layers.append(nn.Linear(w, n_actions))
        #layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

        self.fc1 = nn.Linear(w, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self.fc1.weight, mean=0.0, std=std)
        self.fc1.bias.data.fill_(0.0)
        self.V = nn.Linear(512, 1)
        self.A = nn.Linear(512, n_actions)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        x = self.layers(x).to(device)
        #x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.relu(self.fc1(x))

        V = self.V(x)
        A = self.A(x)
        Q = V + (A - A.mean(dim=0, keepdim=True))

        return Q

class WSN_agent:
    def __init__(self, num_clusters, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.99, EPS_START = 0.9, EPS_END = 0.3, EPS_DECAY = 1500, TAU = 0.005, LR = 0.001, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=512, local_mininet=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet=local_mininet, server_ip=server_ip, server_port=server_port) 
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq # Number of possible transmission frequencies
        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_clusters = num_clusters
        self._num_episodes = num_episodes
        self._train_every = {} # How many steps between updates of target network 

        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        num_actions = self._num_sensors * sampling_freq
        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'clique']
        self._policy_net = {} 
        self._target_net = {}
        self._optimizer = {}
        self._loss = {}
        self._rewards = {}

        for reward_type in self._reward_types:
            self._train_every[reward_type] = train_every
            self._policy_net[reward_type] = DDQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device) # DQN or DDQN
            self._target_net[reward_type] = DDQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
            self._policy_net[reward_type].load_state_dict(self._policy_net[reward_type].state_dict())
            self._optimizer[reward_type] = optim.AdamW(self._policy_net[reward_type].parameters(), lr=LR, amsgrad=True)
            self._loss[reward_type] = []
            self._rewards[reward_type] = []

        self._memory = ReplayMemory(max_steps*num_episodes)

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []

        self._BATCH_SIZE = BATCH_SIZE 
        self._GAMMA = GAMMA 
        self._EPS_START = EPS_START
        self._EPS_END = EPS_END 
        self._EPS_DECAY = EPS_DECAY
        self._TAU = TAU

        time.sleep(20)
        self._state, self._info = self._env.reset()

    def _select_action(self):
        sample = random.random()
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * math.exp(-1. * self._steps_done / self._EPS_DECAY)
        self._steps_done += 1
        action = None

        energy = self._state[:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        energy = energy.squeeze()
        dead_sensors = (energy <= self._recharge_thresh)
        awake_sensors = (energy > self._recharge_thresh)

        if sample > eps_threshold:
            print('Getting best action')
            with torch.no_grad():
                throughput_action_values = self._policy_net['throughput'](self._state.view(-1))
                throughput_action_values = throughput_action_values.view(self._num_sensors, self._sampling_freq)

                clique_action_values = self._policy_net['clique'](self._state.view(-1))
                clique_action_values = clique_action_values.view(self._num_sensors, self._sampling_freq)

                combined_action_values = -throughput_action_values * clique_action_values
                action_probs = F.softmax(combined_action_values, dim=1)

                #print(throughput_action_values)
                #print(clique_action_values)
                #print(combined_action_values)

                action = torch.argmax(combined_action_values, dim=1)
        else:
            print('Getting random action')
            action = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.long, device=device) 
       
        print(f'action: {action}')
        # action[dead_sensors] = 0 # Dont allow dead sensors to transmit
        # action[awake_sensors & (action==0)] = 1 # Force awake sensors to transmit

        return action

    def _optimize_model(self):
        for reward_type in self._reward_types:
            if len(self._memory) < self._BATCH_SIZE:
                print(f'Mem len: {len(self._memory)}')
                return

            # Sample a batch of transitions 
            transitions = self._memory.sample(self._BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            # Get the non-final states in the batch
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            state_batch = torch.cat(batch.state)
           
            # Get the actions from the batch of transitions
            action_batch = torch.cat(batch.action).long().to(device)
            action_batch = action_batch.view(self._BATCH_SIZE, self._num_sensors)
            
            # Get the rewards from the batch of transitions
            if reward_type == 'throughput':
                reward_batch = torch.cat(batch.throughput_reward).view(-1, 1)
            else:
                reward_batch = torch.cat(batch.clique_reward).view(-1, 1)

            # Get the Q-values corresponding to the actions
            state_action_values = self._policy_net[reward_type](state_batch.view(self._BATCH_SIZE, -1)).view(-1, self._num_sensors, self._sampling_freq).gather(2, action_batch.unsqueeze(1))

            # Remove any unneccesary dimensions
            state_action_values = state_action_values.squeeze()
            
            # Calculate expected Q-values
            next_state_values = torch.zeros(self._BATCH_SIZE, self._num_sensors, device=device)
            with torch.no_grad():
                max_values = self._target_net[reward_type](non_final_next_states.view(-1, 12*self._num_sensors)).view(-1, self._num_sensors, self._sampling_freq).max(2)[0]
                next_state_values[non_final_mask] = max_values.view(-1, self._num_sensors)
            expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch

            # Calculate Loss 
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values)

            # Log loss
            self._loss[reward_type].append(loss.item())
            with open('loss.pkl', 'wb') as file:
                pickle.dump(self._loss, file)
            print(f"{reward_type} Loss: {loss.item():.4f}")

            # Optimization step
            self._optimizer[reward_type].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self._policy_net[reward_type].parameters(), 100)
            self._optimizer[reward_type].step()

    def train(self):
        # Offset so training does not occur during the same step 
        self._train_steps = {self._reward_types[0]:0, self._reward_types[1]:10}

        for i_episode in range(self._num_episodes):
            # Initialize the environment and get its state
            print(self._env.reset())
            self._state, self._info = self._env.reset()
            self._state = torch.tensor(self._state, dtype=torch.float32, device=device).unsqueeze(0)
            self._state = torch.tensor(self._state, dtype=torch.float32, device=device)

            print(f'State = {self._state}')
            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                print('Taking next step')

                # Sample an action using the policy 
                action = torch.Tensor.cpu(self._select_action())

                # Sample the next frame from the enviornment, and receive a reward
                observation, throughput_reward, clique_reward, terminated, truncated, _ = self._env.step(self._steps_done, action.numpy())
                
                # Log the reward
                self._rewards['throughput'].append(throughput_reward)
                self._rewards['clique'].append(clique_reward)

                print(f'Throughput reward: {throughput_reward}')
                print(f'Clique reward: {clique_reward}')
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_reward = torch.tensor([throughput_reward], device=device)
                clique_reward = torch.tensor([clique_reward], device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    #next_state = observation.clone().detach()
                    next_state = observation.unsqueeze(0).clone().detach()

                # Store the transition in memory
                self._memory.push(self._state, action, next_state, throughput_reward, clique_reward)

                # Move to the next state
                self._state = next_state
               
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                for reward_type in self._reward_types:
                    if self._train_steps[reward_type] >= self._train_every[reward_type]:
                        self._train_steps[reward_type] = 0
                        # Soft update of the target network's weights
                        target_net_state_dict = self._target_net[reward_type].state_dict()
                        policy_net_state_dict = self._policy_net[reward_type].state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                        self._target_net[reward_type].load_state_dict(target_net_state_dict)

                if done:
                    self._episode_durations.append(t + 1)
                    break
                
                print('Step done')
                for reward_type in self._reward_types:
                    self._train_steps[reward_type] += 1
        
if __name__ == '__main__':
    # UIUC Apartement router
    mininet_server_ip = "173.230.117.220"

    # UIUC campus  
    #cluster_head_ip = "10.195.111.177"

    #cluster_head_ip = 192.168.0.162
    # cluster_head_ip = "10.251.162.82"
    mininet_sever_port = 5000

    sensor_ids = range(5,15)
    agent = WSN_agent(num_clusters=1, sensor_ids=sensor_ids, sampling_freq=4, transmission_size=int(2*1500), transmission_frame_duration=1, file_lines_per_chunk=1, observation_time=0.5, BATCH_SIZE=256, num_episodes=1, max_steps=3000, LR=0.25e-6, train_every=500, local_mininet=False, server_ip=mininet_server_ip, server_port=mininet_server_port)
    agent.train()
    
