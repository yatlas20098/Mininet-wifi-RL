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
from model import Actor, Critic

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
        num_hidden_layers = 128 

        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, n_actions))
        layers.append(nn.ReLU())

        self._layers = nn.Sequential(*layers)

        # Initalize random weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        return self._layers(x).to(device)

class DDQN(nn.Module):
    def __init__(self, sampling_freq, n_observations, n_actions, num_sensors):
        super(DDQN, self).__init__()

        w = 1024 # number of nodes in a hidden layer
        num_hidden_layers = 64 

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

        self._fc1 = nn.Linear(w, 512)
        std = math.sqrt(2.0 / (64 * 64 * 3 * 3))
        nn.init.normal_(self._fc1.weight, mean=0.0, std=std)
        self._fc1.bias.data.fill_(0.0)
        self._V = nn.Linear(512, 1)
        self._A = nn.Linear(512, n_actions)

     # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        x = self._layers(x).to(device)
        x = F.relu(self._fc1(x))

        V = self._V(x)
        A = self._A(x)
        Q = V + (A - A.mean(dim=0, keepdim=True))

        return Q

class WSN_agent:
    def __init__(self, num_clusters, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.99, EPS_START = 1, EPS_END = 0.0, EPS_DECAY = 900, TAU = 0.01, LR = 0.25e-3, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=8, local_mininet_simulation=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port) 
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq # Number of possible transmission frequencies
        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_clusters = num_clusters
        self._num_episodes = num_episodes
        self._train_every = train_every # How many steps between updates of target network 

        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        num_actions = self._num_sensors * sampling_freq 
        self._policy_net = DDQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device) # DQN or DDQN
        self._target_net = DDQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = ReplayMemory(max_steps*num_episodes)

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []
        self._rewards = []
        self._loss = []

        self._episode_durations = []
        self._energy_consumption = []

        self._BATCH_SIZE = BATCH_SIZE 
        self._GAMMA = GAMMA 
        self._EPS_START = EPS_START
        self._EPS_END = EPS_END 
        self._EPS_DECAY = EPS_DECAY
        self._TAU = TAU

        self._state, self._info = self._env.reset()

    def _select_action(self):
        sample = random.random()
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * math.exp(-1. * self._steps_done / self._EPS_DECAY)
        self._steps_done += 1
        action = None

        #energy = self._state[:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        #energy = energy.squeeze()
        #dead_sensors = (energy <= self._recharge_thresh)
        #awake_sensors = (energy > self._recharge_thresh)

        if sample > eps_threshold:
            print('Getting best action')
            with torch.no_grad():
                action_values = self._policy_net(self._state.view(-1))
                action_values = action_values.view(self._num_sensors, self._sampling_freq)
                action_probs = F.softmax(action_values, dim=1)
                action = torch.argmax(action_probs, dim=1)
        else:
            print('Getting random action')
            action = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.long, device=device) 
       
        print(f'action: {action}')
        
        # action[dead_sensors] = 0 # Dont allow dead sensors to transmit
        # action[awake_sensors & (action==0)] = 1 # Force awake sensors to transmit

        return action

    def _optimize_model(self):
        if len(self._memory) < self._BATCH_SIZE:
            print(f'Mem len: {len(self._memory)}')
            return
        
        # Sample a batch of transitions 
        transitions = self._memory.sample(self._BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Get the non-final states in the batch
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        non_final_next_states = non_final_next_states.view(-1, 12*self._num_sensors)

        state_batch = torch.stack(batch.state)[non_final_mask]
        action_batch = torch.stack(batch.action)[non_final_mask]
       
        # Get the rewards from the batch of transitions
        reward_batch = torch.cat(batch.reward)

        # Get the Q-values corresponding to the actions
        state_action_values = self._policy_net(state_batch).view(-1, self._num_sensors, self._sampling_freq).gather(2, action_batch.unsqueeze(2)).squeeze()

        # Calculate expected Q-values
        max_values = self._target_net(non_final_next_states)
        next_state_values = torch.zeros(self._BATCH_SIZE, self._num_sensors, device=device)
        with torch.no_grad():
            max_values = self._target_net(non_final_next_states.view(-1, 12*self._num_sensors)).view(-1, self._num_sensors, self._sampling_freq).max(2)[0]
            next_state_values[non_final_mask] = max_values
        expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch.unsqueeze(1)
        expected_state_action_values = expected_state_action_values.squeeze()

        # Calculate Loss 
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        # Log loss
        self._loss.append(loss.item())
        with open('loss.pkl', 'wb') as file:
            pickle.dump(self._loss, file)
        print(f"Loss: {loss.item():.4f}")

        # Optimization step
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()
    
    def train(self):
        self._train_steps = 0

        for i_episode in range(self._num_episodes):
            # Initialize the environment and get its state
            print(self._env.reset())
            self._state, self._info = self._env.reset()
            self._state = torch.tensor(self._state, dtype=torch.float32, device=device)

            print(f'State = {self._state}')
            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                print('Taking next step')

                # Sample an action using the policy 
                action = self._select_action()

                # Sample the next frame from the enviornment, and receive a reward
                observation, reward, terminated, truncated, _ = self._env.step(self._steps_done, action)
                
                # Log the reward
                self._rewards.append(reward)
                print(f'Reward: {reward}')
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation.clone().detach()

 # Store the transition in memory
                self._memory.push(self._state, action, next_state, reward)

                # Move to the next state
                self._state = next_state
               
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                if self._train_steps >= self._train_every:
                    self._train_steps = 0

                    # Soft update of the target network's weights
                    target_net_state_dict = self._target_net.state_dict()
                    policy_net_state_dict = self._policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                    self._target_net.load_state_dict(target_net_state_dict)

                if done:
                    self._episode_durations.append(t + 1)
                    break
                print('Step done')

                self._train_steps += 1

if __name__ == '__main__':
    mininet_server_ip = "0.0.0.0" # IP of device running local mininet simulation
    mininet_server_port = 5000

    sensor_ids = range(5,15)
    agent = WSN_agent(num_clusters=1, sensor_ids=sensor_ids, sampling_freq=4, transmission_size=int(2*1500), transmission_frame_duration=1, file_lines_per_chunk=1, observation_time=1, BATCH_SIZE=128, num_episodes=1, max_steps=3000, LR=0.25e-3, train_every=8, local_mininet_simulation=True, server_ip=mininet_server_ip, server_port=mininet_server_port)
    agent.train()
    
