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
        self._sampling_freq = sampling_freq
        self._num_sensors = num_sensors
        w = 1024
        layers = [nn.Linear(n_observations, w), nn.ReLU()]
        num_hidden_layers = 13
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(w,w))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(w, num_sensors * sampling_freq))
        layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        # Initalize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity='relu')
                init.constant_(m.bias, 0)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x):
        #x = F.relu(self.first_layer(x))
        #for i in range(self.num_hidden_layers):
        #    x = F.relu(self.hidden_layers[i](x))
        #x = F.relu(self.last_layer(x))
        
        return self.layers(x).to(device)

class WSN_agent:
    def __init__(self, num_clusters, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.99, EPS_START = 0.9, EPS_END = 0.3, EPS_DECAY = 1500, TAU = 0.005, LR = 0.001, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=8):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes)
        
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq
        self._n_observations = self._num_sensors*self._num_sensors + 3*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_clusters = num_clusters
        self._num_episodes = num_episodes
        self._train_every = train_every

        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        num_actions = self._env.action_space.shape[0] # All envs have the same action and observation space
        self._policy_net = DQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
        self._target_net = DQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = ReplayMemory(max_steps*num_episodes)

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []
        self._rewards = []
        self._loss = []

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
                action_values = self._policy_net(self._state)
                action_values = action_values.view(self._num_sensors, self._sampling_freq)
                #print(action_values)
                action_probs = F.softmax(action_values, dim=1)
                print(action_probs)
                action = torch.argmax(action_probs, dim=1)
                print(action)
                #action = action.squeeze(0)
        else:
            print('Getting random action')
            action = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.long, device=device) 
       
        print(f'action: {action}')
        action[dead_sensors] = 0 # Dead sensors cannot transmit
        # action[awake_sensors & (action==0)] = 1 # Awake sensors must transmit

        return action

    '''
    For testing purposes only
    '''
    def _select_best_action(self):
        energy = self._state[0][:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        energy = energy.squeeze()
        dead_sensors = (energy <= self._recharge_thresh)
        awake_sensors = (energy > self._recharge_thresh)

        print('Getting best action')
        with torch.no_grad():
            action_values = self._policy_net(self._state[0])
            action_values= action_values.view(1, self._num_sensors, 4)
            action_probs = F.softmax(action_values, dim=-1)
            action = torch.argmax(action_probs, dim=-1)
            action = action.squeeze(0)

        self._steps_done[0] += 1
        action[dead_sensors] = 0
        action[awake_sensors & (action==0)] = 1

        return action

    def _optimize_model(self):
        if len(self._memory) < self._BATCH_SIZE:
            print(f'Mem len: {len(self._memory)}')
            return

        # Get a batch of transitions
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
        reward_batch = torch.cat(batch.reward).view(-1, 1)

        # Get the Q-values corresponding to the actions
        state_action_values = self._policy_net(state_batch).view(-1, self._num_sensors, self._sampling_freq).gather(2, action_batch.unsqueeze(1))
        # Remove any unneccesary dimensions
        state_action_values = state_action_values.squeeze()
        
        next_state_values = torch.zeros(self._BATCH_SIZE, self._num_sensors, device=device)
        with torch.no_grad():
            max_values = self._target_net(non_final_next_states).view(-1, self._num_sensors, self._sampling_freq).max(2)[0]
            next_state_values[non_final_mask] = max_values.view(-1, self._num_sensors)

        expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        self._loss.append(loss.item())
        with open('loss.pkl', 'wb') as file:
            pickle.dump(self._loss, file)

        print(f"Loss: {loss.item():.4f}")

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
            self._state = torch.tensor(self._state, dtype=torch.float32, device=device).unsqueeze(0)
            print(f'State = {self._state}')

            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                action = torch.Tensor.cpu(self._select_action())

                print('Taking next step')
                observation, reward, terminated, truncated, _ = self._env.step(self._steps_done, action.numpy())
                self._rewards.append(reward)
                print(f'Reward: {reward}')
                
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation.unsqueeze(0).clone().detach()

                # Store the transition in memory
                self._memory.push(self._state, action, next_state, reward)

                # Move to the next state
                self._state = next_state
               
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                if self._train_steps >= self._train_every:
                    self._train_steps = 0

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
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
    sensor_ids = range(5,15)
    agent = WSN_agent(num_clusters=1, sensor_ids=sensor_ids, sampling_freq = 4, transmission_size=int(1024*3), transmission_frame_duration=1, file_lines_per_chunk=1, observation_time=10, BATCH_SIZE=512, num_episodes=300, LR=0.1e-5, train_every=500)
    agent.train()
    
