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
from models import DDQN 

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'throughput_reward', 'similarity_reward'))

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

class WSN_agent:
    def __init__(self, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.7, EPS_START = 1, EPS_END = 0.0, EPS_DECAY = 400, TAU = 0.01, LR = 0.25e-3, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=512, local_mininet_simulation=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port) 
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq # Number of possible transmission frequencies
        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_episodes = num_episodes
        # self._train_every = {} # How many steps between updates of target network 

        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'similarity']
        self._policy_net = {}
        self._target_net = {}
        self._optimizer = {}
        self._loss = {}

        for reward_type in self._reward_types:
            self._policy_net[reward_type] = [DDQN(sampling_freq, self._n_observations, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            self._target_net[reward_type] = [DDQN(sampling_freq, self._n_observations, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            
            self._optimizer[reward_type] = [optim.AdamW(self._policy_net[reward_type][agent].parameters(), lr=LR, amsgrad=True) for agent in range(self._num_sensors)]
            self._loss[reward_type] = [[] for _ in range(self._num_sensors)]

        memory_capacity = 256 
        self._memory = ReplayMemory(memory_capacity)

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

    def _optimize_model(self):
        if len(self._memory) < self._BATCH_SIZE:
            return
        
        total_loss = {r:0 for r in self._reward_types} 
        for agent in range(self._num_sensors):
            # Sample a batch of transitions 
            transitions = self._memory.sample(self._BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            # Get the non-final states in the batch
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_states = non_final_next_states.view(-1, 12*self._num_sensors)

            state_batch = torch.stack(batch.state)[non_final_mask]
            action_batch = torch.stack(batch.action)[non_final_mask][:, agent]

            throughput_reward = torch.stack(batch.throughput_reward)[non_final_mask]
            similarity_reward = torch.stack(batch.similarity_reward)[non_final_mask]
                        
            for reward_type in self._reward_types:
                state_action_values = self._policy_net[reward_type][agent](state_batch).gather(1, action_batch.long().unsqueeze(1)).squeeze()
                max_values = self._target_net[reward_type][agent](non_final_next_states)
                next_state_values = torch.zeros(self._BATCH_SIZE, device=device)
                with torch.no_grad():
                    max_values = self._target_net[reward_type][agent](non_final_next_states).max(1)[0]
                    next_state_values[non_final_mask] = max_values

                if reward_type == 'throughput':
                    reward = throughput_reward[:, agent]
                else:
                    reward = similarity_reward[:, agent]

                predicted_q_values = (next_state_values * self._GAMMA).squeeze() + reward

                # Calculate Loss 
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, predicted_q_values)

                # Log loss
                self._loss[reward_type][agent].append(loss.item())
                total_loss[reward_type] += loss.item()

                #print(f"{agent} {reward_type} Loss: {loss.item():.4f}")

                # Optimization step
                self._optimizer[reward_type][agent].zero_grad()
                loss.backward()

                # Clip graidents
                torch.nn.utils.clip_grad_value_(self._policy_net[reward_type][agent].parameters(), 1)
                self._optimizer[reward_type][agent].step()

        print(f"Average throughput loss: {(total_loss['throughput'] / self._num_sensors):.4f}")
        print(f"Average similarity loss: {(total_loss['similarity'] / self._num_sensors):.4f}")
        with open(f'loss.pkl', 'wb') as file:
            pickle.dump(self._loss, file)
            
    def _select_action(self):
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * math.exp(-1. * self._steps_done / self._EPS_DECAY)
        self._steps_done += 1
        action = None

        #if eps_threshold < 0.2:
        #    eps_threshold = 0

        #energy = self._state[:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        #energy = energy.squeeze()
        #dead_sensors = (energy <= self._recharge_thresh)
        #awake_sensors = (energy > self._recharge_thresh)
        # energy = self._state[:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        # energy = energy.squeeze()
        # dead_sensors = (energy <= self._recharge_thresh)
        # awake_sensors = (energy > self._recharge_thresh)
        
        samples = torch.rand(self._num_sensors, device=device)
        exploration_mask = samples <= eps_threshold
        policy_mask = ~exploration_mask
 
        actions = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.int, device=device)

        with torch.no_grad():
            #action_values = [self._policy_net['throughput'][agent](self._state) + self._policy_net['similarity'][agent](self._state) for agent in range(self._num_sensors)]
            action_values = [self._policy_net['throughput'][agent](self._state) for agent in range(self._num_sensors)]
            action_values = torch.stack(action_values)
            print(action_values)
            action_probs = F.softmax(action_values, dim=1)
            print(action_probs)
            policy_actions = torch.argmax(action_probs, dim=1)
            policy_actions = torch.round(policy_actions).int().squeeze()

        print("Policy mask: ", policy_mask)

        actions[policy_mask] = policy_actions[policy_mask]
        # action[dead_sensors] = 0 # Dont allow dead sensors to transmit
        # action[awake_sensors & (action==0)] = 1 # Force awake sensors to transmit

        print(f'action: {action}')
        
        return actions
    
    def train(self):
        for i_episode in range(self._num_episodes):
            # Initialize the environment and get its state
            print(self._env.reset())
            self._state, self._info = self._env.reset()

            print(f'State = {self._state}')
            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                print('Taking next step')

                action = self._select_action()                    
                
                # Sample the next frame from the enviornment, and receive a reward
                observation, throughput_reward, similarity_reward, terminated, truncated, _ = self._env.step(self._steps_done, action)
                
                print(f'Throughput reward: {throughput_reward}')
                print(f'Similarity reward: {similarity_reward}')
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_reward = torch.tensor(throughput_reward, device=device)
                similarity_reward = torch.tensor(similarity_reward, device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    #next_state = observation.clone().detach()
                    next_state = observation.clone().detach()

                # Store the transition in memory
                self._memory.push(self._state, action, next_state, throughput_reward, similarity_reward)

                # Move to the next state
                self._state = next_state
                
                if t > 400:
                    continue

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                for agent in range(self._num_sensors):
                    # Update target networks
                    for reward_type in self._reward_types:
                        #if self._train_steps[reward_type] >= self._train_every[reward_type]:
                        #self._train_steps[reward_type] = 0
                        # Soft update of the target network's weights
                        target_net_state_dict = self._target_net[reward_type][agent].state_dict()
                        policy_net_state_dict = self._policy_net[reward_type][agent].state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                        self._target_net[reward_type][agent].load_state_dict(target_net_state_dict)

                if done:
                    self._episode_durations.append(t + 1)
                    break
                
                print('Step done')
                #for reward_type in self._reward_types:
                    #self._train_steps[reward_type] += 1
        
if __name__ == '__main__':
    # Simulation parmaters
    sensor_ids = range(5,15)
    sampling_freq = 4
    transmission_size = 2*1500
    observation_time = 1
    local_mininet_simulation = False 
    server_ip = "10.251.169.23" # IP of mininet simulation; ignored if local_mininet_simulation = True
    server_port = 5000 # Ignored if local_mininet_simulation = True

    # RL parametrs
    BATCH_SIZE = 64 
    GAMMA = 0.70
    EPS_DECAY = 200 
    EPS_START = 1
    EPS_END = 0
    max_steps = 3000
    LR = 0.25e-2
    
    agent = WSN_agent(sensor_ids=sensor_ids, sampling_freq=sampling_freq, transmission_size=transmission_size, observation_time=observation_time, BATCH_SIZE=BATCH_SIZE, max_steps=max_steps, LR=LR, EPS_DECAY=EPS_DECAY, EPS_START=EPS_START, EPS_END=EPS_END, GAMMA=GAMMA, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port)
    agent.train()
    
