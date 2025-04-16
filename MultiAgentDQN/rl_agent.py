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
from models import DQN 

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

class WSN_agent:
    def __init__(self, num_clusters, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.7, EPS_START = 1, EPS_END = 0.0, EPS_DECAY = 400, TAU = 0.01, LR = 0.25e-3, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=512, local_mininet_simulation=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port) 
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

        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'clique']
        self._policy_net = {}
        self._target_net = {}
        self._optimizer = {}
        self._loss = {}
        self._rewards = {}

        for reward_type in self._reward_types:
            self._policy_net[reward_type] = [DQN(sampling_freq, self._n_observations, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            self._target_net[reward_type] = [DQN(sampling_freq, self._n_observations, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            
            self._train_every[reward_type] = train_every
            self._optimizer[reward_type] = [optim.AdamW(self._policy_net[reward_type][agent].parameters(), lr=LR, amsgrad=True) for agent in range(self._num_sensors)]
            self._loss[reward_type] = [[] for _ in range(self._num_sensors)]
            #self._rewards[reward_type][agent] = []

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
            clique_reward = torch.stack(batch.clique_reward)[non_final_mask]
                        
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
                    reward = clique_reward[:, agent]

                predicted_q_values = (next_state_values * self._GAMMA).squeeze() + reward

                # Calculate Loss 
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, predicted_q_values)

                # Log loss
                self._loss[reward_type][agent].append(loss.item())
                total_loss[reward_type] += loss.item()

                
                #with open('loss.pkl', 'wb') as file:
                #    pickle.dump(self._loss, file)
                #print(f"{agent} {reward_type} Loss: {loss.item():.4f}")

                # Optimization step
                self._optimizer[reward_type][agent].zero_grad()
                loss.backward()

                # Clip graidents
                #torch.nn.utils.clip_grad_value_(self._policy_net[reward_type].parameters(), 100)
                self._optimizer[reward_type][agent].step()

        print(f"Average throughput loss: {(total_loss['throughput'] / self._num_sensors):.4f}")
        print(f"Average clique loss: {(total_loss['clique'] / self._num_sensors):.4f}")

            
    def _select_action(self):
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * math.exp(-1. * self._steps_done / self._EPS_DECAY)
        self._steps_done += 1
        action = None

        #energy = self._state[:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        #energy = energy.squeeze()
        #dead_sensors = (energy <= self._recharge_thresh)
        #awake_sensors = (energy > self._recharge_thresh)
        
        samples = torch.rand(self._num_sensors, device=device)
        exploration_mask = samples <= eps_threshold
        policy_mask = ~exploration_mask
 
        actions = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.int, device=device)

        with torch.no_grad():
            action_values = [self._policy_net['throughput'][agent](self._state) + self._policy_net['clique'][agent](self._state) for agent in range(self._num_sensors)]
            action_values = torch.stack(action_values) 
            action_probs = F.softmax(action_values, dim=1)
            policy_actions = torch.argmax(action_probs, dim=1)
            policy_actions = torch.round(policy_actions).int().squeeze()

        print("Policy mask: ", policy_mask)

        actions[policy_mask] = policy_actions[policy_mask]
        #for agent in range(self._num_sensors):
        #    sample = random.random()
        #    if sample > eps_threshold:
        #        # Sample an action using the policy 
        #        action = self._actor_net[agent](self._state.view(-1))
        #        action = torch.round(action).int()
        #        actions[agent] = action
        #    else:
        #        actions[agent] = torch.randint(0, self._sampling_freq, (1), dtype=torch.long, device=device) 
           
        print(f'action: {action}')
        # action[dead_sensors] = 0 # Dont allow dead sensors to transmit
        # action[awake_sensors & (action==0)] = 1 # Force awake sensors to transmit

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
                observation, throughput_reward, clique_reward, terminated, truncated, _ = self._env.step(self._steps_done, action)
                
                print(f'Throughput reward: {throughput_reward}')
                print(f'Clique reward: {clique_reward}')
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_reward = torch.tensor(throughput_reward, device=device)
                clique_reward = torch.tensor(clique_reward, device=device)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    #next_state = observation.clone().detach()
                    next_state = observation.clone().detach()

                # Store the transition in memory
                self._memory.push(self._state, action, next_state, throughput_reward, clique_reward)

                # Move to the next state
                self._state = next_state
               
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
    # UIUC Apartement router
    mininet_server_ip = "10.251.163.138"

    # UIUC campus  
    #cluster_head_ip = "10.195.111.177"

    #cluster_head_ip = 192.168.0.162
    # cluster_head_ip = "10.251.162.82"
    mininet_server_port = 5000

    sensor_ids = range(5,15)
    agent = WSN_agent(num_clusters=1, sensor_ids=sensor_ids, sampling_freq=4, transmission_size=int(2*1500), transmission_frame_duration=1, file_lines_per_chunk=1, observation_time=1, BATCH_SIZE=128, num_episodes=1, max_steps=3000, LR=0.25e-3, train_every=500, local_mininet_simulation=True, server_ip=mininet_server_ip, server_port=mininet_server_port)
    agent.train()
    
