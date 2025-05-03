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
from models import Actor, Critic

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )
device = torch.device("cpu")

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
    def __init__(self, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, GAMMA = 0.7, EPS_START = 1, EPS_END = 0.0, EPS_DECAY = 400, TAU = 0.01, LR = 0.25e-3, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=512, local_mininet_simulation=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port) 
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq # Number of possible transmission frequencies
        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_episodes = num_episodes
        self._train_every = {} # How many steps between updates of target network 

        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        num_actions = self._num_sensors 
        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'clique']
        self._critic_net = {}
        self._critic_target_net = {}
        self._optimizer = {}
        self._loss = {}
        self._rewards = {}

        for reward_type in self._reward_types:
            self._critic_net[reward_type] = [Critic(sampling_freq, self._n_observations, num_actions, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            self._critic_target_net[reward_type] = [Critic(sampling_freq, self._n_observations, num_actions, self._num_sensors, device).to(device) for _ in range(self._num_sensors)]
            
            self._train_every[reward_type] = train_every
            self._optimizer[reward_type] = [optim.AdamW(self._critic_net[reward_type][agent].parameters(), lr=LR, amsgrad=True) for agent in range(self._num_sensors)]
            self._loss[reward_type] = [[] for _ in range(self._num_sensors)]
            #self._rewards[reward_type][agent] = []

        self._actor_net = [Actor(sampling_freq, self._n_observations, num_actions, self._num_sensors, device, min_freq = 0, max_freq=3).to(device) for _ in range(self._num_sensors)]
        self._actor_target_net = [Actor(sampling_freq, self._n_observations, num_actions, self._num_sensors, device, min_freq=0, max_freq=3).to(device) for _ in range(self._num_sensors)]
        self._optimizer['actor'] = [optim.AdamW(self._actor_net[agent].parameters(), lr=LR, amsgrad=True) for agent in range(self._num_sensors)]

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
        
        for agent in range(self._num_sensors):
            # Sample a batch of transitions 
            transitions = self._memory.sample(self._BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            # Get the non-final states in the batch
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
            non_final_next_states = non_final_next_states.view(-1, 12*self._num_sensors)

            #state_batch = torch.stack(batch.state)[non_final_mask]
            #action_batch = torch.stack(batch.action)[non_final_mask][:, agent]

            
            with torch.no_grad():
                next_actions = self._actor_net[agent](non_final_next_states)


            for reward_type in self._reward_types:
                with torch.no_grad():
                    target_q_values = self._critic_target_net[reward_type][agent](torch.cat((non_final_next_states, next_actions), -1))
                    target_q_values *= self._GAMMA
                    target_q_values = target_q_values.squeeze()
                    
                    if reward_type == 'throughput':
                        reward = torch.tensor([r[agent] for r in batch.throughput_reward], device=device, dtype=torch.float32) 
                    else:
                        reward = torch.tensor([r[agent] for r in batch.clique_reward], device=device, dtype=torch.float32) 

                    target_q_values += torch.tensor(reward, device=device, dtype=torch.float32).squeeze()

                predicted_q_values = self._critic_net[reward_type][agent](torch.cat((non_final_next_states, next_actions), -1)).squeeze()

                # Calculate Loss 
                criterion = nn.SmoothL1Loss()
                loss = criterion(target_q_values, predicted_q_values)

                # Log loss
                self._loss[reward_type][agent].append(loss.item())
                #with open('loss.pkl', 'wb') as file:
                #    pickle.dump(self._loss, file)
                print(f"{agent} {reward_type} Loss: {loss.item():.4f}")

                # Optimization step
                self._optimizer[reward_type][agent].zero_grad()
                loss.backward()

                # Clip graidents
                torch.nn.utils.clip_grad_value_(self._critic_net[reward_type][agent].parameters(), 1)
                self._optimizer[reward_type][agent].step()
            
            self._actor_net[agent].train()
            self._optimizer['actor'][agent].zero_grad()
            action_pred = self._actor_net[agent](non_final_next_states)
            q_values = self._critic_net['throughput'][agent](torch.cat((non_final_next_states, action_pred),-1)) + self._critic_net['clique'][agent](torch.cat((non_final_next_states, action_pred), -1))
            print(f"q_values: {q_values[:5]}")
            q_values = q_values.mean()
            print(f"Agent {agent} Actor Loss: {q_values:.4f}")
            q_values.backward()
            torch.nn.utils.clip_grad_value_(self._actor_net[agent].parameters(), 1)
            self._optimizer['actor'][agent].step()

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

        policy_actions = torch.stack([self._actor_net[agent](self._state) for agent in range(self._num_sensors)])
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
                    target_net_state_dict = self._actor_target_net[agent].state_dict()
                    policy_net_state_dict = self._actor_net[agent].state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                    self._actor_target_net[agent].load_state_dict(target_net_state_dict)
                
                    for reward_type in self._reward_types:
                        #if self._train_steps[reward_type] >= self._train_every[reward_type]:
                        #self._train_steps[reward_type] = 0
                        # Soft update of the target network's weights
                        target_net_state_dict = self._critic_target_net[reward_type][agent].state_dict()
                        policy_net_state_dict = self._critic_net[reward_type][agent].state_dict()
                        for key in policy_net_state_dict:
                            target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                        self._critic_target_net[reward_type][agent].load_state_dict(target_net_state_dict)

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
    local_mininet_simulation = True 
    server_ip = "10.251.169.23" # IP of mininet simulation; ignored if local_mininet_simulation = True
    server_port = 5000 # Ignored if local_mininet_simulation = True

    # RL parametrs
    BATCH_SIZE = 16 
    GAMMA = 0.70
    EPS_DECAY = 200 
    EPS_START = 1
    EPS_END = 0
    max_steps = 3000
    LR = 0.25e-2
    
    agent = WSN_agent(sensor_ids=sensor_ids, sampling_freq=sampling_freq, transmission_size=transmission_size, observation_time=observation_time, BATCH_SIZE=BATCH_SIZE, max_steps=max_steps, LR=LR, EPS_DECAY=EPS_DECAY, EPS_START=EPS_START, EPS_END=EPS_END, GAMMA=GAMMA, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port)
    agent.train()
    
