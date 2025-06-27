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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch.distributions.categorical import Categorical 

from WSN_env import WSNEnvironment
from memory import ReplayMemory, Transition, PPOMemory
from configs import Mininet_Simulation_Config, Multi_Agent_PPO_Config
from models import DDQN, CriticNetwork, ActorNetwork 

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )

class WSN_agent:
    def __init__(self, mininet_simulation_parameters, config):
        self._config = config 
        self._env = WSNEnvironment(mininet_simulation_parameters, max_steps, device) 
        self._num_sensors = len(mininet_simulation_parameters.sensor_ids)
        self._sampling_freq = mininet_simulation_parameters.sampling_freq
        self._transmission_rates = [20, 40, 60, 80]

        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        #self._n_observations = [self._num_sensors, self._num_sensors + 2]

        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'similarity']
        self._policy_net = {}
        self._target_net = {}
        self._loss = {}
        
        # Iniatlize Nerual Networks
        self._critic_net = {}
        for agent in range(self._num_sensors):
            for reward_type in self._reward_types:
                self._critic_net[reward_type] = [CriticNetwork(self._n_observations, 1, config.lr, device).to(device) for _ in range(self._num_sensors)]
                self._loss[reward_type] = [[] for _ in range(self._num_sensors)]

        self._actor_net = [ActorNetwork(sampling_freq, self._n_observations, config.lr, device).to(device) for agent in range(self._num_sensors)]

        # Replay memory for trainning
        self._memory = [PPOMemory(batch_size=self._config.batch_size) for _ in range(self._num_sensors)]

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []

        time.sleep(20)
        self._state, self._info = self._env.reset()

    def _optimize_model(self):
        #if len(self._memory[0]) < self._training_params.BATCH_SIZE:
        #    return
         
        total_loss_log = {r:0 for r in self._reward_types}
        total_loss = 0

        for agent in range(self._num_sensors):
            for _ in range(self._config.n_epochs):
                # Sample a batch of transitions 
                transitions, batches = self._memory[agent].generate_batches()
                transitions = Transition(*zip(*transitions))

                #non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
                #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
                #non_final_next_states = non_final_next_states.view(-1, 12*self._num_sensors)

                states = torch.stack(transitions.state).view(-1, 12*self._num_sensors)
                actions = torch.stack(transitions.action)

                throughput_reward = torch.stack(transitions.throughput_reward)
                similarity_reward = torch.stack(transitions.similarity_reward)
                reward_len = len(throughput_reward)
                
                values = torch.stack(transitions.value)
                #print(values)
                # values = torch.tensor(values).to(self._actor.device)
                n_states = states.shape[0]
                            
                for reward_type in self._reward_types:
                    if reward_type == 'throughput':
                        reward = throughput_reward
                    else:
                        reward = similarity_reward
                    
                    advantage = np.zeros(len(reward), dtype=np.float32)
                    for t in range(reward_len - 1):
                        discount = 1
                        a_t = 0.0
                        for k in range(t, reward_len - 1):
                            #a_t += discount*(reward[k] + self._config.GAMMA*values[k+1]*(1-int(dones_arr[k])) - values[k])

                            a_t += discount*(reward[k] + self._config.gamma*values[k+1]*(1) - values[k])
                            discount *= self._config.gamma* self._config.gae_lambda
                        advantage[t] = a_t
                    advantage = torch.tensor(advantage).to(self._actor_net[agent].device)

                    for batch in batches:
                        #states = torch.tensor(states_arr[batch], dtype=torch.float).to(self._actor.device)
                        old_probs = torch.tensor(transitions.probs)[batch].to(self._actor_net[agent].device)
                        #actions = torch.tensor(actions_arr[batch]).to(self._actor.device)

                        dist = self._actor_net[agent](states[batch])
                        critic_value = self._critic_net[reward_type][agent](states[batch])
                        critic_value = torch.squeeze(critic_value)

                        new_probs = dist.log_prob(actions[batch])
                        prob_ratio = new_probs.exp() / old_probs.exp()

                        weighted_probs = advantage[batch] * prob_ratio
                        weighted_clipped_probs = torch.clamp(prob_ratio, (1-self._config.policy_clip)*torch.ones_like(prob_ratio),
                                1 + self._config.policy_clip*advantage[batch])
                        actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                        returns = advantage[batch] + values[batch]
                        critic_loss = (returns - critic_value)**2
                        critic_loss = critic_loss.mean()

                        total_loss += actor_loss + 0.5*critic_loss
                        #total_loss_log[reward_type] += total_loss
                        self._actor_net[agent].optimizer.zero_grad()
                        self._critic_net[reward_type][agent].optimizer.zero_grad()
                        self._actor_net[agent].optimizer.step()
                        self._critic_net[reward_type][agent].optimizer.step()

            self._memory[agent].clear()

        print(f"Average throughput loss: {(total_loss / self._num_sensors):.4f}")
        #print(f"Average similarity loss: {(total_loss['similarity'] / self._num_sensors):.4f}")
        #with open(f'loss.pkl', 'wb') as file:
        #    pickle.dump((self._config.BATCH_SIZE, self._loss), file)
            
    def _select_action(self):
        action = torch.zeros(self._num_sensors, dtype=torch.int, device=device)
        value = [torch.zeros(self._num_sensors, dtype=torch.float, device=device) for _ in range(self._num_sensors)]
        probs = [torch.zeros(self._sampling_freq, dtype=torch.float, device=device) for _ in range(self._num_sensors)]
        
        for sensor in range(self._num_sensors):
            dist = self._actor_net[sensor](self._state)
            #value[sensor] = self._critic['throughput'][sensor](self._state) + self._critic['similarity'][sensor](self._state)
            value[sensor] = self._critic_net['similarity'][sensor](self._state)
            action[sensor] = dist.sample()
            probs[sensor] = torch.squeeze(dist.log_prob(action[sensor])).item()
            # action currently has indicies. Switch to rates
            action[sensor] = self._transmission_rates[action[sensor]]

        print(f'action: {action}')
        return action, probs, value
    
    def train(self):
        train_steps = 0
        throughput_reward_log = [-1 for _ in range(10)]
        for i_episode in range(self._config.num_episodes):
            # Initialize the environment and get its state
            print(self._env.reset())
            self._state, self._info = self._env.reset()

            print(f'State = {self._state}')
            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                print('Taking next step')

                action, probs, value = self._select_action()                    
                
                # Sample the next frame from the enviornment, and receive a reward
                observation, rewards, terminated, truncated, _ = self._env.step(self._steps_done, action)
                
                print(f'Throughput reward: {rewards["throughput"][0]}')
                print(f'Similarity reward: {rewards["similarity"]}')
                
                throughput_reward_log.append(rewards["throughput"][0])
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_reward = torch.tensor(rewards["throughput"], device=device, dtype=torch.float32)
                similarity_reward = torch.tensor(rewards["similarity"], device=device, dtype=torch.float32)

                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation.clone().detach()
                
                for i in range(self._num_sensors):
                    # Store the transition in memory
                    # Normalize actions
                    self._memory[i].push(self._state, action[i]/20 - 1, probs[i], value[i], next_state, throughput_reward[i], similarity_reward[i])

                # Move to the next state
                self._state = next_state
                
                
                train_steps += 1                
                with torch.no_grad():
                    if train_steps >= self._config.train_every:
                        train_steps = 0

                        # Perform one step of the optimization (on the policy network)
                        self._optimize_model()
                
                if done:
                    self._episode_durations.append(t + 1)
                    break
                
                print('Step done')
                #for reward_type in self._reward_types:
                    #self._train_steps[reward_type] += 1
        
if __name__ == '__main__':
    # RL parametrs
    batch_size = 64 
    gamma = 0.99
    max_steps = 9999 
    lr = 0.25e-2
    n_epochs = 8 
    train_every = 1028
    training_config = Multi_Agent_PPO_Config(batch_size=batch_size, max_steps=max_steps, lr=lr, gamma=gamma, n_epochs=n_epochs, train_every=train_every)
    
    # Simulation parmaters
    sensor_ids = range(5,15)
    sampling_freq = 4
    transmission_size = 2*1500
    observation_time = 1
    local_mininet_simulation = False 
    server_ip = "192.168.1.114" # IP of mininet simulation; ignored if local_mininet_simulation = True
    server_port = 5000 # Ignored if local_mininet_simulation = True
    mininet_simulation_config = Mininet_Simulation_Config(sensor_ids, sampling_freq=sampling_freq, transmission_size=transmission_size, observation_time=observation_time, local_simulation=local_mininet_simulation, remote_simulation_ip=server_ip, remote_simulation_port=server_port)

    agent = WSN_agent(mininet_simulation_config, training_config)
    agent.train()
