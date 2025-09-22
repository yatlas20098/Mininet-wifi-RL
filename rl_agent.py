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
import multiprocessing
from torch_geometric.loader import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.init as init
from torch.distributions.categorical import Categorical 
from torch_geometric.data import Batch

from WSN_env import WSNEnvironment
from memory import ReplayMemory, Transition, PPOMemory
from configs import Mininet_Simulation_Config, Multi_Agent_PPO_Config
from models import DDQN, CriticNetwork, ActorNetwork 

import os
torch.set_num_threads(os.cpu_count())

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
        )

def _save_agent(reward_types, actor_net, critic_net):
    return
    # Save the actor network
    torch.save({
        'model_state_dict': actor_net.state_dict(),
        'optimizer_state_dict': actor_net.optimizer.state_dict(),
    }, f'saved_agents/actor_net.pth')

    for reward_type in reward_types:
        critic_net_r = critic_net[reward_type]
        
        # Save the critic network
        torch.save({
            'model_state_dict': critic_net_r.state_dict(),
            'optimizer_state_dict': critic_net_r.optimizer.state_dict(),
        }, f'saved_agents/critic_net{reward_type}.pth')

def _load_critic(reward_types, sampling_freq, n_obs, critic_lr):
    critic_net = {reward_type: CriticNetwork(n_obs, 1, critic_lr, device).to(device) for reward_type in reward_types}

    for reward_type in reward_types:
        # Load Critic Network (model and optimizer states)
        checkpoint_critic = torch.load(f'saved_agents/critic_net{reward_type}.pth')
        critic_net[reward_type].load_state_dict(checkpoint_critic['model_state_dict'])
        critic_net[reward_type].optimizer.load_state_dict(checkpoint_critic['optimizer_state_dict'])

    return critic_net


def _load_agent(num_sensors, reward_types, sampling_freq, n_obs, actor_lr):
    actor_net = ActorNetwork(num_sensors, sampling_freq, n_obs, actor_lr, device).to(device)
    # Load Actor Network (model and optimizer states)
    checkpoint_actor = torch.load(f'saved_agents/actor_net.pth')
    actor_net.load_state_dict(checkpoint_actor['model_state_dict'])
    actor_net.optimizer.load_state_dict(checkpoint_actor['optimizer_state_dict'])
   
    return actor_net

#def _optimize_model_for_agent(config, agent, memory, num_sensors, reward_types, sampling_freq, n_obs, actor_lr, critic_lr):

class WSN_agent:
    def __init__(self, mininet_simulation_parameters, config):
        self._config = config 
        self._sim_config = mininet_simulation_parameters
        self._env = WSNEnvironment(mininet_simulation_parameters, max_steps, device) 
        self._num_sensors = len(mininet_simulation_parameters.sensor_ids)
        self._sampling_freq = mininet_simulation_parameters.sampling_freq
        min_ids = min(self._sim_config.sensor_ids)
        self._agent_ids = [s_id - min_ids for s_id in self._sim_config.sensor_ids]
        self._agent_ids = torch.tensor(self._agent_ids)
        #self._transmission_rates = mininet_simulation_parameters.transmission_rates

        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        #self._n_observations = [self._num_sensors, self._num_sensors + 2]

        # Seperate policy network for both reward types
        self._reward_types = ['throughput', 'similarity']
        self._policy_net = {}
        self._target_net = {}
        self._loss = {}

        load_agent = True 
        n_obs = 2

        if load_agent:
            print("Loading agents")
            #self._actor_net, self._critic_net = zip(*[_load_agent(agent, self._reward_types, sampling_freq, n_obs, config.actor_lr, config.critic_lr) for agent in range(self._num_sensors)])

            self._actor_net = _load_agent(self._num_sensors, self._reward_types, sampling_freq, n_obs, config.actor_lr)
            self._critic_net = _load_critic(self._reward_types, sampling_freq, n_obs, config.critic_lr)
        else:
            # Iniatlize Nerual Networks
            self._critic_net = {reward_type: CriticNetwork(n_obs, 1, config.critic_lr, device).to(device) for reward_type in self._reward_types} 
            self._actor_net = ActorNetwork(self._num_sensors, sampling_freq, n_obs, config.actor_lr, device).to(device)
        
        # Replay memory for trainning
        self._memory = [PPOMemory(batch_size=self._config.batch_size) for _ in range(self._sim_config.num_clusters)]

        self._steps_done = 0
        self._episode_durations = []
        self._energy_consumption = []

        self._state, self._info = self._env.reset()

    def _save_agent(self):
        # Save the actor network
        torch.save({
            'model_state_dict': self._actor_net.state_dict(),
            'optimizer_state_dict': self._actor_net.optimizer.state_dict(),
        }, f'saved_agents/actor_net.pth')

    def _save_critic(self):
        for reward_type in self._reward_types:
            critic_net_r = self._critic_net[reward_type]
            
            # Save the critic network
            torch.save({
                'model_state_dict': critic_net_r.state_dict(),
                'optimizer_state_dict': critic_net_r.optimizer.state_dict(),
            }, f'saved_agents/critic_net{reward_type}.pth')

    def _optimize_model_for_cluster(self, cluster):
        total_loss = 0
        for epoch in range(self._config.n_epochs):
            minibatches = []
            critic_total_loss = 0
            actor_total_loss = 0

            transitions, batches = self._memory[cluster].generate_batches()
            transitions = Transition(*zip(*transitions))

            states_arr = transitions.state
            actions_arr = torch.stack(transitions.action).squeeze()
            values_arr = torch.stack(transitions.value).squeeze()
            old_probs_arr = torch.stack(transitions.probs).squeeze()
            dones_arr = torch.tensor(transitions.done).squeeze()

            throughput_reward = torch.stack(transitions.throughput_reward).squeeze()
            similarity_reward = torch.stack(transitions.similarity_reward).squeeze()

            for batch in batches:
                states = [states_arr[i] for i in batch]
                states = Batch.from_data_list(states).to(device)
                last_state = states_arr[batch[-1]]

                actions = actions_arr[batch].view(-1)
                values = values_arr[batch].view(-1)
                old_probs = old_probs_arr[batch].view(-1)
                dones = dones_arr[batch].float()
                                        
                for reward_type in self._reward_types:
                    if reward_type == 'throughput':
                        reward = throughput_reward[batch]
                    else:
                        # Ignore similarity for now
                        continue 
                        reward = similarity_reward[batch]

                    with torch.no_grad():
                        last_value = self._critic_net[reward_type](last_state).squeeze()
                        values = torch.cat([values, last_value.unsqueeze(0)], dim=0)
                        # Note that deltas cannot be accurately computed for 
                        # terminated states in our IoT env. Env changes are 
                        # effectivly random and independent of actions. Thus, 
                        # when training, we ignore terminated states.
                        deltas = (reward + self._config.gamma*values[1:] - values[:-1])*(1-dones)
                        advantage = torch.zeros_like(deltas)

                    gae = 0.0
                    for t in reversed(range(len(deltas))):
                        mask = 1.0 - dones[t]
                        gae = deltas[t] + self._config.gamma*self._config.gae_lambda*gae*mask
                        advantage[t] = gae
                
                    norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                    #norm_advantage = advantage

                    dists = self._actor_net(states, self._agent_ids)
                    critic_value = self._critic_net[reward_type](states)
                    critic_value = torch.squeeze(critic_value)
                    
                    new_probs = dists.log_prob(actions.view(-1))
                    prob_ratio = torch.exp(new_probs - old_probs).reshape(self._config.batch_size, -1)
                    
                    weighted_probs = norm_advantage.unsqueeze(1) * prob_ratio
                    weighted_clipped_probs = torch.clamp(prob_ratio, 
                                                            (1.0 - self._config.policy_clip)*torch.ones_like(prob_ratio),
                                                            (1.0 + self._config.policy_clip)*torch.ones_like(prob_ratio))*norm_advantage.unsqueeze(1)
                    actor_loss =- torch.min(weighted_probs, weighted_clipped_probs).mean()
                    entropy = dists.entropy().mean()
                    self._actor_net.optimizer.zero_grad()
                    (actor_loss - self._config.entropy_coef*entropy).backward()
                    torch.nn.utils.clip_grad_norm_(self._actor_net.parameters(), max_norm=1.0)
                    self._actor_net.optimizer.step()

                    # print("Advantage: ", norm_advantage[:10])
                    # print("Values: ", values[:10])
                    # print("Rewards: ", reward[:10])
                
                    returns = (advantage + values[:-1]).detach()
                    critic_loss = (returns - critic_value)**2 
                    critic_loss = critic_loss.mean()
                    self._critic_net[reward_type].optimizer.zero_grad()
                    (self._config.value_loss_coef*critic_loss).backward()
                    torch.nn.utils.clip_grad_norm_(self._critic_net[reward_type].parameters(), max_norm=1.0)
                    self._critic_net[reward_type].optimizer.step()
                    
                    actor_total_loss += actor_loss
                    critic_total_loss += critic_loss

                    total_loss = actor_loss + critic_loss
            
            batch_len = self._config.train_every // self._config.batch_size
            print(f"Epoch {epoch}")
            print("\tCritic loss: ", critic_total_loss / batch_len / self._num_sensors)
            print("\tActor loss: ", actor_total_loss / batch_len) 
            print("")
        # Save the network
        self._save_agent()
        self._save_critic();

    def _optimize_model(self):
        self._optimize_model_for_cluster(0)
        self._memory[0].clear()

    def _select_action(self):
        actions, values, probs = [], [], []

        for i in range(self._sim_config.num_clusters):
            dists = self._actor_net(self._state[i], self._agent_ids)
            value = self._critic_net['throughput'](self._state[i]).detach()
            action = dists.sample()
            prob = dists.log_prob(action).detach()

            actions.append(action)
            values.append(value)
            probs.append(prob)
        return actions, probs, values
    
    def train(self):
        train_steps = -1
        throughput_reward_log = [-1 for _ in range(10)]
        init_actor_lr = self._config.actor_lr         
        init_critic_lr = self._config.critic_lr
        init_entropy_coef = self._config.entropy_coef
        min_lr = 3e-8
        first_step = True
        prev_throughput_rewards, prev_similarity_rewards = None, None
        throughput_rewards, similarity_rewards = None, None

        for i_episode in range(self._config.num_episodes):
            # Initialize the environment and get its state
            self._state, self._info = self._env.reset()

            # for t in count():
            for t in range(self._config.max_steps):
                print(f'\n\nEpisode {i_episode}, Step: {t}, Buffer size: {len(self._memory[0])}/{self._config.train_every}')

                actions, probs, values = self._select_action()
                prev_state, prev_actions, prev_probs, prev_values = self._state, actions, probs, values
                
                # Sample the next frame from the enviornment, and receive a reward
                observations, rewards, terminated, truncated, _ = self._env.step(self._steps_done, actions)


                prev_throughput_rewards, prev_similarity_rewards = throughput_rewards, similarity_rewards

                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_rewards = [torch.tensor(reward["throughput"], device=device, dtype=torch.float32) for reward in rewards]
                similarity_rewards = [torch.tensor(reward["similarity"], device=device, dtype=torch.float32) for reward in rewards]
                
                done = False 
                next_state = [None for _ in range(self._sim_config.num_clusters)]
                for cluster_idx in range(self._sim_config.num_clusters):
                    # Note done signifies that the previous step was a termination or truncation.
                    # Thus, memory updates must be delayed by a step.
                    done = terminated[cluster_idx] or truncated[cluster_idx]
                                        
                    if not first_step:
                        # Store the transition in memory
                        self._memory[cluster_idx].push(prev_state[cluster_idx].clone(), done, prev_actions[cluster_idx], prev_probs[cluster_idx], prev_values[cluster_idx].detach(), next_state[cluster_idx], prev_throughput_rewards[cluster_idx].detach(), prev_similarity_rewards[cluster_idx].detach())

                    if not done:
                        next_state[cluster_idx] = observations[cluster_idx].clone().detach()

                # Move to the next state
                self._state = next_state
                first_step = False 
                
                train_steps += self._sim_config.num_clusters
                if len(self._memory[0]) >= self._config.train_every:
                    #train_steps = 0

                    # Perform one step of the optimization (on the policy network)
                    self._optimize_model()
                    #if(((t  + 1)// self._config.train_every) % 20): 
                    #    self._config.actor_lr = max(self._config.actor_lr / 10, min_lr/10)
                    #    self._config.critic_lr = max(self._config.critic_lr / 10, min_lr)

                if done:
                    self._episode_durations.append(t + 1)
                    break
                
if __name__ == '__main__':
    # RL parametrs
    batch_size = 10000 
    gamma = 0.95
    gae_lambda = 0.97
    max_steps = 1500
    critic_lr = 1e-3
    actor_lr = 5e-4
    n_epochs = 15 
    train_every = 20000

    training_config = Multi_Agent_PPO_Config(batch_size=batch_size, max_steps=max_steps, critic_lr=critic_lr, actor_lr=actor_lr, gamma=gamma, gae_lambda=gae_lambda, n_epochs=n_epochs, train_every=train_every)
    
    # Simulation parmaters
    num_clusters = 1
    sensor_ids = range(5,15)
    sampling_freq = 4
    transmission_size = 1*1024 # bytes (1 packet)
    observation_time = 0.04
    local_mininet_simulation = True 
    server_ip = "192.168.1.114" # IP of mininet simulation; ignored if local_mininet_simulation = True
    server_port = 5000 # Ignored if local_mininet_simulation = True
    sim_config = Mininet_Simulation_Config(sensor_ids, num_clusters=num_clusters, sampling_freq=sampling_freq, transmission_size=transmission_size, observation_time=observation_time, local_simulation=local_mininet_simulation, remote_simulation_ip=server_ip, remote_simulation_port=server_port)

    agent = WSN_agent(sim_config, training_config)
    agent.train()
