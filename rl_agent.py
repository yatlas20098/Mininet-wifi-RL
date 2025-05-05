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

class SumTree:
    def __init__(self, size):
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size

        self.size = size
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)

        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        #assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"

class PrioritizedReplayMemory:
    def __init__(self, buffer_size, eps=1e-2, alpha=0.1, beta=0.1):
        self.tree = SumTree(size=buffer_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = eps  # priority for new samples, init as eps

        # transition: state, action, reward, next_state, done
        self.memory = [None for _ in range(buffer_size)] 
        #self.state = torch.empty(buffer_size, state_size, dtype=torch.float)
        #self.action = torch.empty(buffer_size, action_size, dtype=torch.float)
        #self.reward = torch.empty(buffer_size, dtype=torch.float)
        #self.next_state = torch.empty(buffer_size, state_size, dtype=torch.float)
        #self.done = torch.empty(buffer_size, dtype=torch.int)

        self.count = 0
        self.real_size = 0
        self.size = buffer_size
    
    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        #self.memory.append(Transition(*args))
        self.memory[self.count] = Transition(*args)

        #state, action, reward, next_state, done = transition

        # store transition index with maximum priority in sum tree
        self.tree.add(self.max_priority, self.count)

        # store transition in the buffer
        #self.state[self.count] = torch.as_tensor(state)
        #self.action[self.count] = torch.as_tensor(action)
        #self.reward[self.count] = torch.as_tensor(reward)
        #self.next_state[self.count] = torch.as_tensor(next_state)
        #self.done[self.count] = torch.as_tensor(done)

        # update counters
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        #assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        # To sample a minibatch of size k, the range [0, p_total] is divided equally into k ranges.
        # Next, a value is uniformly sampled from each range. Finally the transitions that correspond
        # to each of these sampled values are retrieved from the tree. (Appendix B.2.1, Proportional prioritization)
        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)

        # Concretely, we define the probability of sampling transition i as P(i) = p_i^α / \sum_{k} p_k^α
        # where p_i > 0 is the priority of transition i. (Section 3.3)
        probs = priorities / self.tree.total

        # The estimation of the expected value with stochastic updates relies on those updates corresponding
        # to the same distribution as its expectation. Prioritized replay introduces bias because it changes this
        # distribution in an uncontrolled fashion, and therefore changes the solution that the estimates will
        # converge to (even if the policy and state distribution are fixed). We can correct this bias by using
        # importance-sampling (IS) weights w_i = (1/N * 1/P(i))^β that fully compensates for the non-uniform
        # probabilities P(i) if β = 1. These weights can be folded into the Q-learning update by using w_i * δ_i
        # instead of δ_i (this is thus weighted IS, not ordinary IS, see e.g. Mahmood et al., 2014).
        # For stability reasons, we always normalize weights by 1/maxi wi so that they only scale the
        # update downwards (Section 3.4, first paragraph)
        weights = (self.real_size * probs) ** -self.beta

        # As mentioned in Section 3.4, whenever importance sampling is used, all weights w_i were scaled
        # so that max_i w_i = 1. We found that this worked better in practice as it kept all weights
        # within a reasonable range, avoiding the possibility of extremely large updates. (Appendix B.2.1, Proportional prioritization)
        weights = weights / weights.max()

        #batch = (
        #    self.state[sample_idxs].to(device()),
        #    self.action[sample_idxs].to(device()),
        #    self.reward[sample_idxs].to(device()),
        #    self.next_state[sample_idxs].to(device()),
        #    self.done[sample_idxs].to(device())
        #)
        #return batch, weights, tree_idxs
        return [self.memory[i] for i in sample_idxs], weights, tree_idxs

    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)

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
    def __init__(self, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame_duration=1, BATCH_SIZE = 64, memory_capacity=256, GAMMA = 0.7, EPS_START = 1, EPS_END = 0.0, EPS_DECAY = 400, TAU = 0.01, LR = 0.25e-3, recharge_thresh=0.2, max_steps=100, num_episodes=10, train_every=512, local_mininet_simulation=True, server_ip="", server_port=""):
        self._env = WSNEnvironment(max_steps=max_steps, sampling_freq=sampling_freq, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device, num_episodes=num_episodes, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port) 
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq # Number of possible transmission frequencies
        self._n_observations = self._num_sensors*self._num_sensors + 2*self._num_sensors
        self._recharge_thresh = recharge_thresh
        self._num_episodes = num_episodes
        self._train_every = 64 # How many steps between updates of target network 

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

        self._memory = [PrioritizedReplayMemory(memory_capacity) for _ in range(self._num_sensors)]

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
        if len(self._memory[0]) < self._BATCH_SIZE:
            return
        
        total_loss = {r:0 for r in self._reward_types} 
        for agent in range(self._num_sensors):
            # Sample a batch of transitions 
            transitions, weights, tree_idxs = self._memory[agent].sample(self._BATCH_SIZE)
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
                
                if reward_type == 'throughput':
                    td_error = (state_action_values - predicted_q_values).abs().detach().cpu()
                    self._memory[agent].update_priorities(tree_idxs, td_error.numpy()) 

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
            pickle.dump((self._BATCH_SIZE, self._loss), file)
            
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
            print("Q values for action:")
            print(action_values)
            action_probs = F.softmax(action_values, dim=1)
            policy_actions = torch.argmax(action_probs, dim=1)
            policy_actions = torch.round(policy_actions).int().squeeze()

        print("Policy mask: ", policy_mask)

        actions[policy_mask] = policy_actions[policy_mask]
        # action[dead_sensors] = 0 # Dont allow dead sensors to transmit
        # action[awake_sensors & (action==0)] = 1 # Force awake sensors to transmit

        print(f'action: {action}')
        
        return actions
    
    def train(self):
        train_steps = 0
        throughput_reward_log = [-1 for _ in range(10)]
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
                
                print(f'Throughput reward: {throughput_reward[0]}')
                print(f'Similarity reward: {similarity_reward}')
                
                throughput_reward_log.append(throughput_reward[0])
                
                # Move the reward onto the correct device (memory, cpu, or gpu)
                throughput_reward = torch.tensor(throughput_reward, device=device, dtype=torch.float32)
                similarity_reward = torch.tensor(similarity_reward, device=device, dtype=torch.float32)


                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation.clone().detach()

                for i in range(self._num_sensors):
                    # Store the transition in memory
                    self._memory[i].push(self._state, action, next_state, throughput_reward, similarity_reward)

                # Move to the next state
                self._state = next_state
                
                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                train_steps += 1                
                with torch.no_grad():
                    if train_steps >= self._train_every:
                        self._train_steps = 0
                        for agent in range(self._num_sensors):
                            # Update target networks
                            for reward_type in self._reward_types:
                                self._target_net[reward_type][agent].load_state_dict(self._policy_net[reward_type][agent].state_dict())
                            #if self._train_steps[reward_type] >= self._train_every[reward_type]:
                            #self._train_steps[reward_type] = 0
                            # Soft update of the target network's weights
                            #target_net_state_dict = self._target_net[reward_type][agent].state_dict()
                            #policy_net_state_dict = self._policy_net[reward_type][agent].state_dict()
                            #for key in policy_net_state_dict:
                            #    target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                            #self._target_net[reward_type][agent].load_state_dict(target_net_state_dict)

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
    server_ip = "10.192.135.56" # IP of mininet simulation; ignored if local_mininet_simulation = True
    server_port = 5000 # Ignored if local_mininet_simulation = True

    # RL parametrs
    BATCH_SIZE = 64 
    memory_capacity = 1024 
    GAMMA = 0.99
    EPS_DECAY = 3000 
    EPS_START = 0.3
    EPS_END = 0
    max_steps = 9999 
    LR = 0.25e-2
    
    agent = WSN_agent(sensor_ids=sensor_ids, sampling_freq=sampling_freq, transmission_size=transmission_size, observation_time=observation_time, BATCH_SIZE=BATCH_SIZE, memory_capacity=memory_capacity, max_steps=max_steps, LR=LR, EPS_DECAY=EPS_DECAY, EPS_START=EPS_START, EPS_END=EPS_END, GAMMA=GAMMA, local_mininet_simulation=local_mininet_simulation, server_ip=server_ip, server_port=server_port)
    agent.train()
    
