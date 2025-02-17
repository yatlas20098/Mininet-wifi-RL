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

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler

from RL_agent import WSNEnvironment

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
        w = 128 

        self.layer1 = nn.Linear(num_sensors*num_sensors + 2*num_sensors, w)
        self.layer2 = nn.Linear(w,w)
        self.layer3 = nn.Linear(w,w)
        self.layer4 = nn.Linear(w, num_sensors * sampling_freq)

    # Called with either one element to determine next action, or a batchduring optimization.
    # Returns tensor([[left0exp, right0exp]...])
    def forward(self, x, first_step=False):
        #x = x.view(-1, 2 * self._num_sensors)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        #x = self.relu1(self.batch_norm1(self.layer1(x)))
        #x = self.relu2(self.batch_norm2(self.layer2(x)))

        #x = self.layer1(x)
        #if not first_step:
        #    x = self.batch_norm1(x)
        #x = self.relu1(x)

        #x = self.layer2(x)
        #if not first_step:
        #    x = self.batch_norm1(x)
        #x = self.relu2(x)

        #return self.layer3(x).view(-1, self._num_sensors, self._sampling_freq)
        return x

class WSN_agent:
    def __init__(self, num_clusters, sensor_ids, sampling_freq = 3, observation_time=10, transmission_size=4*1024, file_lines_per_chunk=5, transmission_frame=1, BATCH_SIZE = 64, GAMMA = 0.99, EPS_START = 0.9, EPS_END = 0.3, EPS_DECAY = 1500, TAU = 0.005, LR = 0.001, recharge_thresh=0.2):
        self._env = [WSNEnvironment(max_steps=100, sampling_freq=sampling_freq, cluster_id=i, sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame=transmission_frame, file_lines_per_chunk=file_lines_per_chunk, recharge_thresh=recharge_thresh, device=device) for i in range(num_clusters)]
        self._num_sensors = len(sensor_ids) 
        self._sampling_freq = sampling_freq
        self._state, self._info = [list(t) for t in zip(*[env.reset() for env in self._env])]
        self._n_observations = self._num_sensors
        self._recharge_thresh = 0.2
        self._update_lock = threading.Lock()
        self._num_clusters = num_clusters

        self._transmission_frame = transmission_frame
        self._sensor_ids = sensor_ids
        self._observation_time = observation_time
        self._transmission_size = transmission_size
        self._file_lines_per_chunk = file_lines_per_chunk
        self._recharge_thresh = recharge_thresh

        num_actions = self._env[0].action_space.shape[0] # All envs have the same action and observation space
        self._policy_net = DQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
        self._target_net = DQN(sampling_freq, self._n_observations, num_actions, self._num_sensors).to(device)
        self._target_net.load_state_dict(self._policy_net.state_dict())

        self._optimizer = optim.AdamW(self._policy_net.parameters(), lr=LR, amsgrad=True)
        self._memory = ReplayMemory(3000)

        self._steps_done = [0 for i in range(num_clusters)]
        self._episode_durations = []
        self._energy_consumption = []
        self._rewards = [[] for _ in range(num_clusters)]

        self._BATCH_SIZE = BATCH_SIZE 
        self._GAMMA = GAMMA 
        self._EPS_START = EPS_START
        self._EPS_END = EPS_END 
        self._EPS_DECAY = EPS_DECAY
        self._TAU = TAU

        time.sleep(10)

    def select_action(self, cluster_id):
        sample = random.random()
        eps_threshold = self._EPS_END + (self._EPS_START - self._EPS_END) * math.exp(-1. * self._steps_done[cluster_id] / self._EPS_DECAY)
        self._steps_done[cluster_id] += 1
        action = None

        energy = self._state[cluster_id][:, self._num_sensors*self._num_sensors: self._num_sensors*self._num_sensors + self._num_sensors]
        energy = energy.squeeze()
        dead_sensors = (energy <= self._recharge_thresh)
        awake_sensors = (energy > self._recharge_thresh)

        if sample > eps_threshold:
            print('Getting best action')
            with torch.no_grad():
                action_values = self._policy_net(self._state[cluster_id])
                action_values= action_values.view(1, self._num_sensors, 4)
                action_probs = F.softmax(action_values, dim=-1)
                action = torch.argmax(action_probs, dim=-1)
                action = action.squeeze(0)
        else:
            print('Getting random action')
            action = torch.randint(0, self._sampling_freq, (self._num_sensors,), dtype=torch.long, device=device) 
       
        print(f'action: {action}')
        action[dead_sensors] = 0 # Dead sensors cannot transmit
        action[awake_sensors & (action==0)] = 1 # Awake sensors must transmit

        return action

    def select_best_action(self):
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

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self._episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def plot_rewards(self, show_result=False):
        plt.figure(2)
        durations_t = torch.tensor(self._rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def plot_consumption(self, show_result=False):
        plt.figure(3)
        durations_t = torch.tensor(self._energy_consumption, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Step')
        plt.ylabel('Energy')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


    def _optimize_model(self, first_step=False):
        if len(self._memory) < self._BATCH_SIZE:
            print(f'Mem len: {len(self._memory)}')
            return

        transitions = self._memory.sample(self._BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).long()
        action_batch = action_batch.view(self._BATCH_SIZE, self._num_sensors)
        reward_batch = torch.cat(batch.reward).view(-1, 1)

        state_action_values = self._policy_net(state_batch, first_step).gather(2, action_batch.unsqueeze(-1))
        state_action_values = state_action_values.squeeze(-1)

        next_state_values = torch.zeros(self._BATCH_SIZE, self._num_sensors, device=device)
        with torch.no_grad():
            max_values = self._target_net(non_final_next_states, first_step).max(2)[0]
            next_state_values[non_final_mask] = max_values

        reward_batch = reward_batch.expand(-1, self._num_sensors)
        expected_state_action_values = (next_state_values * self._GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        #loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = criterion(state_action_values, expected_state_action_values)

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)
        self._optimizer.step()
        #self._scheduler.step()

    def _train(self, cluster_id):
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            print('Torch Cuda is available')
            num_episodes = 3 
        else:
            print('Torch Cuda is not available')
            num_episodes = 3 

        self._train_every = 10
        self._train_steps = 0

        for i_episode in range(num_episodes):
            # Initialize the environment and get its state
            print(self._env[cluster_id].reset())
            self._state[cluster_id], self._info[cluster_id] = self._env[cluster_id].reset()
            self._state[cluster_id] = torch.tensor(self._state[cluster_id], dtype=torch.float32, device=device).unsqueeze(0)
            print(f'Cluster {cluster_id} State = {self._state[cluster_id]}')
            first_step = True

            for t in count():
                print(f'\n\nEpisode {i_episode}, Step: {t}')
                action = torch.Tensor.cpu(self.select_action(cluster_id))

                print('Taking next step')
                observation, reward, terminated, truncated, _ = self._env[cluster_id].step(self._steps_done[cluster_id], action.numpy())
                self._rewards[cluster_id].append(reward)
                print(f'Reward: {reward}')
                
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation.unsqueeze(0).clone().detach()

                with self._update_lock:
                    # Store the transition in memory
                    self._memory.push(self._state[cluster_id], action, next_state, reward)

                    # Move to the next state
                    self._state[cluster_id] = next_state
                    
                    if self._train_steps >= self._train_every:
                    # Perform one step of the optimization (on the policy network)
                        self._optimize_model(first_step)
                        self._train_steps = 0
                        first_step = False

                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self._target_net.state_dict()
                    policy_net_state_dict = self._policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                    self._target_net.load_state_dict(target_net_state_dict)

                    if done:
                        self._episode_durations.append(t + 1)
                        self.plot_durations()
                        break
                    print('Step done')

                    self._train_steps += 1

    def start(self):
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            print('Torch Cuda is available')
            num_episodes = 15 
        else:
            print('Torch Cuda is not available')
            num_episodes = 15 

        self._train_every = 10
        self._train_steps = 0

        training_threads = []
        for i in range(self._num_clusters):
            train_thread = threading.Thread(target=(self._train), args=(i,))
            train_thread.start()
            training_threads.append(train_thread)

        for thread in training_threads:
            thread.join()

        print('Complete; validating')
        validate_ep = 0
        test_env = WSNEnvironment(max_steps=100, sampling_freq=self._sampling_freq, cluster_id=2, sensor_ids=self._sensor_ids, observation_time=self._observation_time, transmission_size=self._transmission_size, transmission_frame=self._transmission_frame, file_lines_per_chunk=self._file_lines_per_chunk, recharge_thresh=self._recharge_thresh, device=device)
        self._steps_done[0] = 0

        while True:
            print(f'Episode: {validate_ep}') 
            # Initialize the environment and get its state
            self._state[0], self._info[0] = test_env.reset()
            self._state[0] = torch.tensor(self._state, dtype=torch.float32, device=device).unsqueeze(0)

            for t in count():
                print(f'Step: {t}')
                action = self.select_best_action()

                observation, reward, terminated, truncated, _ = test_env.step(self._steps_done[0], action.numpy())
                self._rewards[0].append(reward)
                
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                self._memory.push(self._state[0], action, next_state, reward)

                # Move to the next state
                self._state[0] = next_state

                # Perform one step of the optimization (on the policy network)
                self._optimize_model()

                print("Updating target network's weights")
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self._target_net.state_dict()
                policy_net_state_dict = self._policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self._TAU + target_net_state_dict[key]*(1-self._TAU)
                self._target_net.load_state_dict(target_net_state_dict)

                if done:
                    self._episode_durations.append(t + 1)
                    self.plot_durations()
                    break
            validate_ep += 1

        self.plot_durations(show_result=True)
        plt.ioff()
        plt.savefig('test.png')
        plt.show()
        
        self.plot_rewards(show_result=True)
        plt.ioff()
        plt.savefig('test1.png')
        plt.show()

        self.plot_consumption(show_result=True)
        plt.ioff()
        plt.savefig('test2.png')
        plt.show()

if __name__ == '__main__':
    sensor_ids = range(5,15)
    agent = WSN_agent(num_clusters=1, sensor_ids=sensor_ids, sampling_freq = 4, transmission_size=int(1024*3), transmission_frame=1/16, file_lines_per_chunk=1, observation_time=10)
    agent.start()
