import sys
import socket
import os
import time
import threading
import random
import multiprocessing
import json
import subprocess
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
import time
import math
import pickle
import os
from torch_geometric.utils import from_networkx

from redundancy_graph import compute_similarity_and_redundancy_graph


from configs import Mininet_Simulation_Config
from reward_config import reward_config
from rewards import get_rewards 
import networkx as nx
from networkx.algorithms import approximation as approx
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from itertools import combinations
from ortools.linear_solver import pywraplp
import scipy.interpolate 

from tqdm import tqdm
from mininet.node import Controller
from mininet.node import RemoteController
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from datetime import datetime, timedelta
from collections import defaultdict

class Cluster_Handler():
    def _start_receiver(self):
        info("*** Starting receiver\n")
        print("Starting receiver")
        
        # Activate cluster head listening threads
        receive_threads = []
        receive_thread = threading.Thread(target=self._receive_messages, args=(self._cluster_head,))
        receive_thread.start()
        receive_threads.append(receive_threads)
        return receive_threads

    def _start_senders(self):    
        print("Starting senders")
        info("*** Starting senders\n")
        sender_threads = []
        ch_ip = f'192.168.{self._cluster_idx}.100'
        
        for i, sensor in enumerate(self._sensors):
            # tcpdump_file = f'{self._log_directory}/pcaps/tcpdump_sender_sensor{i}.pcap'
            # sensor.cmd(f'tcpdump -U -i s{i}-wlan0 -w {tcpdump_file} &')
            
            thread = threading.Thread(target=self._send_messages_to_cluster_head, args=(sensor, ch_ip, i))
            thread.start()
            sender_threads.append(thread)

    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_idx (int): index of sensor
    """
    def _send_messages_to_cluster_head(self, sensor, ch_ip, sensor_idx):
        # Get the interpolated temperature data corresponding to this sensor
        sensor_data = self._datasets[sensor_idx] 
        
        # Check if sensor data is valid
        if not sensor_data:
            info(f"Sensor {sensor_idx}: No data available. Skipping send_messages.\n")
            return
        
        # Create a file to store the packets from this sensor received by the cluster head
        sensor.cmd(f'touch {self._log_directory}/ch{self._cluster_idx}_received_data/sensor_{self._config.sensor_ids[sensor_idx]}.txt')
        
        # Track the number of packets sent
        packets_sent = 0

        # Port to use when communicating with the cluser head 
        port = 5001 + sensor_idx 
        
        # TODO: Energy is currenty ignored
        # Initalize the sensors energy and the recharge count 
        energy = self._full_energy
        # self._energy[sensor_idx] = self._full_energy
        charge_count = 0

        # Log the initial transmission rate
        # self.rate_log[sensor_idx].append(self._transmission_rates[sensor_idx])

        # Create filler to pad packets to full size
        filler = 'G' * (self._config.transmission_size)
        
        while True:
            # Recharge sensor if energy is below the recharge threshold
            if False:
            #if self._energy[sensor_idx] < self._recharge_threshold:
                charge_count += 1

                recharge_time = time.time()
                rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
                #info(f'Sensor {sensor_idx}: Energy {self._energy[sensor_idx]} below threshold ({self._recharge_threshold}). Recharging...')
                time.sleep(self._recharge_time)

                # Update the sensors energy and log the new energy level
                self._energy[sensor_idx] = self._full_energy
                sensor.energy = energy

                #info(f'Sensor {sensor_idx}: Energy Recharged to full energy ({self._full_energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.')
                
                #if next_chunk_idx >= len(chunks):
                    #self._transmit_data_status[sensor_idx].set()
                #    break

            # Get the sensors current transmission rate
            transmit_rate = self._transmission_rates[sensor_idx]
            transmit_freq = self._config.transmission_frequencies[transmit_rate]
            
            # Sensor should skip tranmission during the current frame
            if transmit_freq == 0:
                #next_chunk_idx += int(max(self._transmission_frequencies))

                time.sleep(self._config.transmission_frame_duration)
                continue

            # Store the current time
            transmission_start_time = time.perf_counter()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(transmission_start_time))
            ms = int((transmission_start_time - time.perf_counter()) * 1000)
            
            # Get the temp 
            temp = f'{sensor_data((transmission_start_time - self._simulation_start_time + self._offset) % (int(self._min_dataset_t) - 1)):.4f}' 
         
            # Send the recorded temperature with filler as padding
            cmd = f'echo "\n{temp}\n{filler[len(temp) + 6:]}\n" | nice -n 10 nc -v -w0 -u {ch_ip} {port + (self._cluster_idx*(self._num_sensors + 1))} >> {self._log_directory}/error{self._cluster_idx}/nc{sensor_idx} 2>&1 &'

            # cmd = f'echo "\n{temp}\n{filler[len(temp) + 6:]}\n" | nc -v -w0 -u {ch_ip} {port + (self._cluster_idx*(self._num_sensors + 1))} >> {self._log_directory}/error{self._cluster_idx}/nc{sensor_idx} 2>&1 &'
            sensor.cmd(cmd)
            self._chunks_sent[sensor_idx] += 1
            #self._transmissions_log[sensor_idx].append((temp, transmission_start_time - self._simulation_start_time))

            transmission_time = time.perf_counter() - transmission_start_time
            #print(f'Transmission time: {transmission_time}\n')

            sleep_time = 1 / transmit_freq
            #print(f'Sleep time: {sleep_time}\n')

            if(sleep_time - transmission_time > 0):
                time.sleep(sleep_time - transmission_time)

        info(f"Sensor {sensor_idx}: Finished sending messages\n")

    """
    Start background netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _receive_messages(self, node):
        base_output_file = f'{self._log_directory}/ch{self._cluster_idx}_received_data/sensor'

        for i in range(self._num_sensors):
            # Create a file to store data received from sensor i
            output_file = f'{base_output_file}_{self._config.sensor_ids[i]}.txt'
            node.cmd(f'touch {output_file}')

            # Create a listener for sensor i 
            node.cmd(f'stdbuf -o0 -i0 nc -n -vv -ul -p {5001 + i + ((self._num_sensors + 1) * self._cluster_idx)} -k >> {output_file} 2> {self._log_directory}/error{self._cluster_idx}/listen_err &')
            info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")

        # Capture the network by pcap
        # pcap_file = f'{self._log_directory}/pcaps/capture.pcap'
        # node.cmd(f'tcpdump -U -i {node.defaultIntf().name} -n udp portrange {5001}-{5001+self._num_sensors-1} -U -w {pcap_file} &')
        # info(f"Receiver: Started tcpdump capture on ports 5001-{5001+self._num_sensors - 1}\n")

    """
    Read data received by cluster head from sensors

    Args:
        file_path (string): Path to file with received data 
        sensor_id (int): ID of the sensor whose data should be read 

    Returns:
        String List: List of received packets 
    """
    def _read_temperature_data_from_file(self, file_path, sensor_id):
        with open(file_path, 'r') as file:
            data = []
            bytes_received = 0
            for line in file:
                bytes_received += len(line)
                self._process_temperature_data_file_line(line, data)
                            
        throughput = len(data)
        self._total_transmissions_received += len(data) 
        
        return np.array(data), throughput

    """
    Set sensor transmission rates 

    Args:
        new_rates (integer list): List of indicies for new sensor transmission rates 
    """

    def set_rates(new_rates):
        for i in range(len(new_rates)):
            self._transmission_rates[i] = new_rates[i]
    
    """
    Establish connection with RL-Agent 

    The mininet simulaton receives requests from the RL-Agent for observations.  
    """
    def establish_connection_with_rl_agent(self):
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_port = 5000
        listen.bind(("0.0.0.0", listen_port))
        listen.listen(5)
        print("Listening for connection")
        self._rl_agent, _ = listen.accept()
    
    def _process_temperature_data_file_line(self, line, data):
        # Ignore filler lines
        if line[0] == 'G':
            return
        try:
            # Split the line and try to convert the temperature value (9th column, index 8) to float
            #temperature = np.float32(line.strip().split(',')[8])
            temperature = np.float32(line.strip())
            data.append(temperature)
        except (ValueError, IndexError):
            # If conversion fails or the line doesn't have enough columns, skip this line
            pass

    """
    Split dataset into chunks. A chunk corresponds to a transmissions worth of data.
    Note: The chunks returned must be padded before transmission to the correct size.

    Args:
        dataset_dir (string): Path to dataset file
        sensor_id (int): ID of the sensor associcated to the dataset file
        file_lines_per_transmission (int): How many file lines to include per transmission 

    Returns:
        String List: List of chunks 
    """
    def _preprocess_dataset_into_chunks(self, dataset_dir, sensor_id, file_lines_per_chunk):
        # Cache the dataset into memory 
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()

        # Get the number of lines in the file 
        file_size = len(lines)

        # Get the number of chunks the data set can split into
        num_chunks_in_file = file_size // file_lines_per_chunk

        if num_chunks_in_file < self._config.num_transmission_frames:
            print(f'The number of chunks to send or the number of file lines per chunk must be decreased. File {dataset_dir} contains enough data for {num_chunks_in_file} but the number of transmission frames is {self._config.num_transmission_frames}')
            return []

        # Split file into chunks
        chunks = [lines[i:i + file_lines_per_chunk] for i in range(0, len(lines), file_lines_per_chunk)]
        chunks = [''.join(chunk) for chunk in chunks]
        print(f"Sensor {sensor_id} has {len(chunks)} chunks")

        return chunks

    """
    Interpolate dataset using linear splines.
    Note: A second in the spline corresponds to 250 data points in the dataset file.

    Args:
        dataset_dir (string): Path to dataset file
        sensor_id (int): ID of the sensor associcated to the dataset file
        file_lines_per_transmission (int): How many file lines to include per transmission 

    Returns:
        scipy function: function for interpolated data 
    """

    def _interpolate_dataset(self, dataset_dir, maxlen=100000000):
        # Cache the dataset into memory 
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()

            # Skip the first 4 lines
            data = []
            for line in lines:
                if len(data) > maxlen:
                    break
                try:
                    # Split the line and try to convert the temperature value (9th column, index 8) to float
                    temperature = np.float32(line.strip().split(',')[8])
                    data.append(temperature)
                except (ValueError, IndexError):
                    # If conversion fails or the line doesn't have enough columns, skip this line
                    continue

            xs = np.arange(len(data))/120
            interp_func = scipy.interpolate.interp1d(xs, data)
            self._min_dataset_t = min(len(data)/120, self._min_dataset_t)
            return interp_func

    """
    Get the maximal independent set that maximizes the minimum throughput of a sensor in the set.

    Args:
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors. 

    Returns:
        list: list of sensors in the maximum independent set 
    """

    def _get_max_ind_set(self, redundancy_graph):
        # TODO: Switch to polynomial time approximation? 
        LP = LpProblem("Weighted_Max_Independent_Set", LpMaximize)

        # binary variabls for each node: 1 if selected, 0 otherwise
        x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in redundancy_graph.nodes}

        # Objective: maximize total throughput 
        LP += lpSum(redundancy_graph.nodes[v]["throughput"] * x[v] for v in redundancy_graph.nodes)

        # Constraint: for each edge, at most one endpoint can be in the independent st
        for u,v in redundancy_graph.edges:
            LP += x[u] + x[v] <= 1

        LP.solve(PULP_CBC_CMD(msg=0))

        independent_set = [v for v in redundancy_graph.nodes if x[v].varValue == 1]
        return independent_set

        # Can also maximize the min throughput for a sensor in the ind set
        # Objective: maximize min throughput for sensor in ind set
        #LP += z # mimimum selected sensor throughput

        #M = max(nx.get_node_attributes(G, "throughput").values()) + 1
        #for v in redudancy_graph.nodes:
        #    w = redudancy_graph.nodes[v]['throughput']
        #    LP += z <= w * x[v] + (1 - x[v]) * M

        # Constraint: the independent set is maximal
        # for v in redudancy_graph.nodes:
        #    LP += x[v] + lpSum(x[u] for u in redudancy_graph.neighbors(v)) >= 1

    """
    Calculate throughput and similarity rewards.
    Note: All sensors receive the same throughput reward. 

    Args:
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors.
        sensor_effective_throughputs: a list of the effective throughputs for each sensor

    Returns:
        reward (dict): dict with keys as reward type and values as a list of rewards
        max_ind_set: the independent set of highest throughput in the redudancy graph. 
    """

    def _calculate_throughput_reward(self, redundancy_graph, sensor_effective_throughputs):
        ind_set_with_max_throughput = self._get_max_ind_set(redudancy_graph)
        ind_set_total_throughput = np.sum([redudancy_graph.nodes[v]["throughput"] for v in ind_set_with_max_throughput])

        max_ind_set = approx.maximum_independent_set(redudancy_graph) 
        maxF = np.max(self._transmission_rates)
        total_throughput_bound = self._num_sensors * maxF / 2
        throughput_reward = [bounded_log(ind_set_total_throughput / total_throughput_bound) for i in range(self._num_sensors)]

        return max_ind_set, throughput_reward

    def _get_sensor_temperature_data(self, sensor_idx):
        # Name of file where transmisions received by the cluster head from sensor i is stored
        file_name = f'sensor_{self._config.sensor_ids[sensor_idx]}.txt'

        # Create a copy of the original file
        subprocess.run(["cp", f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', f'{self._log_directory}/ch{self._cluster_idx}_received_data/.{file_name}'])
        file_path = os.path.join(self._log_directory, f'ch{self._cluster_idx}_received_data/.{file_name}')
        curr_time = time.perf_counter()
        true_observation_period = (curr_time - self._prev_obs_end_time[sensor_idx])
        self._prev_obs_end_time[sensor_idx] = curr_time

        # Clear the file for future transmissions
        # TODO: Lock before clearing?
        with open(f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}', 'r+') as file:
            file.truncate(0)
       
        # Read the temperature data for sensor i from the copied file
        data, throughput = self._read_temperature_data_from_file(file_path, sensor_idx)
        
        throughput = throughput / true_observation_period
        return data, throughput

    """
    Get the temperature data for all sensors for the previous observation period.  

    Returns:
        temperature_data (dict): a dict with sensors as keys and values as a list of the temprature data transmitted by a sensor 
    """
    def _get_temperature_data(self):
        # Dict with keys as awake sensor ids and values as the data received by the cluster head from a sensor
        temperature_data = {}
        throughputs = []

        for i in range(self._num_sensors):
            data, throughput = self._get_sensor_temperature_data(i)
            throughputs.append(throughput)
            if len(data) > 0:
                temperature_data[i] = data
        
        return temperature_data, throughputs

    def _update_rates(self, rates):
        for i in range(self._num_sensors):
            self._transmission_rates[i] = int(rates[i])

            # Name of file where transmisions received by the cluster head from sensor i is stored
            # file_name = f'sensor_{self._config.sensor_ids[i]}.txt'
            # file_path = f'{self._log_directory}/ch{self._cluster_idx}_received_data/{file_name}'
            
            # self._total_transmissions_received += os.path.getsize(file_path) / self._config.transmission_size

            # Clear the file for future transmissions
            # TODO: Lock before clearing?
            # with open(file_path, 'r+') as file:
            #    file.truncate(0)
    
    def _update_logs(self, throughputs, similarity, rewards, reward_output):
        self.similarity_log.append(similarity)
        # Log sensor data and rewards 
        self.chunks_sent_log.append(list(self._chunks_sent))
        for i in range(self._num_sensors):
            self.throughput_log[i].append(throughputs[i])
            # self.energy_log[i].append(self._energy[i])
            self.rate_log[i].append(self._transmission_rates[i])
            # self._chunks_sent[i] = 0
        if reward_output: 
            max_ind_set = reward_output["throughput"][0]
            print(f"Cluster {self._cluster_idx} Max ind set", [self._config.sensor_ids[v] for v in max_ind_set])
        else:
            max_ind_set = []
        self.max_ind_set_log.append(max_ind_set)
        self.reward_log["similarity"].append(rewards["similarity"])
        self.reward_log["throughput"].append(rewards["throughput"])
        self.total_transmissions_received_log.append(self._total_transmissions_received)
        
        # Pickle logs for plotting
        self._log_step_count += 1
        if self._log_step_count > self._log_every:
            self._log_step_count = 0
            with open(f'figure_data{self._cluster_idx}.pkl', 'wb') as file:
                pickle.dump((self._config.sensor_ids, list(self._transmission_rates), self.rate_log, self.throughput_log, self.reward_log, self.similarity_reward_log, self.throughput_reward_log, self.max_ind_set_log, self.chunks_sent_log, self.total_transmissions_received_log, self._transmissions_log), file)
    
    def _similarity_changed(self, new_similarity):
        if self._prev_similarity is None or self._t_since_last_sim_change == 0:
            self._prev_similarity = new_similarity
            self._t_since_last_sim_change += 1
            return False 

        for i in range(self._num_sensors - 1):
            for j in range(i, self._num_sensors):
                if(new_similarity[i][j] != self._prev_similarity[i][j]):
                    self._prev_similarity = new_similarity
                    self._t_since_last_sim_change = 0 

                    return True
        
        self._t_since_last_sim_change += 1
        return False 


    """
    Get an observation of the enviornment.
    
    Args:
        rates (int list): Transmission Frequency index to be used for each sensor
        heuristic (bool): If heursitic is True, rates are discared and a heursitic policy is used to assign transmission rates
        replay (int): If replay is greater than -1, the similarity matrix is retrived from step replay to calculate rewards  

    Returns:
        tuple: (rates_id, similarity, energy, throughputs, reward)
            similarity - num_sensor x num_sensor matrix where entry i,j denotes wheter the data generated by sensor i was similar to the data generated by sensor j
            energy - percent of energy remaining for each sensor
            throughputs - vector of average throughput for each sensor
            reward - reward according to reward function
    """

    def get_observation(self, action):
        rates = action[:self._num_sensors]
        replay = action[-1]
        
        self._update_rates(rates)

        # Wait for observation time
        time.sleep(self._config.observation_time)
        
        # Temperature data is a dict with keys as awake sensor ids and values as the data received by the cluster head from a sensor
        temperature_data, throughputs = self._get_temperature_data()
        #self._prev_obs_end_time.value = time.perf_counter()
                
        # Get similarity matrix and redudancy graph
        transmit_freqs = [self._config.transmission_frequencies[int(r)] for r in rates]

        similarity, redundancy_graph = compute_similarity_and_redundancy_graph(self._config, transmit_freqs, temperature_data, throughputs, replay)
        total_throughput = np.sum(throughputs)

        terminated = self._similarity_changed(similarity)

        print(f'Cluster {self._cluster_idx} total throughput over observation: {total_throughput} transmissions/s')
        
        if total_throughput == 0:
            rewards = {
                    "throughput": -10,
                    "similarity": [-5]*self._num_sensors
                    }

            self._update_logs(throughputs, similarity, rewards, None)
            redundancy_graph = from_networkx(redundancy_graph)

            return (terminated, redundancy_graph, rewards)

        rewards, reward_output = get_rewards(reward_config, self._config, redundancy_graph)
        self._update_logs(throughputs, similarity, rewards, reward_output)

        redundancy_graph = from_networkx(redundancy_graph)
        return (terminated, redundancy_graph, rewards)


    def _send_observation_to_rl_agent(self, rates):
        obs = pickle.dumps(self.get_observation(rates))
        obs_length = len(obs)

        # Send the size of the observation in byts to the server
        self._rl_agent.sendall(struct.pack('!I', obs_length))

        # Send the observation to the server
        self._rl_agent.sendall(obs)

    def receive_rates_from_rl_agent(self):
        while(True):
            packed_data = self._rl_agent.recv(4 * (self._num_sensors))

            # Connection terminated by server
            if len(packed_data) == 0:
                break;

            unpacked_data = struct.unpack('!' + 'i'*(self._num_sensors), packed_data)

            print(f"\n\nReceived request from server for rates {unpacked_data}")

            rates = unpacked_data
            for sensor_idx in range(self._num_sensors):
                self._transmission_rates[sensor_idx] = rates[sensor_idx]

            self._send_observation_to_rl_agent(rates)
    
    def _save_sensor_temps(self):
        while True:
            with open(f'sensortemps{self._cluster_idx}.pkl', 'wb') as file:
                pickle.dump((self._transmissions_log), file)
            time.sleep(20)

    def __init__(self, cluster_head, sensors, cluster_idx, simulation_config, log_directory='data/log', dataset_directory='data/towerdataset'):
        self._config = simulation_config
        self._log_every = 100
        self._log_step_count = 0
        self._total_transmissions_received = 0
        self.total_transmissions_received_log = []
        self._num_sensors = len(self._config.sensor_ids)
        self._transmissions_log = [[] for _ in range(self._num_sensors)]
        self._cluster_idx = cluster_idx
        self._throughputs = [0 for i in range(self._num_sensors)]
        self._total_throughput = 0
        self._max_total_throughput = 0
        self._prev_obs_end_time = multiprocessing.Array('d', [time.perf_counter()] * self._num_sensors)
        self._offset = random.randint(0, 100000)
        self._min_dataset_t = 99999999
        self._prev_similarity = None
        self._t_since_last_sim_change = 999999999

        self._cluster_head = cluster_head
        self._sensors = sensors

        # Directories
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
       
        # Rate configuration
        self._transmission_rates = multiprocessing.Array('i', [0] * self._num_sensors)
        #self._transmission_frequencies = np.array([20, 40, 60, 80]) # Possible number of times a sensor can transmit per frame

        # Energy configuration
        self._full_energy = 100
        self._recharge_time = 10 * self._config.transmission_frame_duration
        self._recharge_threshold = 20
        self._energy = [self._full_energy for _ in range(self._num_sensors)]

        # Data logs 
        self.rate_log = [[] for _ in range(self._num_sensors)]
        self.throughput_log = [[] for _ in range(self._num_sensors)]
        self.reward_log = {"similarity":[], "throughput":[]}
        self.similarity_reward_log = []
        self.similarity_log = []
        self.throughput_reward_log = []
        self.max_ind_set_log = []
        self.chunks_sent_log = []
        self._chunks_sent = multiprocessing.Array('i', [0] * self._num_sensors)

        #self.chunks_sent = [0 for _ in range(self._num_sensors)] # List of the number of transmissions sent by each sensor

        # Delete the log folder if already it exists
        # subprocess.run(["rm", "-rf", log_directory])

        # Create log directories 
        # os.makedirs(log_directory)
        os.makedirs(log_directory + f'/ch{self._cluster_idx}_received_data')
        os.makedirs(log_directory + f'/error{self._cluster_idx}')
        os.makedirs(log_directory + f'/pcaps{self._cluster_idx}')

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        # Initalize the dataset for each sensor 
        self._datasets = {}
        for i in range(self._num_sensors):
            tower_number = self._config.sensor_ids[i]
            file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
            if os.path.exists(file_path):
                #self._datasets[i] = self._preprocess_dataset_into_chunks(file_path, i, file_lines_per_chunk)
                self._datasets[i] = self._interpolate_dataset(file_path)
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []

        self._simulation_start_time = time.perf_counter()
        # self._create_topology()

    
    """
    Kill netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _stop_receivers(self, node):
        # Kill the receiver
        node.cmd('pkill -f "nc -ul"')
        # Kill the tcpdump capture
        node.cmd('pkill tcpdump')
        info("Stopped all nc receivers and tcpdump\n")

    """
    Begin message transmission from sensors to cluster head and reap all threads after transmission concludes.   
    """
    def start(self):
        print("STARTING")
        
        info("*** Setting up communication flow\n")
        try:
            receive_thread = self._start_receiver()

            # Give listening thread time to start
            time.sleep(10) 
            
            self._start_time = time.perf_counter()
            # Start senders
            self._start_senders()

            #plot_thread = threading.Thread(target=self._save_sensor_temps, args=())
            #plot_thread.start()
            #plot_thread.join()

            # Wait for senders to finish
            print("Waiting for senders to finish")
            for thread in sender_threads:
                thread.join()

            # Pickle simulation results
            #print(f'Simulation complete; saving data\n')
            # with open(f'figure_data{self._cluster_idx}.pkl', 'wb') as file:
                #pickle.dump((self._parameters.sensor_ids, self._rate_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.clique_reward_log, self.throughput_reward_log, self.clique_log, self.chunks_sent), file)
            #print(f'Pickle dump succesfuly made\n')

            self._stop_receivers(cluster_head)
            receive_thread.join()

            for sensor in sensors:
                sensor.cmd('pkill tcpdump')
            cluster_head.cmd('pkill nc')

            self._plot_energy()
            self._plot_throughput()
            self._plot_rates(3)
            self._create_plot(self, self.rewards, 'rewards', self.timestaps, 'Time (seconds)', 'Reward', 'Reward over Time')
            self._create_plot(self, self.similarity, 'Similarity Penalty', self.timestaps, 'Time (seconds)', 'Penalty', 'Similarity Penalty over Time') 
            
        except Exception as e:
            info(f"*** Error occurred during communication: {str(e)}\n")
            
        info("*** Running CLI\n")
        # CLI(self._net)

        info("*** Stopping network\n")
        # self._net.stop()

        # self._server.close()

    def _parse_tcpdump_output(self):
        print("Starting to parse tcpdump output...")
        start_time = time.time()
        udp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d+)\sIP\s(\d+\.\d+\.\d+\.\d+)\.(\d+)\s>\s(\d+\.\d+\.\d+\.\d+)\.(\d+):\sUDP,\slength\s(\d+)')
        sensor_packets = defaultdict(list)

        # Create a copy of the original file
        result = subprocess.run(["cp", f'{self._log_directory}/pcaps/capture.pcap', f'{self._log_directory}/pcaps/.capture.pcap'])

        with open(f'{self._log_directory}/pcaps/capture.pcap', 'r+') as file:
            file.truncate(0)

        result = subprocess.run(["sudo", "bash", f"{os.getcwd()}/extract_pcap.sh"])

        if result.returncode == 0:
            print("Output:", result.stdout)
        else:
            print("Error:", result.stderr)

        with open(f'{self._log_directory}/pcaps/extracted_data/tcpdump_output_capture.txt', 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if i % 10000 == 0:
                    match = udp_pattern.search(line)
                    
                    if match:
                        time_str = match.group(1)
                        src_ip = match.group(2)
                        packet_size = int(match.group(6))
                        timestamp = datetime.strptime(time_str, '%H:%M:%S.%f')
                        sensor_packets[src_ip].append((timestamp, packet_size))

        print(f"Parsing completed in {time.time() - start_time:.2f} seconds")
        return sensor_packets

    def _aggregate_throughput(self, sensor_ip, packets, interval=1):
        print("Aggregating throughput...")
        start_time = time.time()
        if not packets:
            return []

        packets.sort(key=lambda x: x[0])
        start_time_packet = packets[0][0]
        end_time_packet = packets[-1][0]
        current_time = start_time_packet
        #throughput_data = []

        total_intervals = int((end_time_packet - start_time_packet).total_seconds() / interval)
        processed_intervals = 0

        while current_time <= end_time_packet:
            next_time = current_time + timedelta(seconds=interval)
            interval_packets = [p for p in packets if current_time <= p[0] < next_time]
            total_data = sum(p[1] for p in interval_packets) * 8  # Convert to bits
            throughput = total_data / interval  # bits per second
            self._throughput_data[sensor_ip].append((current_time, throughput / 1e6))  # Convert to Mbps
            current_time = next_time

            processed_intervals += 1
            if processed_intervals % 100 == 0:
                print(f"Aggregation progress: {processed_intervals}/{total_intervals} intervals ({processed_intervals/total_intervals*100:.2f}%)")

        print(f"Aggregation completed in {time.time() - start_time:.2f} seconds")

if __name__== '__main__':
    sensor_ids = range(5,15)
    observation_time = 1
    transmission_size = 2*1024
    transmission_frame_duration = 1
    file_lines_per_chunk = 1
    num_transmission_frames = 3000
    sim_config = Mininet_Simulation_Config(sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, num_transmission_frames=num_transmission_frames)
    

    cluster = Cluster(0, sim_config, log_directory=f'data/log')
    # cluster.establish_connection_with_rl_agent()
    print("Starting cluster")
    cluster_thread = threading.Thread(target=cluster.start, args=())
    cluster_thread.start()
    

    # receive_rates_thread = threading.Thread(target=cluster.receive_rates_from_rl_agent, args=())
    # receive_rates_thread.start()

    cluster_thread.join()
    #receive_rates_thread.join()

