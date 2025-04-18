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
import networkx as nx
#from networkx.algorithms.clique import find_cliques as maximal_cliques, approximation as approx
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from itertools import combinations
from ortools.linear_solver import pywraplp
import scipy.interpolate 

from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from datetime import datetime, timedelta
from collections import defaultdict

class sensor_cluster():
    """
    Set sensor transmission rates 

    Args:
        new_rates (integer list): List of indicies for new sensor transmission rates 
    """

    def set_rates(new_rates):
        for i in range(len(new_rates)):
            self.transmission_freq_idxs[i] = new_rates[i]
    """
    Establish connection with RL-Agent on HPC 

    The mininet simulaton is treated as the server and receives requests from the RL-Agent for observations.  
    """
    def establish_connection_with_rl_agent(self):
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_port = 5000
        listen.bind(("0.0.0.0", listen_port))
        listen.listen(5)
        print("Listening for connection")
        self._rl_agent, _ = listen.accept()

    """
    Read data received by cluster head from sensors

    Args:
        file_path (string): Path to file with received data 
        sensor_id (int): ID of the sensor whose data should be read 

    Returns:
        String List: List of received packets 
    """
    def _read_temperature_data(self, file_path, sensor_id):
        with open(file_path, 'r') as file:
            data = []
            bytes_received = 0
            for line in file:
                bytes_received += len(line)
                # Ignore filler lines
                if line[0] == 'G':
                    continue
                try:
                    # Split the line and try to convert the temperature value (9th column, index 8) to float
                    #temperature = np.float32(line.strip().split(',')[8])
                    temperature = np.float32(line.strip())
                    data.append(temperature)
                except (ValueError, IndexError):
                    # If conversion fails or the line doesn't have enough columns, skip this line
                    continue
            
            # Update throughputs
            self._previous_throughputs[sensor_id] = self._throughputs[sensor_id]
            self._throughputs[sensor_id] = bytes_received // self._transmission_size # Measured by number of succesful transmissions
        return np.array(data)

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

        if num_chunks_in_file < self._num_transmission_frames:
            print(f'The number of chunks to send or the number of file lines per chunk must be decreased. File {dataset_dir} contains enough data for {num_chunks_in_file} but the number of transmission frames is {self._num_transmission_frames}')
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

    def _interpolate_dataset(self, dataset_dir, sensor_id, maxlen=100000000):
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

            xs = np.arange(len(data)) / 250 
            interp_func = scipy.interpolate.interp1d(xs, data)
            return interp_func

    """
    Get the maximium independent set of highest throughput in the redudancy graph.

    Args:
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors. 

    Returns:
        list: list of sensors in the maximum independent set 
    """

    def _get_max_ind_set(self, redudancy_graph):
        LP = LpProblem("Weighted_Max_Independent_Set", LpMaximize)

        # binary variabls for each node: 1 if selected, 0 otherwise
        x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in redudancy_graph.nodes}

        # Objective: maximize total throughput 
        LP += lpSum(redudancy_graph.nodes[v]["throughput"] * x[v] for v in redudancy_graph.nodes)

        # Constraint: for each edge, at most one endpoint can be in the independent st
        for u,v in redudancy_graph.edges:
            LP += x[u] + x[v] <= 1

        LP.solve(PULP_CBC_CMD(msg=0))

        independent_set = [v for v in redudancy_graph.nodes if x[v].varValue == 1]
        return independent_set
   
    """
    Calculate throughput and similarity rewards.
    Note: All sensors receive the same throughput reward. 

    Args:
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors.
        sensor_effective_throughputs: a list of the effective throughputs for each sensor

    Returns:
        throughput_reward: list of throughput reward for each sensor
        similarity_reward: list of similarity reward for each sensor
        max_ind_set: the maximum indepdent set of highest throughput in the redudancy graph. 
    """
    def _calculate_rewards(self, redudancy_graph, sensor_effective_throughputs):
        bounded_log = lambda x: np.log2(min(1, max(0.1, x))) 
        
        # Get similarity reward
        max_throughput = max(self._throughputs)
        similarity_reward = [bounded_log(sensor_effective_throughputs[i] / max_throughput) for i in range(self._num_sensors)]

        # Get throughput reward 
        max_ind_set = self._get_max_ind_set(redudancy_graph)
        ind_set_total_throughput = np.sum([redudancy_graph.nodes[v]["throughput"] for v in max_ind_set])

        maxF = np.max(self._transmission_frequencies)
        max_total_throughput = maxF * len(max_ind_set)
        throughput_reward = [bounded_log(ind_set_total_throughput / max_total_throughput) for i in range(self._num_sensors)]

        print(f'Max Indepdenent Set: {max_ind_set}')
        return throughput_reward, similarity_reward, max_ind_set

    """
    Compute the similarity matrix and create a redudancy graph

    Args:
        temperature_data (list): list of temperatures transmitted by each sensor
        
    Returns:
        similarity (numpy 2d list): matrix with i,j=1 if sensor i and j transmitted similar data and 0 otherwise 
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors
        sensor_effective_throughputs: a list of the effective throughputs for each sensor (i.e. fixing a sensor s, the maximium throughput of a sensor - possibily s itself - transmitting similar data to s)
    """

    def _compute_similarity_and_redudancy_graph(self, temperature_data):
        # Initalize similarity matrix
        similarity = np.zeros((self._num_sensors, self._num_sensors))

        # Get list of sensors that had at least one succesfull transmission 
        awake_sensors = list(temperature_data.keys())

        sensor_effective_throughputs = self._throughputs.copy()

        # Create a graph with veritices representing sensors and edges denoting similarity
        redudancy_graph = nx.Graph()
        redudancy_graph.add_nodes_from([(i, {"throughput": self._throughputs[i]}) for i in range(self._num_sensors)])

        for i in range(len(awake_sensors)):
            for j in range(i + 1, len(awake_sensors)):
                # TODO: Currently only using first data point for similiarity 
                similarity[awake_sensors[i], awake_sensors[j]] = int(temperature_data[awake_sensors[i]][0] - temperature_data[awake_sensors[j]][0] <= self._similarity_threshold)

                # Add edge from vertex i to vertex j if sensors i and j are similar
                if similarity[awake_sensors[i],awake_sensors[j]] == 1:
                    redudancy_graph.add_edge(awake_sensors[i],awake_sensors[j])
                    max_throughput = max(sensor_effective_throughputs[i], sensor_effective_throughputs[j])
                    sensor_effective_throughputs[i] = sensor_effective_throughputs[j] = max_throughput

        return similarity, redudancy_graph, sensor_effective_throughputs

    """
    Get the temperature data for all sensors for the previous observation period.  

    Returns:
        temperature_data (dict): a dict with sensors as keys and values as a list of the temprature data transmitted by a sensor 
    """


    def _get_temperature_data(self):
        # Dict with keys as awake sensor ids and values as the data received by the cluster head from a sensor
        temperature_data = {}

        for i in range(self._num_sensors):
            # Name of file where transmisions received by the cluster head from sensor i is stored
            file_name = f'sensor_{self._sensor_ids[i]}.txt'

            # Create a copy of the original file
            subprocess.run(["cp", f'{self._log_directory}/ch_received_data/{file_name}', f'{self._log_directory}/ch_received_data/.{file_name}'])
            file_path = os.path.join(self._log_directory, f'ch_received_data/.{file_name}')

            # Clear the file for future transmissions
            # TODO: Lock before clearing?
            with open(f'{self._log_directory}/ch_received_data/{file_name}', 'r+') as file:
                file.truncate(0)
           
            # Read the temperature data for sensor i from the copied file
            sensor_data = self._read_temperature_data(file_path, i)
            if len(sensor_data) > 0:
                temperature_data[i] = sensor_data
                data_arrived = True

        return temperature_data

    """
    Get an observation of the enviornment.

    Returns:
        tuple: (rates_id, similarity, energy, throughputs, reward)
            similarity - num_sensor x num_sensor matrix where entry i,j denotes wheter the data generated by sensor i was similar to the data generated by sensor j
            energy - percent of energy remaining for each sensor
            throughputs - vector of average throughput for each sensor
            reward - reward according to reward function
    """
    def get_observation(self, rates):
        # Clear transmissions
        for i in range(self._num_sensors):
            self.transmission_freq_idxs[i] = rates[i] 

            # Name of file where transmisions received by the cluster head from sensor i is stored
            file_name = f'sensor_{self._sensor_ids[i]}.txt'

            # Clear the file for future transmissions
            # TODO: Lock before clearing?
            with open(f'{self._log_directory}/ch_received_data/{file_name}', 'r+') as file:
                file.truncate(0)

        observation_start_time = time.time()
        print('\n\nGetting observation')

        time.sleep(self._observation_time)
        true_observation_period_time = (time.time() - observation_start_time) / self._observation_time
        self._throughput = [t / true_observation_period_time for t in self._throughputs]

        # Dict with keys as awake sensor ids and values as the data received by the cluster head from a sensor
        temperature_data = self._get_temperature_data()

        # Get similarity matrix and redudancy graph
        similarity, redudancy_graph, sensor_effective_throughputs = self._compute_similarity_and_redudancy_graph(temperature_data)
        total_throughput = np.sum(self._throughputs)

        #print(f'Awake sensors: {awake_sensors}')
        #print(f'Rates (Transmissions per second) : {[a for a in self.transmission_freq_idxs]}')
        #print(f'Chunks sent: {list(self._chunks_sent)}')
        #print(f'Throughputs (# of sucessfull transmissions) : {self._throughputs}')
        print(f'Total throughput over observation: {total_throughput} succesfull transmissions')

        if total_throughput == 0:
            return (similarity, self._throughputs, [0]*self._num_sensors, [0]*self._num_sensors, rates)

        throughput_reward, similarity_reward, max_ind_set = self._calculate_rewards(redudancy_graph, sensor_effective_throughputs) 

        # Log sensor data and rewards 
        self.chunks_sent_log.append(list(self._chunks_sent))
        for i in range(self._num_sensors):
            self.throughput_log[i].append(self._throughputs[i])
            self.energy_log[i].append(self._energy[i])
            self.rate_log[i].append(self.transmission_freq_idxs[i])
            self._chunks_sent[i] = 0

        self.max_ind_set_log.append(max_ind_set)
        self.similarity_reward_log.append(similarity_reward)
        self.throughput_reward_log.append(throughput_reward)
        
        # Pickle logs for plotting 
        with open('figure_data.pkl', 'wb') as file:
            pickle.dump((self._sensor_ids, self._transmission_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.similarity_reward_log, self.throughput_reward_log, self.max_ind_set_log, self.chunks_sent_log), file)

        return (similarity, self._throughputs, throughput_reward, similarity_reward, rates)

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

            print(f"Received request from server for rates {unpacked_data}")

            rates = unpacked_data
            for sensor_idx in range(self._num_sensors):
                self.transmission_freq_idxs[sensor_idx] = rates[sensor_idx]

            self._send_observation_to_rl_agent(rates)
   
    """
    Create Mininet topology
    """
    def _create_topology(self):
        # build network
        self._net = Mininet_wifi(controller=Controller, link=wmediumd, wmediumd_mode=interference)

        info("*** Creating nodes\n")

        # create accesspoint
        self._net.addAccessPoint(f'ap23', ssid=f'new-ssid', mode='g', channel='5', position='50,50,0', cpu=1, mem=1024*2)
       
        #create cluster head
        self._cluster_head = self._net.addStation(f'ch', ip=f'192.168.0.100/24',
                                      range='150', position='30,30,0', cpu=1, mem=1024*2)
        
        #create sensors 
        self._sensors = []
        for i in range(self._num_sensors):
            ip_address = f'192.168.0.{i + 1}/24'
            self._sensors.append(self._net.addStation(f's{i}', ip=ip_address, range='116', position=f'{-90 - i},-90,0', cpu=1, mem=1024*2))

        info("*** Adding Controller\n")
        self._net.addController(f'c0')

        info("*** Configuring wifi nodes\n")
        self._net.configureWifiNodes()

        self._net.build()
        self._net.start()
   
    def __init__(self, sensor_ids, transmission_size=2*1024, transmission_frame_duration=1, num_transmission_frames=10000, observation_time=10, file_lines_per_chunk=10, log_directory='data/log', dataset_directory='data/towerdataset'):
        # Simulation parameters 
        self._sensor_ids = sensor_ids  
        self._transmission_size = transmission_size # In bytes
        self._transmission_frame_duration = transmission_frame_duration # Duration of a transmission frame in seconds 
        self._similarity_threshold = 1 # Threshold (in degrees) for two sensors to be considered similar 
        self._num_transmission_frames = num_transmission_frames # Total number of transmission frames 
        self._observation_time = observation_time # The number of seconds between each observation

        self._num_sensors = len(sensor_ids)
        self._throughputs = [0 for i in range(self._num_sensors)]
        self._previous_throughputs = [0 for i in range(self._num_sensors)]
        self._total_throughput = 0
        self._max_total_throughput = 0
        self._tcpdump_lines_parsed = [0 for i in range(self._num_sensors)]

        # Directories
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
       
        # Rate configuration
        self.transmission_freq_idxs = multiprocessing.Array('i', [1] * self._num_sensors)
        self._transmission_frequencies = np.array([10, 20, 30, 40]) # Possible number of times a sensor can transmit per frame

        # Energy configuration
        self._full_energy = 100
        self._recharge_time = 10 * transmission_frame_duration
        self._recharge_threshold = 20
        self._energy = [self._full_energy for _ in range(self._num_sensors)]

        # Data logs 
        self.rate_log = [[] for _ in range(self._num_sensors)]
        self.energy_log = [[] for _ in range(self._num_sensors)]
        self.throughput_log = [[] for _ in range(self._num_sensors)]
        self.reward_log = []
        self.similarity_reward_log = []
        self.throughput_reward_log = []
        self.max_ind_set_log = []
        self.chunks_sent_log = []
        self._chunks_sent = multiprocessing.Array('i', [0] * self._num_sensors)

        #self.chunks_sent = [0 for _ in range(self._num_sensors)] # List of the number of transmissions sent by each sensor

        # Delete the log folder if already it exists
        subprocess.run(["rm", "-rf", log_directory])

        # Create log directories 
        os.makedirs(log_directory)
        os.makedirs(log_directory + '/ch_received_data')
        os.makedirs(log_directory + '/error')
        os.makedirs(log_directory + '/pcaps')

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        # Initalize the dataset for each sensor 
        self._datasets = {}
        for i in range(self._num_sensors):
            tower_number = i + 2  # Start from tower2 
            file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
            if os.path.exists(file_path):
                #self._datasets[i] = self._preprocess_dataset_into_chunks(file_path, i, file_lines_per_chunk)
                self._datasets[i] = self._interpolate_dataset(file_path, i)
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []

        self._simulation_start_time = time.time()
        self._create_topology()

    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_idx (int): index of sensor
    """
    def _send_messages_to_cluster_head(self, sensor, ch_ip, sensor_idx):
        # Get the interpolated data corresponding to this sensor
        sensor_data = self._datasets[sensor_idx] 
        #info(f"Number of Chunks: {len(chunks)}\n")

        if not sensor_data:
            info(f"Sensor {sensor_idx}: No data available. Skipping send_messages.\n")
            return
        
        # Create a file to store the packets received by the cluster head
        sensor.cmd(f'touch {self._log_directory}/ch_received_data/sensor_{self._sensor_ids[sensor_idx]}.txt')
       
        # Index of the next chunk to send 
        #next_chunk_idx = 0
        chunks_sent = 0

        # Port to communicate with the cluser head on 
        port = 5001 + sensor_idx 
        
        # Initalize the sensors energy and the recharge count 
        energy = self._full_energy
        self._energy[sensor_idx] = self._full_energy
        charge_count = 0

        # Log the initial transmission rate
        self.rate_log[sensor_idx].append(self.transmission_freq_idxs[sensor_idx])
        filler = 'G' * (self._transmission_size)

        while chunks_sent < 10000:
            # Recharge sensor if energy is below the recharge threshold
            if self._energy[sensor_idx] < self._recharge_threshold:
                charge_count += 1

                recharge_time = time.time()
                rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
                info(f'Sensor {sensor_idx}: Energy {self._energy[sensor_idx]} below threshold ({self._recharge_threshold}). Recharging...')
                time.sleep(self._recharge_time)

                # Skip chunks that could have been transmitted during the recharge time 
                # next_chunk_idx += (self._recharge_time / self._transmission_frame_duration) // max(self._transmission_frequencies)
                                
                # Update the sensors energy and log the new energy level
                self._energy[sensor_idx] = self._full_energy
                sensor.energy = energy

                info(f'Sensor {sensor_idx}: Energy Recharged to full energy ({self._full_energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.')
                
                #if next_chunk_idx >= len(chunks):
                    #self._transmit_data_status[sensor_idx].set()
                #    break

            # Get the sensors current transmission rate
            transmit_rate = self._transmission_frequencies[self.transmission_freq_idxs[sensor_idx]] 
            
            # Sensor should skip tranmission during the current frame
            if transmit_rate == 0:
                #next_chunk_idx += int(max(self._transmission_frequencies))

                time.sleep(self._transmission_frame_duration)
                continue

            # Store the current time
            transmission_start_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(transmission_start_time))
            ms = int((transmission_start_time - time.time()) * 1000)

            # Get the chunk to send
            #info(f"next chunk idx: {next_chunk_idx}")
            #chunk = chunks[int(next_chunk_idx)]
            temp = f'{sensor_data(time.time() - self._simulation_start_time):.4f}' 
         
            # Send the chunk (with filler to pad the chunk to the correct length)
            cmd = f'echo "\n{temp}\n{filler[len(temp) + 6:]}\n" | nc -v -w0 -u {ch_ip} {port} >> {self._log_directory}/error/nc{sensor_idx} 2>&1 &'
            sensor.cmd(cmd)
            self._chunks_sent[sensor_idx] += 1

            #transmission_time = time.time() - transmission_start_time
            #info(f'Transmission time: {transmission_time}\n')

            """
            Chunks correspond with data for units of time i.e. chunk[0] corresponds to time 0, chunk[1] corresponds to time 1, 
            and so on. max(self._transmission_rates) is the maximum number of transmissions that could occur during a transmission 
            frame. transmit_rate is the number of tranmissions per frame currently made by the sensor. 
            (max(self._transmission_rates)  / transmit_rate) - 1 is therefore the number of tranmissions that are skipped per frame.   
            """

            #next_chunk_idx += (max(self._transmission_frequencies) / transmit_rate) - 1

            sleep_time = self._transmission_frame_duration / transmit_rate
            time.sleep(sleep_time)

        info(f"Sensor {sensor_idx}: Finished sending messages\n")


    """
    Start background netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _receive_messages(self, node):
        base_output_file = f'{self._log_directory}/ch_received_data/sensor'

        for i in range(self._num_sensors):
            # Create a file to store data received from sensor i
            output_file = f'{base_output_file}_{self._sensor_ids[i]}.txt'
            node.cmd(f'touch {output_file}')

            # Create a listener for sensor i 
            node.cmd(f'nc -n -vv -ul -p {5001 + i} -k >> {output_file} 2> {self._log_directory}/error/listen_err &')
            info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")

        # Capture the network by pcap
        pcap_file = f'{self._log_directory}/pcaps/capture.pcap'
        node.cmd(f'tcpdump -U -i {node.defaultIntf().name} -n udp portrange {5001}-{5001+self._num_sensors-1} -U -w {pcap_file} &')
        info(f"Receiver: Started tcpdump capture on ports 5001-{5001+self._num_sensors - 1}\n")

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
            info("*** Starting receivers\n")
            print("Starting receivers")
            
            # Activate cluster head listening threads
            receive_thread = threading.Thread(target=self._receive_messages, args=(self._cluster_head,))
            receive_thread.start()

            self._start_time = time.time()

            # Give listening thread time to start
            time.sleep(10) 
            
            # Start senders
            print("Starting senders")
            info("*** Starting senders\n")
            sender_threads = []
            ch_ip = f'192.168.0.100'

            for i, sensor in enumerate(self._sensors):
                tcpdump_file = f'{self._log_directory}/pcaps/tcpdump_sender_sensor{i}.pcap'
                sensor.cmd(f'tcpdump -U -i s{i}-wlan0 -w {tcpdump_file} &')
                
                thread = threading.Thread(target=self._send_messages_to_cluster_head, args=(sensor, ch_ip, i))
                thread.start()
                sender_threads.append(thread)

            # Wait for senders to finish
            for thread in sender_threads:
                thread.join()

            # Pickle simulation results
            #print(f'Simulation complete; saving data\n')
            #with open('figure_data.pkl', 'wb') as file:
            #    pickle.dump((self._sensor_ids, self._rate_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.clique_reward_log, self.throughput_reward_log, self.clique_log, self.chunks_sent), file)
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
        CLI(self._net)

        info("*** Stopping network\n")
        self._net.stop()

        self._server.close()

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
    transmission_size = 5*1024
    transmission_frame_duration = 1
    file_lines_per_chunk = 1
    num_transmission_frames = 3000

    cluster = sensor_cluster(sensor_ids, log_directory=f'data/log', observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, num_transmission_frames=num_transmission_frames)
    cluster.establish_connection_with_rl_agent()
    cluster_thread = threading.Thread(target=cluster.start, args=())
    cluster_thread.start()

    receive_rates_thread = threading.Thread(target=cluster.receive_rates_from_rl_agent, args=())
    receive_rates_thread.start()

    cluster_thread.join()
    receive_rates_thread.join()

