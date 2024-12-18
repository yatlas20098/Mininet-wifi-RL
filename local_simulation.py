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
from networkx.algorithms.clique import find_cliques as maximal_cliques
from itertools import combinations
from ortools.linear_solver import pywraplp 

from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from datetime import datetime, timedelta
from collections import defaultdict


"""Convert throughput from KiB, MiB, B to KB."""
def convert_to_KB(throughput):
    match = re.match(r"([0-9.]+)([KMG]i?B?/s?)", throughput.strip())
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        if 'ki' in unit:  # KiB
            return value
        elif 'mi' in unit:  # MiB
            return value * 1024 
        elif 'b' in unit:  # B
            return value / 1024
        else:
            return 0
    return 0

"""Extract the latest throughput from the file."""
def get_latest_throughput(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Get the last non-empty line with throughput
    latest_line = lines[-1].strip()
    
    # Extract the throughput value
    throughput_value = latest_line.strip('()')  # Remove the square brackets
    
    # Convert throughput to bytes
    throughput_KB = convert_to_KB(throughput_value)
    
    return throughput_KB 

class sensor_cluster():
    def _read_temperature_data(self, file_path, sensor_id):
        with open(file_path, 'r') as file:
            # Skip the first 4 lines
            data = []
            bytes_received = 0
            for line in file:
                bytes_received += len(line)
                # Ignore filler lines
                if line[0] == 'G':
                    continue
                try:
                    # Split the line and try to convert the temperature value (9th column, index 8) to float
                    temperature = np.float32(line.strip().split(',')[8])
                    data.append(temperature)
                except (ValueError, IndexError):
                    # If conversion fails or the line doesn't have enough columns, skip this line
                    continue

            self._throughputs[sensor_id] = bytes_received / 1024

        return np.array(data)

    """
    Split dataset into chunks

    Args:
        dataset_dir (string): Path to dataset file

    Returns:
        String List: List of chunks 
    """
    def _preprocess_dataset_into_chunks(self, dataset_dir, sensor_id):
        # Cache the dataset into memory 
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()

        # Get number of lines in file 
        file_size = len(lines)

        # Pad each line to have size packet size
        #for i in range(file_size):
        #    if len(lines[i]) < self._packet_size:
        #        lines[i] = lines[i] + 'G' * (self._packet_size - len(lines[i]) - 1) + '\n' 

        if file_size < self._chunks_to_send:
            print('The number of chunks to send must be decreased; not enough data')

        #file_lines_per_chunk = file_size // self._chunks_to_send
        file_lines_per_chunk = 30
        print(f'{file_lines_per_chunk} file lines per chunk')

        # Get the number of bytes in each chunk
        #chunk_size = int(max(self._rate_frequencies) * self._observation_time * 1024) # Convert from KB to bytes

        chunks = [lines[i:i + file_lines_per_chunk] for i in range(0, len(lines), file_lines_per_chunk)]
        chunks = [''.join(chunk) for chunk in chunks]
        #for i in range(len(chunks)):
        #    subprocess.run(["rm", "-rf", f"tmp/chunks{sensor_id}"])
        #    os.makedirs(f'tmp/chunks{sensor_id}')
        #    with open(f'tmp/chunks{sensor_id}/chunk{i}','w') as f:
        #            f.write(chunks[i])

        #filler_packet = 'G' * self._packet_size 
        #for i in range(chunks_to_send):
        #    num_filler_bytes = chunk_size - sum([len(line) for line in chunks[i]])
        #    num_filler_packets = num_filler_bytes // self._packet_size

        #    for j in range(num_filler_packets):
        #        chunks[i].append(filler_packet)

        #    if i % 100 == 0:
        #        print(f'Processed chunk {i} out of {chunks_to_send}')

        """
        # Get how many packets the file should be split into
        num_packets = math.ceil(file_size / self._packet_size)
        print(f'Number of packets: {num_packets}\n')

        # Get how many file lines each packet should have 
        lines_per_packet = len(lines) // num_packets
        if lines_per_packet == 0:
            print('Packet size must be increased\n')
        print(f'{lines_per_packet} lines per packet')

        # Split the file into packets
        packets = [lines[i:i + 1] for i in range(0, len(lines), 1)]
        packet_data = [''.join(packet) for packet in packets]

        # Group packets together into chunks to be transmitted 
        print(f'Chunk size: {chunk_size}')
        if chunk_size < self._packet_size:
            chunk_size = self._packet_size 
        num_chunks = math.ceil(file_size / chunk_size)
        packets_per_chunk = num_packets // num_chunks
        print(f'Number of chunks: {num_chunks}\n')

        return [packet_data[i:i + packets_per_chunk] for i in range(0, num_packets, packets_per_chunk)]
        """
        return chunks

    def _get_agent_obs(self):
        print('\n\nGetting observation')
        temperature_data = {}
        awake_sensors = []
        data_arrived = False 

        for i in range(self._num_sensors):
            file_name = f'ch_received_from_sensor_{self._sensor_ids[i]}.txt'

            # Create a copy of the original file
            subprocess.run(["cp", f'{self._log_directory}/{file_name}', f'{self._log_directory}/.{file_name}'])
            file_path = os.path.join(self._log_directory, f'.{file_name}')

            # Clear the file
            with open(f'{self._log_directory}/{file_name}', 'r+') as file:
                file.truncate(0)
            
            # Get the temperature data
            sensor_data = self._read_temperature_data(file_path, i)
            if len(sensor_data) > 0:
                temperature_data[i] = sensor_data
                awake_sensors.append(i)
                data_arrived = True
            # Skip sending observation if no data was observered
            # else:
            #    return

        #throughput = [0 for _ in range(self._num_sensors)]
        similarity = np.zeros((self._num_sensors, self._num_sensors))
        reward = 0

        if len(awake_sensors) > 0:
            min_length = min(len(temperature_data[sensor]) for sensor in awake_sensors)

        # Truncate all arrays to the shortest length
        temperature_data = {sensor: data[:min_length] for sensor, data in temperature_data.items()}

        # Get the number of sensors that generated data
        n_sensors_awake = len(awake_sensors)

        # Create a graph with veritices representing sensors and edges denoting similarity
        G = nx.Graph()

        print(f'Awake sensors: {awake_sensors}')
        #self._throughputs = [0 for i in range(self._num_sensors)]
        
        for i in range(self._num_sensors):
            G.add_node(i)

            # throughput_file = f'{self._log_directory}/sensor_{self._sensor_ids[i]}_throughput.txt'
            # Convert throughput from KB received to KB received / s
            self._throughputs[i] /= ((self._observation_time) * 3) 

        for i in awake_sensors:
            print(f'Sensor{self._sensor_ids[i]} temp: {temperature_data[i][0]}')
            for j in awake_sensors:
                similarity[i,j] = int(all(abs(a - b) <= self._similarity_threshold for a,b in zip(temperature_data[i], temperature_data[j])))
                if similarity[i,j] == 1 and not i == j:
                    print(f'Node {i} ~ Node {j}')
                    G.add_edge(i,j)
            
        print(f'Rates (Kib/s) : {self._rate_frequencies}')
        max_throughput = (max(self._rate_frequencies))

        print(f'Throughputs (Kib/s) : {self._throughputs}')
        print(f'Total throughput over observation: {np.sum(self._throughputs)} KiB/s')
        print(f'Maximum attainable throughput for a sensor (Kib/s): {max_throughput}')
        print(f'Maximum attainable throughput all sensors (Kib/s): {self._num_sensors * max_throughput}')

        # Get a maximal clique cover for G
        maximal_clique_cover = list(maximal_cliques(G)) 
        print(f'maximal_clique_cover: {maximal_clique_cover}')
        print(f'Max throughput in clique: {[max([self._throughputs[node] for node in clique]) for clique in maximal_clique_cover]}')


        # Reward high throughput of non-redudant data
        max_clique_throughputs = [max([self._throughputs[node] / (max_throughput) for node in clique]) for clique in maximal_clique_cover] 
        reward = 0.5 * (np.median(max_clique_throughputs) - 1)
        reward += 0.5 * ((np.sum(self._throughputs) / (self._num_sensors * max_throughput)) - 1)

        print(f'Weighted avg energy: {np.average(self._energy)/100}')
        print(f'Reward: {reward}')


        for i in range(self._num_sensors):
            self.throughput_log[i].append(self._throughputs[i])
            self.energy_log[i].append(self._energy[i])
            self.rate_log[i].append(self._rates[i])

        self.clique_log.append(maximal_clique_cover)

        self.reward_log.append(reward)
        
        obs = pickle.dumps((self._rates_id, similarity, [e / self._full_energy for e in self._energy], self._throughputs, reward))

        # Send the size of the observation in bytes to the server.
        obs_length = len(obs)
        self._server.sendall(struct.pack('!I', obs_length))

        # Send the observation to the server.
        self._server.sendall(obs)
        print('Sent observation to server\n\n')

    """
    Create Mininet topology
    """
    def _create_topology(self):
            #build network
            self._net = Mininet_wifi(controller=Controller, link=wmediumd,
                               wmediumd_mode=interference)

            info("*** Creating nodes\n")
            #create accesspoint
            ap23 = self._net.addAccessPoint('ap23', ssid='new-ssid', mode='g', channel='5', position='50,50,0', cpu=1, mem=1024*2)
            
            #create 10 sensors
            self._sensors = []
            for i in range(self._num_sensors):
                ip_address = f'192.168.0.{i + 1}/24'
                self._sensors.append(self._net.addStation(f's{i}', ip=ip_address,
                                              range='116', position=f'{-30 - i},-30,0', cpu=1, mem=1024*2))
            #create cluster head
            self._cluster_head = self._net.addStation('ch', ip='192.168.0.100/24',
                                          range='150', position='70,70,0', cpu=1, mem=1024*2)
            
            info("*** Adding Controller\n")
            self._net.addController('c0')

            info("*** Configuring wifi nodes\n")
            self._net.configureWifiNodes()

            self._cluster_head.cmd(f'sysctl -w net.ipv4.ip_no_pmtu_disc=0 >> err.txt 2>&1')
            self._cluster_head.cmd(f'sysctl -w net.core.rmem_max=16777216 >> err.txt 2>&1')
            self._cluster_head.cmd(f'sysctl -w net.core.wmem_max=16777216 >> err.txt 2>&1')

            for i in range(self._num_sensors):
                self._sensors[i].cmd(f'sysctl -w net.ipv4.ip_no_pmtu_disc=0 >> err.txt 2>&1')
                self._sensors[i].cmd(f'sysctl -w net.core.wmem_max=16777216 >> err.txt 2>&1')
                self._sensors[i].cmd(f'sysctl -w net.core.rmem_max=16777216 >> err.txt 2>&1')

            #self._net.setPropagationModel(model='logDistance', exp=2)

    """
    Creates a socket between the server and cluster head. To avoid networking complications with an HPC firewall, the HPC server approaches the cluster head when creating a connection.   
    """
    def _establish_server_connection(self):
        # TODO: Create UDP socket for sending message to server and TCP socket for receiving messages from server 
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket succesfully created")
        listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listen_port = 5000 + self._cluster_id
        listen.bind(("0.0.0.0", port))
        print(f'Socket binded to {port}')

        listen.listen(5)
        print("Socket is listening")

        server, addr = listen.accept()
        print('Got connection from', addr)

        self._server = server
        #self._server.settimeout(0)

    def __init__(self, cluster_id, sensor_ids, server_ip, server_port, num_sensors=10, packet_size=2*1024, log_directory='data/log', dataset_directory='data/towerdataset'):
        # Connection and configuration parameters
        self._cluster_id = cluster_id
        self._server_ip = server_ip
        self._server_port = server_port
        self._packet_size = int(packet_size) # in bytes
        self._sensor_ids = sensor_ids
        self._num_sensors = len(sensor_ids)
        self._similarity_threshold = 1
        self._throughputs = [0 for i in range(num_sensors)]
        self._chunks_to_send = 2000
        self._observation_time = 0.1 # The number of seconds between each observation

        # Directories
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
        self._output_dir = f'{self._log_directory}/output/extracted_data' # Where to output logs 
       
        # Rate configuration
        self._rates_id = 0
        self._rates = [2] * num_sensors
        self._rate_frequencies = np.array([0, 10, 20, 30]) # Rate frequencies in KB

        # Initalize semaphores and events 
        self._update_rates = threading.Semaphore(num_sensors) 
        self._transmit_data_status = [threading.Event() for _ in range(num_sensors)]
        self._ready_to_transmit = [threading.Event() for _ in range(num_sensors)]

        self._update_rates_status = [threading.Event() for _ in range(num_sensors)]
        for update_rate_status in self._update_rates_status:
            update_rate_status.set()
        self._bytes_received = [0 for _ in range(num_sensors)]
        for transmit_status in self._transmit_data_status:
            transmit_status.clear()

       
        # Energy configuration
        self._full_energy = 100
        self._recharge_time = 3
        self._recharge_threshold = 20
        self._energy = [self._full_energy for _ in range(num_sensors)]

        # Logs for tracking performance 
        self.rate_log = [[] for _ in range(num_sensors)]
        self.energy_log = [[] for _ in range(num_sensors)]
        self.throughput_log = [[] for _ in range(num_sensors)]
        self.reward_log = []
        self.clique_log = []

        # RL parameters
        self._alpha = 3
        self._beta = 0.5

        # Delete log folder if already it exists
        subprocess.run(["rm", "-rf", log_directory])
        os.makedirs(log_directory)
        os.makedirs(log_directory + '/tmp')

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        # Initalize the dataset for each sensor 
        self._datasets = {}
        for i in range(self._num_sensors):
            tower_number = i + 2  # Start from tower2 
            file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
            if os.path.exists(file_path):
                self._datasets[i] = self._preprocess_dataset_into_chunks(file_path, i)
                self._datasets[i] = self._datasets[i] + self._datasets[i]
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []

        #print(f'Dataset: {self._datase}')
        # Find the length of the shortest dataset 
        min_length = min(len(chunks) for chunks in self._datasets.values())

        # Truncate all chunks to the shortest length
        # Duplicaite for more data
        # for i in range(self._num_sensors):
        #    self._datasets[i] = self._datasets[i][:min_length] + self._datasets[i][:min_length]
        
        self._establish_server_connection()
        self._create_topology()
        
    def _receive_rates_from_server(self):
        chunks_sent = 0
        while(True):
            # Wait for current data transmissions to end
            for transmit_status in self._transmit_data_status:
                transmit_status.wait()
                transmit_status.clear()

            for transmit_status in self._ready_to_transmit:
                transmit_status.clear()
           
            # Give data time to arrive
            time.sleep(0.1)

            # Send observation to server
            print('\n\nGetting Observation')

            if chunks_sent % 3 == 0:
                chunks_sent += 1
                for update_rate in self._update_rates_status:
                    update_rate.set()
                continue

            self._get_agent_obs()

            print('Getting new rates from server')
            try:
                # Get the packed new transmission rates from server
                packed_data = self._server.recv(4 * (self._num_sensors + 1))
                if len(packed_data) == 0:
                    print("Connection terminated by server")
                    break

                # Unpack the new transmission rates
                unpacked_data = struct.unpack('!' + 'i'*(self._num_sensors + 1), packed_data)

                # Update the current transmission rates
                rates = unpacked_data[1:]
                for sensor_id in range(self._num_sensors):
                    self._rates[sensor_id] = rates[sensor_id]

                self._rates_id = unpacked_data[0]
                print(f'New rates {rates}.\nID = {self._rates_id}\n\n')

                chunks_sent += 1

                if chunks_sent % 30 == 0:
                    with open('figure_data.pkl', 'wb') as file:
                        pickle.dump((self._sensor_ids, self._rate_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.clique_log), file)


            except socket.error as e:
                print("No new rates from server\n\n")

            # Allow next data transmission to begin
            for update_rate in self._update_rates_status:
                update_rate.set()

    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_id (int): id of sensor
    """

    def _send_messages_to_cluster_head(self, sensor, ch_ip, sensor_id):
        # Get the chunks corresponding to this sensor
        chunks = self._datasets[sensor_id] # Every chunk contains enough data for transmission at the highest rate for an observation period 
        info(f"Number of Chunks: {len(chunks)}\n")

        if not chunks:
            info(f"Sensor {sensor_id}: No data available. Skipping send_messages.\n")
            return
        
        packetnumber = 0
        port = 5001 + sensor_id

        # Create a file to track the packets received by the cluster head
        sensor.cmd(f'touch {self._log_directory}/sensor_{sensor_id}_log.txt')
       
        # Track the number of chunks sent 
        chunks_sent = 0
        
        # Initalize the sensors energy and the recharge count 
        energy = self._full_energy
        charge_count = 0

        # Log the initial transmission rate
        self.rate_log[sensor_id].append(self._rates[sensor_id])

        filler_packet = 'G' * (self._packet_size - 1) + '\n' 

        while chunks_sent < len(chunks):
            # Recharge sensor if energy is below the recharge threshold
            if self._energy[sensor_id] < self._recharge_threshold:
                charge_count += 1
                recharge_time = time.time()
                rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy below threshold ({self._recharge_threshold}). Recharging...\n')
                                
                for _ in range(self._recharge_time):
                    # Wait for rate update
                    self._update_rates_status[sensor_id].wait()
                    self._update_rates_status[sensor_id].clear()

                    # Allow other sensors to transmit data 
                    self._ready_to_transmit[sensor_id].set()

                    file_name = f'ch_received_from_sensor_{self._sensor_ids[sensor_id]}.txt'
                    # Clear the file
                    # with open(f'{self._log_directory}/{file_name}', 'r+') as file:
                    #    file.truncate(0)

                    time.sleep(self._observation_time)

                    # Discard chunk that should have been sent during the recharge time
                    chunks_sent += 1

                    # Allow the transmission rate to update
                    self._transmit_data_status[sensor_id].set()
                    
                # Update the sensors energy and log the new energy level
                self._energy[sensor_id] = self._full_energy
                sensor.energy = energy

                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy Recharged to full energy ({self._full_energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.\n')
                
                if chunks_sent >= len(chunks):
                    self._transmit_data_status[sensor_id].set()
                    break

            # Check for rate update before transmitting the next chunck
            self._update_rates_status[sensor_id].wait()
            self._update_rates_status[sensor_id].clear()

            # Get the current transmission rate
            rate = self._rates[sensor_id]
            #rate = chunks_sent % len(self._rate_frequencies) 

            # Get the chunk to send
            chunk = chunks[chunks_sent]
            chunk = chunk + 'G'*(int(self._rate_frequencies[rate] * 1024 * self._observation_time - len(chunk))) 

            #chunk_str = chunk + 'G'*

            # Add filler to file lines so that they have size self._packet_size
            #chunk = [packet + 'G'*(self._packet_size - len(packet)) for packet in chunk]

            #packets_to_send = int(self._rate_frequencies[rate] * 1024 * self._observation_time // self._packet_size)
            
            # Add filler packets
            #while len(chunk) < packets_to_send:
            #    chunk += filler_packet
               
            file_name = f'ch_received_from_sensor_{self._sensor_ids[sensor_id]}.txt'
            

            # Wait until all other sensors are able to transmit
            self._ready_to_transmit[sensor_id].set()
            for transmit_status in self._ready_to_transmit:
                transmit_status.wait()

            # Clear the file
            #with open(f'{self._log_directory}/{file_name}', 'r+') as file:
            #    file.truncate(0)

            # Store the current time
            transmission_start_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(transmission_start_time))
            ms = int((transmission_start_time - time.time()) * 1000)

            bytes_to_transmit = int(self._rate_frequencies[rate] * 1024 * self._observation_time)
            with open(f'tmp/chunk{sensor_id}.txt','w') as f:
                f.write(chunk[:bytes_to_transmit])

            # Send the current chunck if the frequency rate is > 0 
            if self._rate_frequencies[self._rates[sensor_id]] > 0:
                info(f"Sensor {sensor_id}: Sending chunk {chunks_sent} of {(len(chunk) / 1024):.2f} KiB at {timestamp}.{ms:03d}\n")
                cmd = f'cat tmp/chunk{sensor_id}.txt | nc -v -q 1 -u {ch_ip} {port} >> tmp/nc{sensor_id} 2>&1 &'
                sensor.cmd(cmd)

                self._energy[sensor_id] -= 4 * self._rate_frequencies[rate] / max(self._rate_frequencies)
                info(f"Sensor {sensor_id}: Sent chunk {chunks_sent} of size {(len(chunk) / 1024):.2f} KiB at {timestamp}.{ms:03d}\n")
            else:
                self._energy[sensor_id] -= 0.3
                info(f"Sensor {sensor_id}: Skipped sending chunk {chunks_sent} due to rate 0 at {timestamp}.{ms:03d}\n")

            chunks_sent += 1

            info(f'Sent {bytes_to_transmit/1024} KiB at {self._rate_frequencies[rate]} KiB/s\n')
            transmission_time = time.time() - transmission_start_time
            info(f'Transmission time: {transmission_time}\n')

            if transmission_time < self._observation_time:
                time.sleep(self._observation_time - transmission_time)

            # Allow the transmission rate to update
            self._transmit_data_status[sensor_id].set()

        info(f"Sensor {sensor_id}: Finished sending messages\n")
        self._transmit_data_status[sensor_id].set()


    def _send_obs_to_server(self):
        #self._get_agent_obs()
        """
        obs = self._get_agent_obs()

        # Send the size of the observation in bytes to the server.
        obs_length = len(obs)
        self._server.sendall(struct.pack('!I', obs_length))

        # Send the observation to the server.
        self._server.sendall(obs)
        print('Sent observation to server')
        """
   
    def _transfer_received_data(self, min_lines=10):
        file_dir = self._log_directory
        data_sent = False

        # Wait for first date tranmissions to finish before sending an observation
        #for data_transmission_status in self._data_transmission_status:
        #    data_transmission_status.wait()
        """ 
        while(True):
            self._cluster_head.cmd(f'netstat -i >> {self._log_directory}/netstat_output.txt')
            self._cluster_head.cmd(f'ss -i >> {self._log_directory}/ss_output.txt')
            self._cluster_head.cmd(f'iw dev ch-wlan0 link >> {self._log_directory}/iw.txt 2>> {self._log_directory}/iw_err.txt')
            self._cluster_head.cmd(f'ss -u -a >> {self._log_directory}/drops.txt')

            # Skip observation if not enough data received 
            for i in range(self._num_sensors):
                file_name = f'ch_received_from_sensor_{i}.txt'
                if not os.path.exists(f'{file_dir}/{file_name}'):
                   time.sleep(self._observation_time) 
                with open(f'{file_dir}/{file_name}', 'r+') as file:
                    while sum(1 for _ in file) < min_lines:
                       time.sleep(self._observation_time) 
            #self._get_agent_obs()
            time.sleep(self._observation_time)
        """

    """
    Start background netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _receive_messages(self, node):
        base_output_file = f'{self._log_directory}/ch_received_from_sensor'

        for i in range(self._num_sensors):
            # save the data received from a sensor to their respective file
            output_file = f'{base_output_file}_{self._sensor_ids[i]}.txt'
            node.cmd(f'touch {output_file}')

            #listen_cmd = f'nohup nc -n -v -ulk -p {5001 + i} 2> {self._log_directory}/listen_err{i} | pv -m 1 -F "%a" >> {output_file} 2>{self._log_directory}/sensor_{self._sensor_ids[i]}_throughput.txt &'
           # node.cmd(listen_cmd)

            node.cmd(f'nc -n -vv -ul -p {5001 + i} -k >> {output_file} 2> {self._log_directory}/listen_err &')

            #node.cmd(f'echo "{listen_cmd}" | bash 2>> {self._log_directory}/listen_cmd_err.txt')
            info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")

        #capture the network by pcap
        pcap_file = f'{self._log_directory}/capture.pcap'
        node.cmd(f'tcpdump -i {node.defaultIntf().name} -n udp portrange 5001-{5001+self._num_sensors-1} -U -w {pcap_file} &')
        info(f"Receiver: Started tcpdump capture on ports 5001-{5001+self._num_sensors-1}\n")

    """
    Kill netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _stop_receivers(self, node):
        #stop receiver
        node.cmd('pkill -f "nc -ul"')
        node.cmd('pkill tcpdump')
        info("Stopped all nc receivers and tcpdump\n")

    """
    Begin message transmission from sensors to cluster head and reap all threads after transmission concludes.   
    """
    def start(self):
        self._net.build()
        self._net.start()

        info("*** Setting up communication flow\n")
        try:
            info("*** Starting receivers\n")
            
            # Activate cluster head listening threads
            receive_thread = threading.Thread(target=self._receive_messages, args=(self._cluster_head,))
            receive_thread.start()

            self._start_time = time.time()

            # Give listening threads time to start
            time.sleep(10) 

            info("*** Starting senders\n")
            sender_threads = []
            listener_threads = []
            ch_ip = '192.168.0.100'

            for i, sensor in enumerate(self._sensors):
                tcpdump_file = f'{self._log_directory}/tcpdump_sender_sensor{i}.pcap'
                sensor.cmd(f'tcpdump -i s{i}-wlan0 -w {tcpdump_file} &')
                
                thread = threading.Thread(target=self._send_messages_to_cluster_head, args=(sensor, ch_ip, i))
                thread.start()
                sender_threads.append(thread)

            time.sleep(self._observation_time)
           
            rates_thread = threading.Thread(target=self._receive_rates_from_server, args=())
            rates_thread.start()

            # Give cluster head time to receive data
            time.sleep(2 * self._observation_time)
            transfer_thread = threading.Thread(target=self._transfer_received_data, args=())
            transfer_thread.start()

            """
            time.sleep(40)
            self._plot_energy()
            self._plot_throughput()
            self._plot_rates(3)
            self._create_plot(self, self.rewards, 'rewards', self.timestaps, 'Time (seconds)', 'Reward', 'Reward over Time')
            self._create_plot(self, self.similarity, 'Similarity Penalty', self.timestaps, 'Time (seconds)', 'Penalty', 'Similarity Penalty over Time')
            """
            
            for thread in sender_threads:
                thread.join()

            print(f'Simulation complete; saving data\n')
            with open('figure_data.pkl', 'wb') as file:
                pickle.dump((self._sensor_ids, self._rate_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.clique_log), file)
            print(f'Pickle dump succesfuly made\n')

            self._stop_receivers(cluster_head)
            receive_thread.join()

            for thread in listener_threads:
                thread.join()
        
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

    def _parse_tcpdump_output(self, file_path):
        print("Starting to parse tcpdump output...")
        start_time = time.time()
        udp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d+)\sIP\s(\d+\.\d+\.\d+\.\d+)\.(\d+)\s>\s(\d+\.\d+\.\d+\.\d+)\.(\d+):\sUDP,\slength\s(\d+)')
        sensor_packets = defaultdict(list)

        with open(file_path, 'r') as file:
            print(f'Lines parsed: self._tcpdump_lines_parsed')
            lines = file.readlines()
            if len(lines) > self._tcpdump_lines_parsed:
                lines = lines[self._tcpdump_lines_parsed]
                total_lines = len(lines)
                self._tcpdump_lines_parsed += total_lines
                for i, line in enumerate(lines):
                    if i % 10000 == 0:
                        print(f"Parsing progress: {i}/{total_lines} lines ({i/total_lines*100:.2f}%)")
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
        #return throughput_data

    def _create_plot(self, data, output_name, timestamps, xlabel, ylabel, title):
        print(f"Plotting {output_name}...\n")

        plt.figure(figize=(12,6))
        plt.plot(data, self.time_stamps)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(yabel)
        plt.title(title)
        plt.grid(True)
        output_path = f"{self.output_dir}/output_name.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Done plotting\n")

    
    def _plot_rates(self, sampling_freq):
        print("Plotting rates...")
        rates = zip(*self.rates)
        time = [t + 1 for t in range(len(rates))]
        num_sensors_with_rate = []

        for t in time:
            num_sensors_with_rate.append([rates[t].count(n) for n in range(sampling_freq)])
        num_sensors_with_rate = zip(*num_sensors_with_rate)

        plt.figure(figsize=(12, 6))
        plt.xlabel('Time (seconds)')
        plt.ylabel('Number of Sensors with Rate)')
        plt.title('Resdiual Energy over Time for All Sensors')
        
        for rate in range(sampling_freq):
            plt.plot(time, num_sensors_with_rate[rate], label=f'self._rate_sizes[rate] Kbps')

        plt.grid(True)
        plt.legend()
        output_path = f"{self.output_dir}/rates.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Plot for all sensors saved to {output_path}")

    def _plot_energy(self):
        print("Plotting energy...")

        plt.figure(figsize=(12, 6))
        energy = [np.average(e) for e in zip(*self.energy_log)]
        time = [t + 1 for t in range(len(energy))]
        plt.plot(time, energy)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Resdiual Energy (?)')
        plt.title('Resdiual Energy over Time for All Sensors')
        plt.grid(True)
        output_path = f"{self.output_dir}/energy_plot_all_sensors.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Plot for all sensors saved to {output_path}")

        for sensor in range(sensor_id):
            plt.figure(figsize=(12,6))
            energy = [e[i] for e in zip(*self.energy_log)]
            plt.plot(time, energy, label=sensor_id)
            plt.xlabel('Time (seconds)')
            plt.ylabel('Resdiual Energy')
            plt.title(f'Resdiual Energy for Sensor {sensor_id}')
            plt.legend()
            plt.grid(True)
            output_path = f"{self.output_dir}/resdiual_energy_plot_{sensor_id}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Plot for sensor {sensor_id} saved to {output_path}")

    """
    def _plot_throughput(self):
        print("Plotting throughput...")
        output_dir = f'self.{log_directory}/graphics'

        plt.figure(figsize=(12, 6))
        throughput = [np.sum(throughputs) for throughputs in zip(*self.throughputs)]
        time = [t + 1 for t in range(len(throughput)]
        plt.plot(time, throughput)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Throughput (Kbps)')
        plt.title('Throughput over Time for All Sensors')
        plt.grid(True)
        output_path = f"{output_dir}/throughput_plot_all_sensors.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Plot for all sensors saved to {output_path}")

        for sensor in range(sensor_id):
            plt.figure(figsize=(12,6))
            throughput = [throughputs[i] for throughputs in zip(*self.throughputs)]
            plt.plot(time, throughput, label=sensor_id)
            plt.xlabel('Time')
            plt.ylabel('Throughput (Kbps)')
            plt.title(f'Throughput Over Time for Sensor {sensor_id}')
            plt.legend()
            plt.grid(True)
            output_path = f"{output_dir}/throughput_plot_{sensor_id}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Plot for sensor {sensor_id} saved to {output_path}")
        """

    """
    def main():
        input_file = '/mydata/mydata/RL_agent/output/extracted_data/tcpdump_output_capture.txt'
        output_dir = '/mydata/mydata/RL_agent/output/extracted_data'

        overall_start_time = time.time()

        sensor_packets = self._parse_tcpdump_output(input_file)
        sensor_throughput = {}

        print(f"Processing throughput for {len(senor_packets)} sensors...")
        for i, (sensor_ip, packets) in enumerate(sensor_packets.items()):
            print(f"Processing sensor {i+1}/{len(sensor_packets)}: {sensor_ip}")
            self._aggregate_throughput(sensor_ip, packets)
            #sensor_throughput[sensor_ip] = throughput_data

        #plot_throughput(sensor_throughput, output_dir)

        print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")
    """

if __name__ == '__main__':
    setLogLevel('info')
    server_ip = '192.168.20.201'
    port = 5000
    sensor_ids = range(5, 15)
    cluster = sensor_cluster(1, sensor_ids, server_ip, port, packet_size=1472*1)
    cluster.start()
