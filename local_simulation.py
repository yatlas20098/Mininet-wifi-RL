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

from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
from datetime import datetime, timedelta
from collections import defaultdict

def read_temperature_data(file_path):
    with open(file_path, 'r') as file:
        # Skip the first 4 lines
        data = []
        for line in file:
            try:
                # Split the line and try to convert the temperature value (9th column, index 8) to float
                temperature = np.float32(line.strip().split(',')[8])
                data.append(temperature)
            except (ValueError, IndexError):
                # If conversion fails or the line doesn't have enough columns, skip this line
                continue

    return np.array(data)

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
    throughput_value = latest_line.strip('[]')  # Remove the square brackets
    
    # Convert throughput to bytes
    throughput_KB = convert_to_KB(throughput_value)
    
    return throughput_KB 

class RewardNormalizer:
    def __init__(self, alpha=0.99):
        self.mean = 0
        self.std = 1
        self.alpha = alpha

    def normalize(self, reward):
        self.mean = self.alpha * self.std + (1 - self.alpha) * reward
        self.std  = self.alpha * self.std + (1 - self.alpha) * (reward - self.mean)**2
        self.std = np.sqrt(self.std)

        normalized_reward = (reward - self.mean) / (self.std + 1e-8)

        scaled_reward = 1 - ((np.tanh(normalized_reward) + 1) / 2) # bound between 0 and 1 and invert

        return scaled_reward 

class sensor_cluster():
    """
    Split dataset into chunks

    Args:
        dataset_dir (string): Path to dataset file

    Returns:
        String List: List of chunks 
    """
    def _preprocess_dataset_into_chunks(self, dataset_dir):
        # Cache the dataset into memory 
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()

        # Get file size in bytes
        file_size = sum([len(line.encode('utf-8')) for line in lines])

        # Get how many packets the file should be split into
        num_packets = math.ceil(file_size / self._packet_size)
        print(f'Number of packets: {num_packets}\n')

        # Get how many file lines each packet should have 
        lines_per_packet = len(lines) // num_packets
        if lines_per_packet == 0:
            print('Packet size must be increased\n')

        # Split the file into packets
        packets = [lines[i:i + lines_per_packet] for i in range(0, len(lines), lines_per_packet)]
        packet_data = [''.join(packet) for packet in packets]

        # Group packets together into chunks to be transmitted 
        chunk_size = max(self._rate_frequencies) * self._observation_time * 1024 # Convert from KB to bytes
        num_chunks = math.ceil(file_size / chunk_size)
        packets_per_chunk = num_packets // num_chunks
        print(f'Number of chunks: {num_chunks}\n')

        return [packet_data[i:i + packets_per_chunk] for i in range(0, num_packets, packets_per_chunk)]

    def _get_agent_obs(self):
        temperature_data = []
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
            sensor_data = read_temperature_data(file_path)
            if len(sensor_data) > 0:
                temperature_data.append(sensor_data)
                awake_sensors.append(i)
                data_arrived = True
            # Skip sending observation if no data was observered
            else:
                return

        throughput = [0 for _ in range(self._num_sensors)]
        similarity = np.zeros((self._num_sensors, self._num_sensors))
        similarity_penalty = 0
        weighted_similarity_penalty = 0
        reward = 0

        if data_arrived:
            # Find the length of the shortest temperature data array
            min_length = min(len(data) for data in temperature_data)

            # Truncate all arrays to the shortest length
            truncated_data = [data[:min_length] for data in temperature_data]
            temperature_data = np.array(truncated_data)

            # Calculate similarity
            similarity = np.zeros((self._num_sensors, self._num_sensors))
            similarity_penalty = 0

            # Get the number of sensors that generated data
            n_sensors_awake = len(awake_sensors)

            for i in range(n_sensors_awake):
                for j in range(i + 1, n_sensors_awake):
                    # Calculate RMSE using euclidan norm
                    similarity[awake_sensors[i]][awake_sensors[j]] = np.linalg.norm(temperature_data[i] - temperature_data[j], ord=2)/np.sqrt(n_sensors_awake)

                    # Penalize pairs with high similarity and high frequency rates
                    similarity_penalty += similarity[awake_sensors[i]][awake_sensors[j]] * (self._rates[awake_sensors[i]] + self._rates[awake_sensors[j]])
            
                throughput_file = f'{self._log_directory}/sensor_{self._sensor_ids[i]}_throughput.txt' 
                throughput[i] = get_latest_throughput(throughput_file)

            if np.sum(throughput) > self._max_throughput:
                self._max_throughput = np.sum(throughput)

            ### Calculate energy efficiency
            # Utility combines penalties for redundancy and rewards for high average energy
            weighted_similarity_penalty = self._normalizer.normalize(similarity_penalty/10000)
            reward = np.sum(throughput)/self._max_throughput*weighted_similarity_penalty + self._beta * np.average(self._energy)/100 

        print(f'Average throughput over past half second: {np.sum(throughput)} KB/S')
        print(f'Similairty Penalty: {similarity_penalty}')
        print(f'Weighted Similairty Penalty: {weighted_similarity_penalty}')
        print(f'Weighted avg energy: {np.average(self._energy)/100}')
        print(f'Weighted throughputs: {np.sum(throughput)/self._max_throughput}')
        print(f'Reward: {reward}')

        for i in range(self._num_sensors):
            self.throughput_log[i].append(throughput[i])
            self.energy_log[i].append(self._energy[i])
            self.rate_log[i].append(self._rates[i])

        self.reward_log.append(reward)
        self.similarity_log.append(similarity_penalty)
        
        obs = pickle.dumps((self._rates_id, temperature_data, self._energy, throughput, reward))

        # Send the size of the observation in bytes to the server.
        obs_length = len(obs)
        self._server.sendall(struct.pack('!I', obs_length))

        # Send the observation to the server.
        self._server.sendall(obs)
        print('Sent observation to server')

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

            self._net.setPropagationModel(model='logDistance', exp=5)

    """
    Creates a socket between the server and cluster head. To avoid networking complications with an HPC firewall, the HPC server approaches the cluster head when creating a connection.   
    """
    def _establish_server_connection(self):
        # TODO: Create UDP socket for sending message to server and TCP socket for receiving messages from server 
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket succesfully created")
        listen_port = 5000 + self._cluster_id
        listen.bind(("0.0.0.0", port))
        print(f'Socket binded to {port}')

        listen.listen(5)
        print("Socket is listening")

        server, addr = listen.accept()
        print('Got connection from', addr)

        self._server = server

    def __init__(self, cluster_id, sensor_ids, server_ip, server_port, num_sensors=10, packet_size=2*1024, log_directory='data/log', dataset_directory='data/towerdataset'):
        # Connection and configuration parameters
        self._cluster_id = cluster_id
        self._server_ip = server_ip
        self._server_port = server_port
        self._packet_size = packet_size
        self._sensor_ids = sensor_ids
        self._num_sensors = len(sensor_ids) 

        # Directories
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
        self._output_dir = f'{self._log_directory}/output/extracted_data' # Where to output logs 
       
        # Rate configuration
        self._rates_id = 0
        self._rates = [2] * num_sensors
        self._rate_frequencies = [20, 50, 100] # Rate frequencies in KB

        # Initalize semaphores and events 
        self._update_rates = threading.Semaphore(num_sensors) 
        self._data_transmission_status = [threading.Event() for _ in range(num_sensors)] 
        self._bytes_received = [0 for _ in range(num_sensors)]
       
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
        self.similarity_log = []
        self._max_throughput = 1

        # RL parameters
        self._alpha = 0.6
        self._beta = 0.3
        self._normalizer = RewardNormalizer()
        self._observation_time = 1 # The number of seconds between each observation

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
                self._datasets[i] = self._preprocess_dataset_into_chunks(file_path)
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []
        
        self._create_topology()
        self._establish_server_connection()

    def _receive_rates_from_server(self):
        while(True):
            try:
                # Get the packed new transmission rates from server
                packed_data = self._server.recv(4 * (self._num_sensors + 1))
                if len(packed_data) == 0:
                    print("Connection terminated by server")
                    break

                # Unpack the new transmission rates
                print("Received new rates from server\n")
                unpacked_data = struct.unpack('!' + 'i'*(self._num_sensors + 1), packed_data)

                # Wait for current data transmissions to end
                for _ in range(self._num_sensors): 
                    self._update_rates.acquire()

                # Update the current transmission rates
                rates = unpacked_data[1:]
                for sensor_id in range(self._num_sensors):
                    self._rates[sensor_id] = rates[sensor_id]

                # Allow data transmissions to continue 
                for _ in range(self._num_sensors): 
                    self._update_rates.release()

                self._rates_id = unpacked_data[0]
                print(f'New rates {rates}.\nID = {self._rates_id}\n')

            except socket.error as e:
                print("Error receiving rates from server:", e)

    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_id (int): id of sensor
    """

    def _send_messages_to_cluster_head(self, sensor, ch_ip, sensor_id):
        # Get the chunks corresponding to this sensor
        chunks = self._datasets[sensor_id] # Every chunk contains a seconds' worth of data
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

        while chunks_sent != len(chunks):
            # Recharge sensor if energy is below the recharge threshold
            if self._energy[sensor_id] < self._recharge_threshold:
                charge_count += 1
                recharge_time = time.time()
                rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy below threshold ({self._recharge_threshold}). Recharging...\n')

                # Discard chunks that should have been sent during the recharge time
                chunks_sent += self._recharge_time
                if chunks_sent >= len(chunks):
                    break
                
                time.sleep(self._recharge_time)
                # Log the sensors transmission rate every second the sensor is asleep 
                #for _ in range(self._recharge_time):
                #    time.sleep(self.observation_time)
                #    self.rates_log[sensor_id].append(self._rates[sensor_id])

                # Log the sensors throughput and energy when it was asleep (0 KiB/s)
                # TODO: How should energy increase while the sensor is asleep?
                #for _ in range(self._recharge_time - 1):
                #    self.throughput_log[sensor_id].append(0.0)
                #    self.energy_log[sensor_id].append(energy)
    
                #self.throughput_log[sensor_id].append(0.0)

                # Update the sensors energy and log the new energy level
                self._energy[sensor_id] = self._full_energy
                sensor.energy = energy
                #self.energy_log[sensor_id].append(energy)

                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy Recharged to full energy ({self._full_energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.\n')

            # Lock the transmission rate before transmitting the next chunck 
            self._update_rates.acquire()

            # Get the current transmission rate
            rate = self._rates[sensor_id]

            # Get the chunk to send
            chunk = chunks[chunks_sent]

            # Store the current time
            transmission_start_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(transmission_start_time))
            ms = int((transmission_start_time - time.time()) * 1000)
            bytes_sent = 0
            packets_to_send = len(chunk) * self._rate_frequencies[rate] * self._observation_time // max(self._rate_frequencies)

            # Send the current chunck if the frequency rate is > 0 
            if self._rate_frequencies[self._rates[sensor_id]] > 0:
                sensor.cmd(f'netstat -i > {self._log_directory}/sensor_{sensor_id}_netstat.txt')
                for packet in chunk[:packets_to_send]:
                    bytes_sent += len(packet)
                    transmit_time_start = time.time()
                    sensor.cmd(f'echo "{packet}" | nc -w 10 -v -q 1 -w0 -u {ch_ip} {port} 2>{self._log_directory}/sensor_{sensor_id}_log.txt &')
                    transmit_time = time.time() - transmit_time_start
                    if transmit_time < self._observation_time / packets_to_send:
                        time.sleep((self._observation_time / packets_to_send) - transmit_time)
                self._energy[sensor_id] -= 4 * self._rate_frequencies[rate] / max(self._rate_frequencies) 
                info(f"Sensor {sensor_id}: Sent chunk {chunks_sent} of size {(len(packet) / 1024):.2f} KB at {timestamp}.{ms:03d}\n")
            else:
                self._energy[sensor_id] -= 0.3
                info(f"Sensor {sensor_id}: Skipped sending chunk {chunks_sent} due to rate 0 at {timestamp}.{ms:03d}\n")
                time.sleep(self._observation_time)

            chunks_sent += 1
            info(f'Sent {bytes_sent/1024} KiB at {self._rate_frequencies[rate]} KiB/s\n')
            transmission_time = time.time() - transmission_start_time
            info(f'Transmission time: {transmission_time}\n')

            # Log the average throughput over the past ~30 seconds
            #throughput_file = f'{self._log_directory}/sensor_{sensor_id}_throughput.txt' 
            #self.throughput_log[sensor_id].append(get_latest_throughput(throughput_file))

            # Log energy 
            #self.energy_log[sensor_id].append(energy)
            
            # Allow the transmission rate to update
            self._update_rates.release()

            # Log the current rate
            #self.rates_log[sensor_id].append(self._rates[sensor_id])
            
        info(f"Sensor {sensor_id}: Finished sending messages\n")

    def _send_obs_to_server(self):
        self._get_agent_obs()
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
        
        while(True):
            self._cluster_head.cmd(f'netstat -i >> {self._log_directory}/netstat_output.txt')
            self._cluster_head.cmd(f'ss -i >> {self._log_directory}/ss_output.txt')
            self._cluster_head.cmd(f'iw dev ch-wlan0 link >> {self._log_directory}/iw.txt 2>> {self._log_directory}/iw_err.txt')
            self._cluster_head.cmd(f'ss -u -a >> {self._log_directory}/drops.txt')

            """
            # Skip observation if not enough data received 
            for i in range(self._num_sensors):
                file_name = f'ch_received_from_sensor_{i}.txt'
                if not os.path.exists(f'{file_dir}/{file_name}'):
                   time.sleep(self._observation_time) 
                with open(f'{file_dir}/{file_name}', 'r+') as file:
                    while sum(1 for _ in file) < min_lines:
                       time.sleep(self._observation_time) 
            """

            self._send_obs_to_server()
            time.sleep(self._observation_time)

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

            listen_cmd = f'nc -n -ulkv -p {5001 + i} | pv -f -i 0.5 -F "%a" >> {output_file} 2>{self._log_directory}/sensor_{self._sensor_ids[i]}_throughput.txt &'

            node.cmd(f'echo "{listen_cmd}" | bash 2>> {self._log_directory}/listen_cmd_err.txt')
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
            time.sleep(2) 

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
           
            rates_thread = threading.Thread(target=self._receive_rates_from_server, args=())
            rates_thread.start()

            # Give cluster head time to receive data
            time.sleep(2 * self._observation_time)
            transfer_thread = threading.Thread(target=self._transfer_received_data, args=())
            transfer_thread.start()

            """
            while True:
                print("WRITTING!!!!!!!!!!!\n")
                time.sleep(50)
                with open('figure_data.pkl', 'wb') as file:
                    with open('figure_data.pkl', 'wb') as file:
                        pickle.dump((self.energy_log, self.throughput_log, self.rates_log, self.reward_log, self.similarity_log), file)


            time.sleep(320)
            print(f'Saving plots\n')
            with open('figure_data.pkl', 'wb') as file:
                pickle.dump((self.energy_log, self.throughput_log, self.rates_log, self.reward_log, self.similarity_log), file)
            """



            #print("Sufficient data received. Starting RL agent.")
            #info("*** Starting RL agent\n")
           
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
                pickle.dumps((self._sensor_ids, self._rate_frequencies, self.rate_log, self.energy_log, self.throughput_log, self.reward_log, self.similarity_log), file)

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
    port = 5002
    sensor_ids = range(2, 12)
    cluster = sensor_cluster(1, sensor_ids, server_ip, port, packet_size=2*1024)
    cluster.start()
