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
import queue

from redundancy_graph import compute_similarity_and_redundancy_graph
from cluster import Cluster_Handler

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

class Network():
    def get_observation(self, actions):
        result_queue = queue.Queue()
        threads = []

        for cluster_handler, action in zip(self._cluster_handlers, actions):
            thread = threading.Thread(
                target=lambda h, a: result_queue.put(h.get_observation(a)),
                args=(cluster_handler, action)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        observations = []
        while not result_queue.empty():
            observations.append(result_queue.get())
        
        print(observations)
        return observations
    
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
    
    def _create_cluster(self, cluster_idx):
        channels = [6]*10
        # Convert cluster idx to 6 digit hex
        cluster_idx_hex = '{:06x}'.format(cluster_idx)
        
        # Devote first half of mac address to cluster idx
        mac_prefix = ':'.join([cluster_idx_hex[i:i+2] for i in range(0,6,2)])

        # create accesspoint
        ap = self._net.addAccessPoint(f'ap{cluster_idx}', 
                                 ssid=f'ssid-ap{cluster_idx}', 
                                 mac=f'{mac_prefix}:FF:FF:FF',
                                 mode='g', 
                                 channel=f'{channels[cluster_idx]}', 
                                 position=f'{50 + (cluster_idx*300)},50,0')
       
        #create cluster head
        cluster_head = self._net.addStation(f'ch{cluster_idx}', 
                                            ip=f'192.168.{cluster_idx}.100/24',
                                            mac=f'{mac_prefix}:EF:FF:FF',
                                            range='150', 
                                            position=f'{70 + (cluster_idx*300)},70,0')

        #create sensors 
        sensors = []
        for i in range(self._num_sensors):
            idx_hex = '{:06x}'.format(i + 1)
            idx_mac = ':'.join([idx_hex[i:i+2] for i in range(0,6,2)])

            sensor_mac = f'{mac_prefix}:{idx_mac}'
            ip_address = f'192.168.{cluster_idx}.{i + 2}/24'
            sensors.append(self._net.addStation(f's-{cluster_idx}-{i}', 
                                        ip=ip_address,
                                        mac=sensor_mac,
                                        range='116', 
                                        position=f'{30 + i + (cluster_idx*300)},30,0'))
            # cluster = Cluster(cluster_head, sensors, cluster_idx, self._config, log_directory=f'data/log')
        return cluster_head, sensors, ap  

    """
    Create Mininet topology
    """
    def _create_topology(self):
        # build network
        self._net = Mininet_wifi(controller=RemoteController, link=wmediumd, wmediumd_mode=interference)

        info("*** Creating clusters\n")
        self._cluster_heads = []
        self._cluster_sensors = []
        aps = []
        
        for cluster_idx in range(self._config.num_clusters):
            cluster_head, cluster_sensors, cluster_ap = self._create_cluster(cluster_idx)
            self._cluster_heads.append(cluster_head)
            self._cluster_sensors.append(cluster_sensors)
            aps.append(cluster_ap)
                
        info("*** Adding Controller\n")
        c0 = self._net.addController(f'c0', controller=RemoteController, ip="0.0.0.0", port=6653)

       #  self._net.setPropagationModel(model="logDistance", exp=5)

        # self._net.plotGraph(min_x=-800, min_y=-800, max_x=800, max_y=800) 
        info("*** Configuring wifi nodes\n")
        self._net.configureWifiNodes()

        self._net.build()
        
        self._net.start()
        for cluster in self._cluster_sensors:
            for sensor in cluster:
                iface = sensor.params['wlan'][0]
                sensor.cmd(f'iw dev {iface} set retry short 0 2> cmderr.txt')


        for cluster_idx in range(self._config.num_clusters):
            aps[cluster_idx].start([c0])

            for sensor in self._cluster_sensors[cluster_idx]:
                sensor.setAssociation(aps[cluster_idx])
            self._cluster_heads[cluster_idx].setAssociation(aps[cluster_idx])


    def __init__(self, simulation_config, log_directory='data/log'):
        self._config = simulation_config
        self._num_sensors = len(simulation_config.sensor_ids)
        
        # self._num_clusters = 8

        # Delete the log folder if already it exists
        subprocess.run(["rm", "-rf", log_directory])

        # Create log directories 
        os.makedirs(log_directory)

        self._create_topology()

            
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
        cluster_processes = []
        self._cluster_handlers = []
        """
        #for cluster_idx in range(self._config.num_clusters):
        #    cluster_handler = Cluster_Handler(self._cluster_heads[cluster_idx],
                                              self._cluster_sensors[cluster_idx], 
                                              cluster_idx, 
                                              self._config, log_directory=f'data/log')
            
        #    self._cluster_handlers.append(cluster_handler)
        #    cluster_process = multiprocessing.Process(target=cluster_handler.start, args=())
        #    cluster_process.start()
        #    cluster_processes.append(cluster_process)
        """

        for cluster_idx in range(self._config.num_clusters):
            cluster_handler = Cluster_Handler(self._cluster_heads[cluster_idx],
                                              self._cluster_sensors[cluster_idx], 
                                              cluster_idx, 
                                              self._config, log_directory=f'data/log')
            
            self._cluster_handlers.append(cluster_handler)
            cluster_process = threading.Thread(target=cluster_handler.start, args=())
            cluster_process.start()
            cluster_processes.append(cluster_process)


        #while True:
            #self.get_observation([[400]*11] * self._config.num_clusters)

        for cluster_process in cluster_processes:
            cluster_process.join()
    
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
    observation_time = 0.5
    transmission_size = 1*1024
    transmission_frame_duration = 1
    file_lines_per_chunk = 1
    num_transmission_frames = 3000
    num_clusters = 1
    sim_config = Mininet_Simulation_Config(sensor_ids=sensor_ids, observation_time=observation_time, transmission_size=transmission_size, transmission_frame_duration=transmission_frame_duration, file_lines_per_chunk=file_lines_per_chunk, num_transmission_frames=num_transmission_frames, num_clusters=num_clusters)

    network = Network(sim_config)
    network_thread = threading.Thread(target=network.start, args=())
    network_thread.start()

    network_thread.join()
