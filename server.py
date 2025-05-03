import socket
import time
from multiprocessing import Process, Value
import threading 
import os
import struct
import pickle

log_directory = 'log'

if not os.path.exists(log_directory):
    os.makedirs(log_directory)

class mininet_server:
    def set_rates(self, new_rates):
        # Send rates to cluster head
        packed_data = struct.pack('!' + 'i'*self.num_sensors, *action)
        self._cluster_head.send(packed_data)

    def _establish_connection(self, cluster_head_ip, port):
        try:
            self._cluster_head = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cluster_head.settimeout(100)
            print("Socket successfully created")
        except socket.error as err:
            print(f'Socket creation failed with err {err}')
     
        self._cluster_head.connect((cluster_head_ip, cluster_port))
        print("Successfully connected to mininet server")
    
    def get_observation(self, new_rates):
        try:
            # Send new rates
            packed_data = struct.pack('!' + 'i'*self.num_sensors, *new_rates)
            self._cluster_head.send(packed_data)

            # Get the size of the observation in bytes 
            obs_size_bytes = self._cluster_head.recv(4)
            obs_size = struct.unpack('!I', obs_size_bytes)[0]

            obs = b''
            while len(obs) < obs_size:
                chunk = self._cluster_head.recv(4096)
                if not chunk:
                    continue
                obs += chunk
     
            return pickle.loads(obs) # convert observation from bytes
        
        except socket.error as err:
            print(f'Error getting observation: {err}')

    def __init__(self, num_sensors, cluster_head_ip, port):
        self.num_sensors = num_sensors
        self._establish_connection(cluster_head_ip, port)
