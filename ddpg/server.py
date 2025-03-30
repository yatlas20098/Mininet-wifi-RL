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
    def _establish_connection(self, cluster_head_ip, port):
        try:
            self._cluster_head = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cluster_head.settimeout(100)
            print("Socket successfully created")
        except socket.error as err:
            print(f'Socket creation failed with err {err}')
     
        self._cluster_head.connect((cluster_head_ip, cluster_port))
        """
        connected = False
        while not connected:
            try:
                self._cluster_head.connect((cluster_head_ip, cluster_port))
                connected = True
            except:
                pass
        """
        print("Successfully connected to server")

    
    def _get_observation(self):
        # Request observation
        self._cluster_head.sendall("Get OBS".encode())

        # Get the size of the observation in bytes 
        obs_size_bytes = self._cluster_head.recv(4)
        obs_size = struct.unpack('!I', obs_size_bytes)[0]

        obs = b''
        while len(obs) < obs_size:
            chunk = self._cluster_head.recv(4096)
            if not chunk:
                continue
            obs += chunk
 
        similarity, throughputs, throughput_reward, clique_reward, rates = pickle.loads(obs)
        print(f'Observation received:')

        # Dump observation for RL agent
        #with open('obs.pkl', 'wb') as file:
        #    pickle.dump((similarity, throughputs, throughput_reward, clique_reward, rates), file)
    
    def __init__(self, cluster_head_ip, port):
        self._establish_connection(cluster_head_ip, port)

# UIUC Apartement router
cluster_head_ip = "173.230.117.220"

# UIUC campus  
#cluster_head_ip = "10.195.111.177"

#cluster_head_ip = 192.168.0.162
# cluster_head_ip = "10.251.162.82"
cluster_port = 5000

#s = server(10, cluster_head_ip, cluster_port)
print("Successfully connected to server")

#s.connect((cluster_head_ip, cluster_port))

"""
#try:
#    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    print("Socket successfully created")
#except socket.error as err:
    print(f'Socket creation failed with err {err}')

s.connect((cluster_head_ip, cluster_port))
print("Successfully connected to server")

print("Waiting for data")

data = s.recv(4096).decode()
while(True):
    if len(data) == 0:
        data = s.recv(4096).decode()

    # Get index for server id 
    server_id_end = data.find('\r\n')
    print(f'Server id: {data[:server_id_end]}')
    server_id = int(data[:server_id_end])

    with open(f'{log_directory}/sensor_{server_id}.txt', 'w+') as file:
        file.write(data[server_id_end+2:])
        data = s.recv(4096).decode()
        print(f'Reading data from sensor_{server_id}')

        file_end = data.find('\r\n\r\n')
        while file_end == -1:
            file.write(data)
            data = s.recv(4096).decode()
            file_end = data.find('\r\n\r\n')

        file.write(data[:file_end]);
        data = data[file_end+4:]

        print(f'Done Reading data from sensor_{server_id}')
        print(f'Data: {data}')
"""
