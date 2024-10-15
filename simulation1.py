import sys
import socket
import os
import time
import threading
import random
import multiprocessing
import json
import subprocess
from tqdm import tqdm
from mininet.node import Controller
from mininet.log import setLogLevel, info
from mn_wifi.net import Mininet_wifi
from mn_wifi.cli import CLI
from mn_wifi.link import wmediumd
from mn_wifi.wmediumdConnector import interference
import numpy as np

class sensor_cluster():
    """
    Check if the cluster head received at least min_lines lines from every sensor.

    Args:
        base_output_file (string): base of file path
        min_lines (int): min number of lines that the cluster head must recieve

    Returns:
        bool: True if the cluster head received at least min_lines from every sensor, false otherwise
    """
    """
    def _check_received_data(self, base_output_file, min_lines=10):
        #To run the RL agent, the cluster head must recieve at least 10 lines.
        for i in range(self._num_sensors):
            file_path = f'{base_output_file}_{i}.txt'
            if not os.path.exists(file_path):
                return False
            with open(file_path, 'r') as file:
                l = sum(1 for _ in file)
                print(f"File length: {l}") 
                #if sum(1 for _ in file) < min_lines:
                if l < min_lines:
                    return False
        return True
    
    """
    """
    Split dataset into chunks

    Args:
        dataset_dir (string): Path to dataset file

    Returns:
        String List: List of chunks 
    """
    def _preprocess_dataset_into_chunks(self, dataset_dir):
        # cache the dataset into memory first
        with open(dataset_dir, 'r') as file:
            lines = file.readlines()
    
        return [lines[i:i + self._chunk_size] for i in range(0, len(lines), self._chunk_size)]
    

    """
    Create Mininet topology
    """
    def _create_topology(self):
            #build network
            self._net = Mininet_wifi(controller=Controller, link=wmediumd,
                               wmediumd_mode=interference)

            info("*** Creating nodes\n")
            #create accesspoint
            ap23 = self._net.addAccessPoint('ap23', ssid='new-ssid', mode='g', channel='5', position='50,50,0')
            
            #create 10 sensors
            self._sensors = []
            for i in range(self._num_sensors):
                ip_address = f'192.168.0.{i + 1}/24'
                self._sensors.append(self._net.addStation(f's{i}', ip=ip_address,
                                              range='116', position=f'{30 + i},30,0'))
            #create cluster head
            self._cluster_head = self._net.addStation('ch', ip='192.168.0.100/24',
                                          range='150', position='70,70,0')

            info("*** Adding Controller\n")
            self._net.addController('c0')

            info("*** Configuring wifi nodes\n")
            self._net.configureWifiNodes()

    """
    Creates a socket between the server and cluster head. To avoid networking complications with Hal (i.e., a firewall), the server technically connects to the cluster head.   
    """
    def _establish_server_connection(self):
        # TODO: Create UDP socket for sending message to server and TCP socket for receving messages from server 
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket succesfully created")
        listen_port = 5000
        listen.bind(("0.0.0.0", port))
        print(f'Socket binded to {port}')

        listen.listen(5)
        print("Socket is listening")

        server, addr = listen.accept()
        print('Got connection from', addr)

        self._server = server

    """
    def _establish_server_connection(self):
        listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket succesfully created")
        listen_port = 5000
        listen.bind(("0.0.0.0", port))
        print(f'Socket binded to {port}')

        listen.listen(5)
        print("Socket is listening")

        c, addr = listen.accept()
        print('Got connection from', addr)

        self._server_listen = c 

        send_port, _ = c.recvfrom(4096)
        send_port = int(send_port.decode())
        send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._server_send = send
        self._server_addr = (addr[0], send_port)
        print(f'Server addr: {self._server_addr}')
    """


    def __init__(self, server_ip, server_port, num_sensors=10, chunk_size=5000, log_directory='data/log', dataset_directory='data/towerdataset'):
        self._server_ip = server_ip
        self._server_port = server_port
        self._num_sensors = num_sensors
        self._chunk_size = chunk_size
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory

        # Delete log folder if it exists
        subprocess.run(["rm", "-rf", log_directory])
        os.makedirs(log_directory)

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        """
        Instantiate rate files with default rates (2)

        Files are used to represent changes in the rate assigned by the cluster head to circumvent a limiation of thread control in Mininet-Wifi
        """
        for sensor_id in range(self._num_sensors):
            rate_file = f'{self._log_directory}/sensor_{sensor_id}_rate.txt'
            with open(rate_file, 'w') as file:
                file.write("2")

        # Initalize the dataset for each sensor 
        self._datasets = {}
        for i in range(self._num_sensors):
            tower_number = i + 2  # Start from tower2 to tower11
            file_path = f'{dataset_directory}/tower{tower_number}Data_processed.csv'
            if os.path.exists(file_path):
                self._datasets[i] = self._preprocess_dataset_into_chunks(file_path)
            else:
                print(f"Warning: Dataset file not found for sensor {i}: {file_path}")
                self._datasets[i] = []
        
        # Clear log
        base_output_file = f'{self._log_directory}/ch_received_from_sensor'
        for i in range(self._num_sensors):
            output_file = f'{base_output_file}_{i}.txt'
            with open(output_file, 'w') as file:
                file.write("")
         
        self._create_topology()
        self._establish_server_connection()
       
    """
    Send message from a sensor to the cluster head

    Args:
        sensor (station): sensor to send message from
        ch_ip (string): cluster head ip
        sensor_id (int): id of sensor
    """
    def _send_messages_to_cluster_head(self, sensor, ch_ip, sensor_id):
        chunks = self._datasets[sensor_id]
        info(f"Chunk size: {len(chunks)}\n")
        rate_file = f'{self._log_directory}/sensor_{sensor_id}_rate.txt'
        if not chunks:
            info(f"Sensor {sensor_id}: No data available. Skipping send_messages.\n")
            return
        
        packetnumber = 0
        port = 5001 + sensor_id
        rate = 2  # Initial rate
        sensor.cmd(f'touch {self._log_directory}/sensor_{sensor_id}_log.txt')
        
        for chunk in chunks:
            packet_data = ''.join(chunk)
            packet_size_kb = len(packet_data) / 1024.0
            
            # read updated rate from file 
            with open(rate_file, 'r') as file:
                rate = float(file.read().strip())
            
            current_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            ms = int((current_time - int(current_time)) * 1000)
            
            if rate > 0:
                # -q 1 Not working on laptop; using -q 1 -w0
                sensor.cmd(f'echo "{packet_data}" | nc -v -q 1 -w0 -u {ch_ip} {port} 2>{self._log_directory}/sensor_{sensor_id}_log.txt')
                info(f"Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
            else:
                info(f"Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
            
            packetnumber += 1
            if packetnumber == 300:
                break
               
            # TODO: Switch to amount of data sent instead of rate?
            if rate > 0:
                time.sleep(1.0 / rate)
            else:
                time.sleep(1) 

        info(f"Sensor {sensor_id}: Finished sending messages\n")

    def _send_file_to_server(self, file_dir, file_name, sensor_id):
        print(f'Sending file {file_name} to server')

        # Create copy of file
        subprocess.run(["cp", f'{file_dir}/{file_name}', f'{file_dir}/.{file_name}'])

        # TODO: Lock file before clearing
        # Clear file 
        with open(f'{file_dir}\{file_name}', 'w') as file:
            file.write("")
        
        # TODO: Send using UDP instead of TCP
        with open(f'{file_dir}/.{file_name}', 'r') as file:
            self._server.send(f'{sensor_id}'.encode())
            for line in file:
                print(f'message: {sensor_id}{line}')
                self._server.send(line.encode())
            self._server.send('DONE'.encode())

        print(f'File sent')
   
    def _transfer_received_data(self, min_lines=10):
        # To update RL State, min_lines new lines must have been recieved
        file_dir = self._log_directory
        while(True):
            for i in range(self._num_sensors):
                file_name = f'ch_received_from_sensor_{i}.txt'
                if not os.path.exists(file_dir):
                    time.sleep(1)
                    break
                l = 0
                with open(f'{file_dir}/{file_name}', 'r') as file:
                    l = sum(1 for _ in file)
                    print(f'{file_name} has {l} lines')
                if l > min_lines:
                    self._send_file_to_server(file_dir, file_name, i)
            time.sleep(0.5)

    """
    Start background netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _receive_messages(self, node):
        base_output_file = f'{self._log_directory}/ch_received_from_sensor'

        for i in range(self._num_sensors):
            #save the data that receive from different sensor to different file
            output_file = f'{base_output_file}_{i}.txt'
            node.cmd(f'touch {output_file}')
            node.cmd(f'nc -n -ulkv -p {5001 + i} >> {output_file} 2>{self._log_directory}/ch_nc_err_log &')
            info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")
        #capture the network by pcap
        pcap_file = f'{self._log_directory}/capture.pcap'
        node.cmd(f'tcpdump -i {node.defaultIntf().name} -n udp portrange 5001-{5001+self._num_sensors-1} -w {pcap_file} &')
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
            
            # Activate cluster head listining threads
            receive_thread = threading.Thread(target=self._receive_messages, args=(self._cluster_head,))
            #receive_thread = multiprocessing.Process(target=self._receive_messages, args=(self._cluster_head,))

            receive_thread.start()
            
            # Give listining threads time to start
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
                
            #print("Waiting for initial data before starting RL agent")
            #while not self._check_received_data(f'{self._log_directory}/ch_received_from_sensor'):
            #    time.sleep(5)

            transfer_thread = threading.Thread(target=self._transfer_received_data, args=())
            transfer_thread.start()

            print("Sufficient data received. Starting RL agent.")
            info("*** Starting RL agent\n")

            #rl_thread = threading.Thread(target=rl_agent_process, args=(env, agent, sensors, cluster_head))
            #rl_thread.start()   

            for thread in sender_threads:
                thread.join()

            self._stop_receivers(cluster_head)
            receive_thread.join()
            rl_thread.join()
            for thread in listener_threads:
                thread.join()
        
            for sensor in sensors:
                sensor.cmd('pkill tcpdump')
            cluster_head.cmd('pkill nc')
            
        except Exception as e:
            info(f"*** Error occurred during communication: {str(e)}\n")
            
        info("*** Running CLI\n")
        CLI(self._net)

        info("*** Stopping network\n")
        self._net.stop()

        self._server.close()

if __name__ == '__main__':
    setLogLevel('info')
    server_ip = '192.168.20.201'
    port = 5000
    cluster = sensor_cluster(server_ip, port, chunk_size=250, num_sensors=4)
    cluster.start()
