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
                # Set sensor sending buffer for UDP to 10 MB
                #self._sensors[i].cmd('sysctl -w net.ipv4.udp_wmem_max=10485760')
                #self._sensors[i].cmd('sysctl -w net.ipv4.udp_wmem_default=10485760')

            #create cluster head
            self._cluster_head = self._net.addStation('ch', ip='192.168.0.100/24',
                                          range='150', position='70,70,0')
            # Set cluster head receving buffer for UDP to 10 MB
            #self._cluster_head.cmd(f'sysctl -w net.ipv4.udp_wmem_max=10485760 2>> {self._log_directory}/sysctl.txt')
            #self._cluster_head.cmd(f'sysctl -w net.ipv4.udp_rmem_max=10485760 2>> {self._log_directory}/sysctl.txt')


            info("*** Adding Controller\n")
            self._net.addController('c0')

            info("*** Configuring wifi nodes\n")
            self._net.configureWifiNodes()

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

    def __init__(self, cluster_id, server_ip, server_port, num_sensors=10, chunk_size=20, log_directory='data/log', dataset_directory='data/towerdataset'):
        self._cluster_id = cluster_id
        self._server_ip = server_ip
        self._server_port = server_port
        self._num_sensors = num_sensors
        self._chunk_size = chunk_size
        self._log_directory = log_directory
        self._dataset_directory = dataset_directory
        
        self._rates_id = 0
        self._rates = [2 for _ in range(num_sensors)]
        self._receiving_data = threading.Semaphore(1)
        self._bytes_received = [0 for _ in range(num_sensors)]
        self._full_energy = 100
        self._recharge_time = 3
        self._recharge_threshold = 20
        self._throughput_data = {}
        self._tcpdump_lines_parsed = 0

        self._input_file = 'data/log/tcpdump_output_capture.txt'
        self._output_dir = 'log_directory/output/extracted_data'


        # Delete log folder if already it exists
        subprocess.run(["rm", "-rf", log_directory])
        os.makedirs(log_directory)

        if not os.path.exists(dataset_directory):
            print("Dataset directory does not exist. Exiting now")
            exit()

        # Files are used to represent changes in the rate assigned by the cluster head and the energy to circumvent a limiation of thread control in Mininet-Wifi
        # Set all rates to their default value (2)
        for sensor_id in range(self._num_sensors):
            rate_file = f'{self._log_directory}/sensor_{sensor_id}_rate.txt'
            with open(rate_file, 'w') as file:
                file.write("2")

        # Set all energies to their default value (100)
        for sensor_id in range(self._num_sensors):
            energy_file = f'{self._log_directory}/sensor_{sensor_id}_energy.txt'
            with open(energy_file, 'w') as file:
                file.write("100")


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
        
        self._create_topology()
        self._establish_server_connection()

    def _receive_rates_from_server(self):
        while(True):
            try:
                packed_data = self._server.recv(4 * (self._num_sensors + 1))
                if len(packed_data) == 0:
                    print("Connection terminated by server")
                    break
                unpacked_data = struct.unpack('!' + 'i'*(self._num_sensors + 1), packed_data)

                # Wait for current data transmission to end
                self._receiving_data.acquire()
                rates = unpacked_data[1:]
                self._receiving_data.release()

                print("Rates has lock")
                self._rates_id = unpacked_data[0]
                print(f'New rate.\nID = {self._rates_id}')
                print("Rates = ", rates)

                for sensor_id in range(self._num_sensors):
                    #rate_file = f'{self._log_directory}/sensor_{sensor_id}_rate.txt'
                    #with open(rate_file, 'w') as file:
                        #file.write(str(rates[sensor_id] + 1))
                    self._rates[sensor_id] = rates[sensor_id] + 1
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
        chunks = self._datasets[sensor_id]
        info(f"Chunk size: {len(chunks)}\n")
        rate_file = f'{self._log_directory}/sensor_{sensor_id}_rate.txt'
        energy_file = f'{self._log_directory}/sensor_{sensor_id}_energy.txt'
        charge_count = 0
        energy = self._full_energy
        
        # Read current energy
        # with open(energy_file, 'r') as file:
        #   energy = float(file.read().strip())

        if not chunks:
            info(f"Sensor {sensor_id}: No data available. Skipping send_messages.\n")
            return
        
        packetnumber = 0
        port = 5001 + sensor_id
        #rate = 2  # Initial rate
        sensor.cmd(f'touch {self._log_directory}/sensor_{sensor_id}_log.txt')
        
        for chunk in chunks:
            if energy < self._recharge_threshold:
                charge_count += 1
                recharge_time = time.time()
                rechar_time_stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(recharge_time))
                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy below threshold ({energy}). Recharging...\n')

                # Read data for recharge time
                time.sleep(self._recharge_time)
                energy = self._full_energy
                info(f'Cluster {self._cluster_id}, Sensor {sensor_id}: Energy Recharged to full energy ({energy}), Current time: {recharge_time}, Charge Count: {charge_count}. Resuming operations.\n')
                sensor.energy = energy

            packet_data = ''.join(chunk)
            packet_size_kb = len(packet_data) / 1024.0
            
            # read updated rate from file 
            # with open(rate_file, 'r') as file:
             #   rate = float(file.read().strip())
            
            current_time = time.time()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            ms = int((current_time - int(current_time)) * 1000)
            
            if self._rates[sensor_id] > 0:
                # Read x Mb 
                sensor.cmd(f'netstat -i > {self._log_directory}/sensor_{sensor_id}_netstat.txt') 

                # sensor.cmd(f'tc qdisc del dev s{sensor_id}-wlan0 root 2>>{self._log_directory}/tc_log.txt') 
                # Limit transmit rate
                sensor.cmd(f'tc qdisc replace dev s{sensor_id}-wlan0 root tbf rate 2mbit burst 32kbit latency 400ms 2>>{self._log_directory}/tc_log.txt') 
                # -q 1 Not working on laptop; using -q 1 -w0
                # sensor.cmd(f'echo "{packet_data}" | nc -w 10 -v -q 1 -w0 -u {ch_ip} {port} 2>{self._log_directory}/sensor_{sensor_id}_log.txt')
                # Transmit data
                sensor.cmd(f'echo "{packet_data}" | pv -q -L 1m | nc -w 10 -v -q 1 -w0 -u {ch_ip} {port} 2>>{self._log_directory}/sensor_{sensor_id}_log.txt')
                energy -= 4
                info(f"Sensor {sensor_id}: Sent packet {packetnumber} of size {packet_size_kb:.2f} KB at {timestamp}.{ms:03d}\n")
            else:
                energy -= 0.3
                info(f"Sensor {sensor_id}: Skipped sending packet {packetnumber} due to rate 0 at {timestamp}.{ms:03d}\n")
            
            packetnumber += 1
            if packetnumber == 3000:
                break
               
            if rate > 0:
                #time.sleep(1.0 / rate)
                time.sleep(1.0)
            else:
                time.sleep(1) 

        info(f"Sensor {sensor_id}: Finished sending messages\n")

    def _send_file_to_server(self, file, file_dir, file_name, sensor_id):
        # Create a copy of the file
        subprocess.run(["cp", f'{file_dir}/{file_name}', f'{file_dir}/.{file_name}'])

        # Clear the file 
        file.truncate(0)
       
        print("Sending files has lock")
        # TODO: Send using UDP instead of TCP
        # Send the copy of the file to the server
        with open(f'{file_dir}/.{file_name}', 'r') as file_cpy:
            self._server.send(f'{self._rates_id}\r\n{sensor_id}\r\n'.encode()) # rates_id is the RL step that sent the current rates 
            for line in file_cpy:
                self._server.send(line.encode())
            self._server.send('\r\n\r\n'.encode())
   
    def _transfer_received_data(self, min_lines=10):
        file_dir = self._log_directory
        data_sent = False
        self._receiving_data.acquire()
        while(True):
            if data_sent:
                # Wait for rate update before starting next transmission 
                self._receiving_data.acquire()
                data_sent = False
            
            print(f'Executing command: tcpdump -r {self._log_directory}/capture.pcap > {self._log_directory}/tcpdump_output_capture.txt')

            # Kill TCPdump
            # Process pcap file
            overall_start_time = time.time()
            self._cluster_head.cmd(f'tcpdump -r {self._log_directory}/capture.pcap > {self._log_directory}/tcpdump_output_capture.txt')
            self._cluster_head.cmd(f'netstat -i >> {self._log_directory}/netstat_output.txt')
            self._cluster_head.cmd(f'ss -i >> {self._log_directory}/ss_output.txt')
            self._cluster_head.cmd(f'iw dev ch-wlan0 link >> {self._log_directory}/iw.txt 2>> {self._log_directory}/iw_err.txt')
            self._cluster_head.cmd(f'ss -u -a >> {self._log_directory}/drops.txt')

            # Start new TCPdump

            #for sensor in range(self._num_sensors):
                #self._cluster_head.cmd(f'tcpdump -r {self._log_directory}/tcpdump_sender_sensor{sensor}.txt > {self._log_directory}/extracted_data/tcpdump_output_sensor{sensor}.txt')

            # Update throughputs 
            sensor_packets = self._parse_tcpdump_output(self._input_file)

            print(f"Processing throughput for {len(sensor_packets)} sensors...")
            for i, (sensor_ip, packets) in enumerate(sensor_packets.items()):
                print(f"Processing sensor {i+1}/{len(sensor_packets)}: {sensor_ip}")
                self._aggregate_throughput(sensor_ip, packets)
                print(f'Sensor {sensor_ip} last throughput: {self._throughput_data[sensor_ip][-1]}')
            print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")

            print(f'Throughput: {self._throughput_data}')

            # Send data to server
            for i in range(self._num_sensors):
                file_name = f'ch_received_from_sensor_{i}.txt'
                if not os.path.exists(file_dir):
                    time.sleep(1)
                    break
                l = 0
                with open(f'{file_dir}/{file_name}', 'r+') as file:
                    if sum(1 for _ in file) > min_lines:
                        self._send_file_to_server(file, file_dir, file_name, i)
                        # Allow rates to be updated
                        self._receiving_data.release()
                        data_sent = True
                    else:
                        time.sleep(1)

    """
    Start background netcat listining process for cluster head

    Args:
        node (station): cluster head 
    """
    def _receive_messages(self, node):
        base_output_file = f'{self._log_directory}/ch_received_from_sensor'

        for i in range(self._num_sensors):
            # save the data received from a sensor to their respective file
            output_file = f'{base_output_file}_{i}.txt'
            node.cmd(f'touch {output_file}')
            node.cmd(f'nc -n -ulkv -p {5001 + i} >> {output_file} 2>{self._log_directory}/ch_nc_err_log &')
            info(f"Receiver: Started listening on port {5001 + i} for sensor {i}\n")

        #capture the network by pcap
        pcap_file = f'{self._log_directory}/capture.pcap'
        node.cmd(f'tcpdump -i {node.defaultIntf().name} -n udp portrange 5001-{5001+self._num_sensors-1} -U -w {pcap_file} &')
        info(f"Receiver: Started tcpdump capture on ports 5001-{5001+self._num_sensors-1}\n")

    """
    def _measure_throughput(self, node):
        while True:
    """


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

            time.sleep(5) 
            transfer_thread = threading.Thread(target=self._transfer_received_data, args=())
            transfer_thread.start()

            print("Sufficient data received. Starting RL agent.")
            info("*** Starting RL agent\n")

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

    def _plot_throughput(self, sensor_data, output_dir):
        print("Plotting throughput...")
        start_time = time.time()

        # Plot for all sensors
        plt.figure(figsize=(12, 6))
        for sensor_ip, data in sensor_data.items():
            times, throughputs = zip(*data)
            plt.plot(times, throughputs, label=sensor_ip)
        plt.xlabel('Time')
        plt.ylabel('Throughput (Mbps)')
        plt.title('Throughput Over Time for All Sensors')
        plt.legend()
        plt.grid(True)
        output_path = f"{output_dir}/throughput_plot_all_sensors.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Plot for all sensors saved to {output_path}")

        # Plot for each sensor
        total_sensors = len(sensor_data)
        for i, (sensor_ip, data) in enumerate(sensor_data.items()):
            plt.figure(figsize=(12, 6))
            times, throughputs = zip(*data)
            plt.plot(times, throughputs, label=sensor_ip)
            plt.xlabel('Time')
            plt.ylabel('Throughput (Mbps)')
            plt.title(f'Throughput Over Time for Sensor {sensor_ip}')
            plt.legend()
            plt.grid(True)
            output_path = f"{output_dir}/throughput_plot_{sensor_ip.replace('.', '_')}.png"
            plt.savefig(output_path)
            plt.close()
            print(f"Plot for sensor {sensor_ip} saved to {output_path} ({i+1}/{total_sensors})")

        print(f"Plotting completed in {time.time() - start_time:.2f} seconds")

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
    cluster = sensor_cluster(1, server_ip, port, chunk_size=5000, num_sensors=10)
    cluster.start()
