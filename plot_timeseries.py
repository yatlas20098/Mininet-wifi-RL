# # import re
# # from datetime import datetime, timedelta
# # from collections import defaultdict
# # import matplotlib.pyplot as plt

# # def parse_tcpdump_output(file_path):
# #     udp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d+)\sIP\s(\d+\.\d+\.\d+\.\d+)\.(\d+)\s>\s(\d+\.\d+\.\d+\.\d+)\.(\d+):\sUDP,\slength\s(\d+)')
# #     sensor_packets = defaultdict(list)

# #     with open(file_path, 'r') as file:
# #         for line in file:
# #             match = udp_pattern.search(line)
# #             if match:
# #                 time_str = match.group(1)
# #                 src_ip = match.group(2)
# #                 packet_size = int(match.group(6))
# #                 timestamp = datetime.strptime(time_str, '%H:%M:%S.%f')
# #                 sensor_packets[src_ip].append((timestamp, packet_size))

# #     return sensor_packets

# # def aggregate_throughput(packets, interval=1):
# #     if not packets:
# #         return []

# #     packets.sort(key=lambda x: x[0])
# #     start_time = packets[0][0]
# #     end_time = packets[-1][0]
# #     current_time = start_time
# #     throughput_data = []

# #     while current_time <= end_time:
# #         next_time = current_time + timedelta(seconds=interval)
# #         interval_packets = [p for p in packets if current_time <= p[0] < next_time]
# #         total_data = sum(p[1] for p in interval_packets) * 8  # Convert to bits
# #         throughput = total_data / interval  # bits per second
# #         throughput_data.append((current_time, throughput / 1e6))  # Convert to Mbps
# #         current_time = next_time

# #     return throughput_data

# # def plot_throughput(sensor_data, sensor_ip, output_dir):
# #     plt.figure(figsize=(12, 6))

# #     if sensor_ip in sensor_data:
# #         times, throughputs = zip(*sensor_data[sensor_ip])
# #         plt.plot(times, throughputs, label=sensor_ip)

# #     plt.xlabel('Time')
# #     plt.ylabel('Throughput (Mbps)')
# #     plt.title(f'Throughput Over Time for Sensor {sensor_ip}')
# #     plt.legend()
# #     plt.grid(True)
    
# #     output_path = f"{output_dir}/throughput_plot_{sensor_ip.replace('.', '_')}.png"
# #     plt.savefig(output_path)
# #     plt.close()
# #     print(f"Plot saved to {output_path}")

# # def main():
# #     input_file = '/mydata/actuallyuse/size_5000_2_per_s/extracted_data/tcpdump_output_capture.txt'
# #     output_dir = '/mydata/actuallyuse/size_5000_2_per_s'
# #     target_sensor_ip = '192.168.0.6'

# #     sensor_packets = parse_tcpdump_output(input_file)
# #     sensor_throughput = {}

# #     for sensor_ip, packets in sensor_packets.items():
# #         throughput_data = aggregate_throughput(packets)
# #         sensor_throughput[sensor_ip] = throughput_data

# #     plot_throughput(sensor_throughput, target_sensor_ip, output_dir)

# # if __name__ == "__main__":
# #     main()
# import re
# from datetime import datetime, timedelta
# from collections import defaultdict
# import matplotlib.pyplot as plt

# def parse_tcpdump_output(file_path):
#     udp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d+)\sIP\s(\d+\.\d+\.\d+\.\d+)\.(\d+)\s>\s(\d+\.\d+\.\d+\.\d+)\.(\d+):\sUDP,\slength\s(\d+)')
#     sensor_packets = defaultdict(list)

#     with open(file_path, 'r') as file:
#         for line in file:
#             match = udp_pattern.search(line)
#             if match:
#                 time_str = match.group(1)
#                 src_ip = match.group(2)
#                 packet_size = int(match.group(6))
#                 timestamp = datetime.strptime(time_str, '%H:%M:%S.%f')
#                 sensor_packets[src_ip].append((timestamp, packet_size))

#     return sensor_packets

# def aggregate_throughput(packets, interval=1):
#     if not packets:
#         return []

#     packets.sort(key=lambda x: x[0])
#     start_time = packets[0][0]
#     end_time = packets[-1][0]
#     current_time = start_time
#     throughput_data = []

#     while current_time <= end_time:
#         next_time = current_time + timedelta(seconds=interval)
#         interval_packets = [p for p in packets if current_time <= p[0] < next_time]
#         total_data = sum(p[1] for p in interval_packets) * 8  # Convert to bits
#         throughput = total_data / interval  # bits per second
#         throughput_data.append((current_time, throughput / 1e6))  # Convert to Mbps
#         current_time = next_time

#     return throughput_data

# def plot_throughput(sensor_data, output_dir):
#     # Plot for all sensors
#     plt.figure(figsize=(12, 6))
#     for sensor_ip, data in sensor_data.items():
#         times, throughputs = zip(*data)
#         plt.plot(times, throughputs, label=sensor_ip)
#     plt.xlabel('Time')
#     plt.ylabel('Throughput (Mbps)')
#     plt.title('Throughput Over Time for All Sensors')
#     plt.legend()
#     plt.grid(True)
#     output_path = f"{output_dir}/throughput_plot_all_sensors.png"
#     plt.savefig(output_path)
#     plt.close()
#     print(f"Plot for all sensors saved to {output_path}")

#     # Plot for each sensor
#     for sensor_ip, data in sensor_data.items():
#         plt.figure(figsize=(12, 6))
#         times, throughputs = zip(*data)
#         plt.plot(times, throughputs, label=sensor_ip)
#         plt.xlabel('Time')
#         plt.ylabel('Throughput (Mbps)')
#         plt.title(f'Throughput Over Time for Sensor {sensor_ip}')
#         plt.legend()
#         plt.grid(True)
#         output_path = f"{output_dir}/throughput_plot_{sensor_ip.replace('.', '_')}.png"
#         plt.savefig(output_path)
#         plt.close()
#         print(f"Plot for sensor {sensor_ip} saved to {output_path}")

# def main():
#     input_file = '/mydata/output/extracted_data/tcpdump_output_capture.txt'
#     output_dir = '/mydata/output/extracted_data/'

#     sensor_packets = parse_tcpdump_output(input_file)
#     sensor_throughput = {}

#     for sensor_ip, packets in sensor_packets.items():
#         throughput_data = aggregate_throughput(packets)
#         sensor_throughput[sensor_ip] = throughput_data

#     plot_throughput(sensor_throughput, output_dir)

# if __name__ == "__main__":
#     main()
import re
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import time

def parse_tcpdump_output(file_path):
    print("Starting to parse tcpdump output...")
    start_time = time.time()
    udp_pattern = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d+)\sIP\s(\d+\.\d+\.\d+\.\d+)\.(\d+)\s>\s(\d+\.\d+\.\d+\.\d+)\.(\d+):\sUDP,\slength\s(\d+)')
    sensor_packets = defaultdict(list)

    with open(file_path, 'r') as file:
        lines = file.readlines()
        total_lines = len(lines)
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

def aggregate_throughput(packets, interval=1):
    print("Aggregating throughput...")
    start_time = time.time()
    if not packets:
        return []

    packets.sort(key=lambda x: x[0])
    start_time_packet = packets[0][0]
    end_time_packet = packets[-1][0]
    current_time = start_time_packet
    throughput_data = []

    total_intervals = int((end_time_packet - start_time_packet).total_seconds() / interval)
    processed_intervals = 0

    while current_time <= end_time_packet:
        next_time = current_time + timedelta(seconds=interval)
        interval_packets = [p for p in packets if current_time <= p[0] < next_time]
        total_data = sum(p[1] for p in interval_packets) * 8  # Convert to bits
        throughput = total_data / interval  # bits per second
        throughput_data.append((current_time, throughput / 1e6))  # Convert to Mbps
        current_time = next_time

        processed_intervals += 1
        if processed_intervals % 100 == 0:
            print(f"Aggregation progress: {processed_intervals}/{total_intervals} intervals ({processed_intervals/total_intervals*100:.2f}%)")

    print(f"Aggregation completed in {time.time() - start_time:.2f} seconds")
    return throughput_data

def plot_throughput(sensor_data, output_dir):
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

def main():
    input_file = '/mydata/mydata/RL_agent/output/extracted_data/tcpdump_output_capture.txt'
    output_dir = '/mydata/mydata/RL_agent/output/extracted_data'

    overall_start_time = time.time()

    sensor_packets = parse_tcpdump_output(input_file)
    sensor_throughput = {}

    print(f"Processing throughput for {len(sensor_packets)} sensors...")
    for i, (sensor_ip, packets) in enumerate(sensor_packets.items()):
        print(f"Processing sensor {i+1}/{len(sensor_packets)}: {sensor_ip}")
        throughput_data = aggregate_throughput(packets)
        sensor_throughput[sensor_ip] = throughput_data

    plot_throughput(sensor_throughput, output_dir)

    print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()