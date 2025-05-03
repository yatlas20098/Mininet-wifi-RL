import pickle
import numpy as np
import matplotlib.pyplot as plt

def create_multiplot(datasets, title, xlabel, ylabel, legend_labels, output_name):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))

    for label, data in zip(legend_labels, datasets):
        print(f'\tPlotting {label}')
        plt.plot(range(len(data)), data, linestyle='-', label=label, fillstyle='none')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)
    plt.close()
    print(f"Done plotting\n")

def create_plot(data, title, xlabel, ylabel, output_name, average=0, start_step=0):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))
    plt.plot(list(range(start_step, start_step + len(data))), data)
    # Plot line for averages
    if average > 0:
        averages = [np.mean(data[i-100:i]) for i in range(average, len(data))]
        plt.plot(range(average, len(data)), averages, label=f'Average over Past {average} Steps')
        m, b = np.polyfit(range(len(data)), data, 1) # 1 indicates linear fit
        plt.plot(range(len(data)), m * range(len(data)) + b, label=f'Line of Best Fit')
        plt.legend()
        
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    output_path = f"graphics/{output_name}.png"
    plt.savefig(output_path, dpi=500)
    plt.close()
    print(f"Done plotting\n")

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

        return np.array(data)[::30][:2500]
datasets = []
labels = []

# Load sensor temperature data
for sensor_id in range(5,15):
    try:
        file_path = f'data/towerdataset/tower{sensor_id}Data_processed.csv'
        datasets.append(read_temperature_data(file_path))
        labels.append(f'S{sensor_id}')
        #create_plot(tempdata, f'Sensor {sensor_id} Temperature', 'Step', 'Temperature', f'temp/sensor{sensor_id}temp')
    except:
        print("error")

min_data_len = min((len(data) for data in datasets))
datasets = [data[:min_data_len] for data in datasets] 

# Plot sensor temperature data
create_multiplot(datasets, 'Sensor Temperature over Steps', 'Step', 'Temperature', labels, f'temp/sensortemps')
#####################################################################
# Load simulation data
file_path = 'figure_data.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    sensor_ids, rate_frequencies, rate_log, _, throughput_log, reward_log, similarity_reward_log, throughput_reward_log, max_ind_set_log, chunks_sent_log = data


# Plot throughput and chunks lost
throughput_log = np.array(throughput_log).T
throughput = [np.sum(t) for t in throughput_log]
chunks_sent = [np.sum(c) for c in chunks_sent_log]

total_throughput = [throughput[0]]
total_chunks_sent = [chunks_sent[0]]

for i in range(1, len(throughput)):
    total_throughput.append(throughput[i] + total_throughput[i-1])
    total_chunks_sent.append(chunks_sent[i] + total_chunks_sent[i-1])

chunks_lost = np.array(total_chunks_sent) - np.array(total_throughput)
create_plot(chunks_lost, 'Cluster Head Transmissions Lost', 'Step', 'Transmissions Lost (2 Packets)', 'packet_loss')
create_plot(throughput, 'Cluster Head Throughput over Steps', 'Step', 'Succesfull Transmissions (2 Packets)', 'throughput', average=100)

# Plot total MIS Throughputs
total_ind_set_throughputs = []
for t in range(len(max_ind_set_log)):
    sensor_throughputs = throughput_log[t]
    total_throughput = 0
    for sensor in max_ind_set_log[t]:
        total_throughput = total_throughput + sensor_throughputs[sensor] 
    total_ind_set_throughputs.append(total_throughput)

create_plot(total_ind_set_throughputs, 'Total. MIS Throughput over Steps', 'Step', 'Throughput (2 Packets)', 'total_mis_throughput', average=100)

# Plot loss
with open('loss.pkl', 'rb') as file:
    batch_size, loss = pickle.load(file) 
throughput_loss = np.array(loss['throughput'])
throughput_loss = np.reshape(throughput_loss, (-1, 10))
throughput_average_loss = np.mean(throughput_loss, axis=1)
create_plot(throughput_average_loss, 'Throughput Trainning Loss', 'Step', 'Loss', 'throughput_loss', average=100, start_step=batch_size)

similarity_loss = np.array(loss['similarity'])
similarity_loss = np.reshape(similarity_loss, (-1, 10))
similarity_average_loss = np.mean(similarity_loss, axis=1)
create_plot(similarity_average_loss, 'Similarity Trainning Loss', 'Step', 'Loss', 'similarity_loss', average=100, start_step=batch_size)

# Plot rewards
throughput_rewards = np.array(throughput_reward_log).reshape(-1, 10)
average_throughput_rewards = np.mean(throughput_rewards, axis=1)
np.clip(average_throughput_rewards, -2, 0, out=average_throughput_rewards) # Clip for better readability
create_plot(average_throughput_rewards, 'Throughput Rewards', 'Step', 'Reward', 'throughput_reward', average=100)

similarity_rewards = np.array(similarity_reward_log).reshape(-1, 10)
average_similarity_rewards = np.mean(similarity_rewards, axis=1)
np.clip(average_similarity_rewards, -1, 0, out=average_similarity_rewards) # Clip for better readability 
create_plot(average_similarity_rewards, 'Average Similarity Rewards', 'Step', 'Reward', 'similarity_reward', average=100)


