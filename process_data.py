import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy

def create_multiplot(datasets, title, xlabel, ylabel, legend_labels, output_name, xs=None):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))

    if xs is None:
        xs = [np.arange(len(data)) for data in datasets]

    for label, data, x in zip(legend_labels, datasets, xs):
        print(f'\tPlotting {label}')
        plt.plot(x, data, linestyle='-', label=label, fillstyle='none')

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

def create_plot(data, title, xlabel, ylabel, output_name, x=None, average=0, start_step=0, clip=-100):
    print(f"Plotting {output_name}...")
    plt.figure(figsize=(6,6))
    y = np.array(data)
    np.clip(data, clip, None, out=y) # Clip for better readability
    if x is None:
        x = list(range(start_step, start_step + len(data)))
    plt.plot(x, y)
    # Plot line for averages
    if average > 0 and len(data) > average:
        sums = [np.sum(data[0: average])]
        for i in range(0, len(data) - average - 1):
            sums.append(sums[i] - data[i] + data[i + 1 + average])
        averages = [sums[i] / average for i in range(len(sums))]
        plt.plot(range(average, len(data)), averages, label=f'Mean over Past {average} Steps')
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

#min_data_len = min((len(data) for data in datasets))
#datasets = [data[:min_data_len] for data in datasets] 
#minLen = min(len(data) for data in datasets)
#xs = np.arange(minLen) / 10
#interp_funcs = [scipy.interpolate.interp1d(xs, data) for data in datasets]
#y = [[interp_func(x/10) for x in range(0, minLen)] for interp_func in interp_funcs]
#print(len(y[0]))

# Plot sensor temperature data
#create_multiplot(y, 'Sensor Temperature over Simulation Time', 'Time', 'Temperature', labels, f'temp/sensortemps')

#####################################################################
# Load simulation data
#file_path = 'graphics60kreward/figure_data0.pkl'
file_path = 'figure_data0.pkl'

#file_path = 'tmp.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)
    if(len(data) == 10):
        sensor_ids, rate_frequencies, rate_log, throughput_log, reward_log, similarity_reward_log, throughput_reward_log, max_ind_set_log, chunks_sent_log, total_transmissions_received_log = data
    else:
        sensor_ids, rate_frequencies, rate_log, throughput_log, reward_log, similarity_reward_log, throughput_reward_log, max_ind_set_log, chunks_sent_log, total_transmissions_received_log, transmissions_log = data


s = 10
a = 20000
rate_log = rate_log[s:]
reward_log['throughput'] = reward_log['throughput'][s:]
similarity_reward_log = similarity_reward_log[s:]
throughput_reward_log = throughput_reward_log[s:]
max_ind_set_log = max_ind_set_log[s:]
chunks_sent_log = chunks_sent_log[s:]
total_transmissions_received_log = total_transmissions_received_log[s:]
throughput_log = np.array(throughput_log).T[s:]

"""
file_path = 'sensortemps0.pkl'
with open(file_path, 'rb') as file:
    data = pickle.load(file)
    transmissions_log = data
xs = []
ys = []

for transmissions in transmissions_log:
    print(transmissions)
    temps, time = zip(*transmissions)
    temps = [float(t) for t in temps]
    time = [t for t in time]
    xs.append(time)
    ys.append(temps)

create_multiplot(ys, 'Sensor Temperature over Simulation Time', 'Time (Seconds)', 'Temperature', labels, f'temp/sensortemps2', xs)
#create_plot(temps, 'Sensor temps', 'Step', 'Temperature', 'sensor_temps', x=time)
"""

# Plot throughput and chunks lost
#throughput_log = np.array(throughput_log).T
#throughput_log = np.array(throughput_log).T[s:]
throughput = [np.sum(t) for t in throughput_log]
chunks_sent = [np.sum(c) for c in chunks_sent_log]

total_throughput = [throughput[0]]
total_chunks_sent = [chunks_sent[0]]

for i in range(1, len(throughput)):
    total_throughput.append(throughput[i] + total_throughput[i-1])
    total_chunks_sent.append(chunks_sent[i] + total_chunks_sent[i-1])

chunks_lost = np.array(chunks_sent) - np.array(total_transmissions_received_log)
create_plot(chunks_lost, 'Cluster Head Transmissions Possibly Lost', 'Step', 'Transmissions Lost (Packets)', 'packet_loss')
create_plot(throughput, 'Cluster Head Throughput over Steps', 'Step', 'Succesfull Transmissions (Packets/S)', 'throughput', average=a)

# Plot total MIS Throughputs
total_ind_set_throughputs = []
for t in range(len(max_ind_set_log)):
    sensor_throughputs = throughput_log[t]
    total_throughput = 0
    for sensor in max_ind_set_log[t]:
        total_throughput = total_throughput + sensor_throughputs[sensor] 
    total_ind_set_throughputs.append(total_throughput)

create_plot(total_ind_set_throughputs, 'Total MIS. Throughput over Steps', 'Step', 'Throughput (Packets/S)', 'total_mis_throughput', average=a)

# Plot loss
# with open('loss.pkl', 'rb') as file:
#    batch_size, loss = pickle.load(file) 
# throughput_loss = np.array(loss['throughput'])
# throughput_loss = np.reshape(throughput_loss, (-1, 10))
# throughput_average_loss = np.mean(throughput_loss, axis=1)
# create_plot(throughput_average_loss, 'Throughput Trainning Loss', 'Step', 'Loss', 'throughput_loss', average=100, start_step=batch_size)

"""
similarity_loss = np.array(loss['similarity'])
similarity_loss = np.reshape(similarity_loss, (-1, 10))
similarity_average_loss = np.mean(similarity_loss, axis=1)
create_plot(similarity_average_loss, 'Similarity Trainning Loss', 'Step', 'Loss', 'similarity_loss', average=100, start_step=batch_size)
"""

# Plot rewards
throughput_rewards = np.array(reward_log["throughput"])
#average_throughput_rewards = np.mean(throughput_rewards, axis=1)
#np.clip(throughput_rewards, -0.3, 2, out=throughput_rewards) # Clip for better readability
create_plot(throughput_rewards, 'Throughput Rewards', 'Step', 'Reward', 'throughput_reward', average=a, clip=-10)

similarity_rewards = np.array(reward_log["similarity"]).reshape(-1, 10)
average_similarity_rewards = np.mean(similarity_rewards, axis=1)
np.clip(average_similarity_rewards, -5, 0, out=average_similarity_rewards) # Clip for better readability 
create_plot(average_similarity_rewards, 'Average Similarity Rewards', 'Step', 'Reward', 'similarity_reward', average=a)


