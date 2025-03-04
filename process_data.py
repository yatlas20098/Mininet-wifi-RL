import pickle
import numpy as np
import matplotlib.pyplot as plt

def create_multiplot(datasets, title, xlabel, ylabel, legend_labels, output_name):
    print(f"Plotting {output_name}...\n")
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

def create_plot(data, title, xlabel, ylabel, output_name):
    print(f"Plotting {output_name}...\n")
    plt.figure(figsize=(6,6))
    plt.plot(range(len(data)), data)
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

create_multiplot(datasets, 'Sensor Temperature over Steps', 'Step', 'Temperature', labels, f'temp/sensortemps')

#create_multiplot(datasets, title, xlabel, ylabel, legend_labels, output_name):

#file_path = 'prev_results/data_241204.pkl'
file_path = 'figure_data.pkl'

with open(file_path, 'rb') as file:
    data = pickle.load(file)

if len(data) == 7:
    sensor_ids, rate_frequencies, rate_log, energy_log, throughput_log, reward_log, clique_log = data
else:
    sensor_ids, rate_frequencies, rate_log, energy_log, throughput_log, reward_log, clique_reward_log, throughput_reward_log, clique_log = data
print(f'Data len: {len(data)}')

print(rate_frequencies)
print(f'Total Steps :{len(reward_log)}')
#clique_log = [clique_log[i] for i in range(len(clique_log)) if i % 10 == 0]
throughput_log = np.array(throughput_log).T
throughput = [np.sum(t) for t in throughput_log]

print(f'Mean throughput: {np.mean(throughput[:1500])}')
print(f'Mean throughput: {np.mean(throughput[1500:])}')


print(f'Mean reward: {np.mean(reward_log[:1500])}')
print(f'Mean reward: {np.mean(reward_log[1500:])}')

sum_throughput = [np.sum(throughput) for throughput in throughput_log]
num_cliques = [len(cliques) for cliques in clique_log]
energy_log = np.array(energy_log).T
mean_energy = [np.mean(energy) for energy in energy_log]

create_plot(sum_throughput, 'Cluster Head Throughput over Steps', 'Step', 'Throughput (KiB/s)', 'throughput')
create_plot(reward_log, 'Rewards over Steps', 'Step', 'Reward', 'rewards')

create_plot(mean_energy, 'Average Sensor Energy over Steps', 'Step', 'Average Percent of Energy Remaining', 'energy') 

create_plot(num_cliques[:300], 'Number of cliques over Steps', 'Step', 'Number of Cliques', 'num_cliques')

if len(data) == 9: 
    create_plot(clique_reward_log, 'Clique Rewards over Steps', 'Step', 'Reward', 'rewards')
    create_plot(throughput_reward_log, 'Throughput Rewards over Steps', 'Step', 'Reward', 'rewards')


