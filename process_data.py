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

def create_plot(data, title, xlabel, ylabel, output_name, average=0):
    print(f"Plotting {output_name}...\n")
    plt.figure(figsize=(6,6))
    plt.plot(range(len(data)), data)
    # Plot line for averages
    if average > 0:
        averages = [np.mean(data[i-100:i]) for i in range(average, len(data))]
        plt.plot(range(average, len(data)), averages, label=f'Average over past {average} steps')
        m, b = np.polyfit(range(len(data)), data, 1) # 1 indicates linear fit
        plt.plot(range(len(data)), m * range(len(data)) + b, label=f'Line of Best Fit')
        
        # Plot best fit


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
    sensor_ids, rate_frequencies, rate_log, _, throughput_log, reward_log, clique_reward_log, throughput_reward_log, clique_log, chunks_sent_log = data
print(f'Data len: {len(data)}')

with open('loss.pkl', 'rb') as file:
    data = pickle.load(file)
    loss = data

print(rate_frequencies)
print(f'Total Steps :{len(reward_log)}')
#clique_log = [clique_log[i] for i in range(len(clique_log)) if i % 10 == 0]
throughput_log = np.array(throughput_log).T
throughput = [np.sum(t) for t in throughput_log]
chunks_sent = [np.sum(c) for c in chunks_sent_log]

#throughput = np.array([throughput[i] for i in valid])[100:]
#chunks_sent = np.array([chunks_sent[i] for i in valid])[100:]

total_throughput = [throughput[0]]
total_chunks_sent = [chunks_sent[0]]

for i in range(1, len(chunks_sent)):
    total_throughput.append(throughput[i] + total_throughput[i-1])
    total_chunks_sent.append(chunks_sent[i] + total_chunks_sent[i-1])

chunks_lost = np.array(total_chunks_sent) - np.array(total_throughput)



#chunks_lost = chunks_sent_over_10_steps - throughput_over_10_steps 


print(f'Mean throughput: {np.mean(throughput[:1500])}')
print(f'Mean throughput: {np.mean(throughput[1500:])}')


print(f'Mean reward: {np.mean(reward_log[:1500])}')
print(f'Mean reward: {np.mean(reward_log[1500:])}')

#sum_throughput = [np.sum(throughput) for throughput in throughput_log]
#sum_throughput = [x for x in throughput if x < 2000] 
num_cliques = [len(cliques) for cliques in clique_log]
#energy_log = np.array(energy_log).T
#mean_energy = [np.mean(energy) for energy in energy_log]

create_plot(chunks_lost, 'Cluster Head Chunks Lost', 'Step', 'Chunks Lost', 'chunks_lost')

create_plot(throughput, 'Cluster Head Throughput over Steps', 'Step', 'Succesfull transmissions', 'throughput', average=100)
create_plot([r for r in reward_log if r > -4], 'Rewards over Steps', 'Step', 'Reward', 'rewards')

create_plot(loss, 'Trainning Loss', 'step', 'Loss', 'Loss', average=100)
#create_plot(mean_energy, 'Average Sensor Energy over Steps', 'Step', 'Average Percent of Energy Remaining', 'energy') 

create_plot(num_cliques[:300], 'Number of cliques over Steps', 'Step', 'Number of Cliques', 'num_cliques')

create_plot(clique_reward_log, 'Clique Rewards over Steps', 'Step', 'Reward', 'Clique Rewards', average=100)
#    create_plot(throughput_reward_log, 'Throughput Rewards over Steps', 'Step', 'Reward', 'rewards')


