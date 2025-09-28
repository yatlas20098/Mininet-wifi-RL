import networkx as nx
import numpy as np

"""
    Compute the similarity matrix and create a redudancy graph

    Args:
        temperature_data (list): list of temperatures transmitted by each sensor
        
    Returns:
        similarity (numpy 2d list): matrix with i,j=1 if sensor i and j transmitted similar data and 0 otherwise 
        redudancy_graph (networkx graph): graph with verticies as sensors and edges as similarity between sensors
        sensor_effective_throughputs: a list of the effective throughputs for each sensor (i.e. fixing a sensor s, the maximium throughput of a sensor - possibily s itself - transmitting similar data to s)
    """

def compute_similarity_and_redundancy_graph(sim_config, rates, temperature_data, throughputs, replay=-1):
    num_sensors = len(sim_config.sensor_ids)
    
    # Get list of sensors that had at least one succesfull transmission 
    awake_sensors = list(temperature_data.keys())
    sensor_effective_throughputs = throughputs.copy()
    
    # Create a graph with veritices representing sensors and edges denoting similarity
    """
    redundancy_graph = nx.Graph()
    redundancy_graph.add_nodes_from([(i, {"throughput": throughputs[i]/400,
                                         "transmision_rate": rates[i]/400
                                         }) for i in range(num_sensors)])
    print("Throughputs: ", [t/400 for t in throughputs])
    print("Rates: ", [r/400 for r in rates])
    for node in redundancy_graph.nodes:
        throughput = redundancy_graph.nodes[node]["throughput"]
        transmission_rate = redundancy_graph.nodes[node]["transmision_rate"]
        redundancy_graph.nodes[node]["x"] = [throughput, transmission_rate]

    """

    redundancy_graph = nx.Graph()
    redundancy_graph.add_nodes_from([(i, {"throughput": throughputs[i]/400}) for i in range(num_sensors)])
    print("Throughputs: ", [t/400 for t in throughputs])
    print("Rates: ", [r/400 for r in rates])
    for node in redundancy_graph.nodes:
        throughput = redundancy_graph.nodes[node]["throughput"]
        #transmission_rate = redundancy_graph.nodes[node]["transmision_rate"]
        redundancy_graph.nodes[node]["x"] = [throughput]

    connectivity_graph = redundancy_graph.copy()
    for i in range(num_sensors - 1):
        for j in range(i+1, num_sensors):
            connectivity_graph.add_edge(i, j)
    
    # Initalize similarity matrix
    similarity = np.zeros((num_sensors, num_sensors))
    
    for i in range(len(awake_sensors)):
        for j in range(i + 1, len(awake_sensors)):
            # TODO: Currently only using first data point for similiarity
            l = min(len(temperature_data[awake_sensors[i]]), len(temperature_data[awake_sensors[j]]))
            similarity[awake_sensors[i], awake_sensors[j]] = int(((temperature_data[awake_sensors[i]][:l] - temperature_data[awake_sensors[j]][:l])**2).mean() <= sim_config.similarity_threshold)
            similarity[awake_sensors[j], awake_sensors[i]] = similarity[awake_sensors[i], awake_sensors[j]] 

    for i in range(len(awake_sensors)):
        for j in range(i + 1, len(awake_sensors)):
            # Add edge from vertex i to vertex j if sensors i and j are similar
            if similarity[awake_sensors[i],awake_sensors[j]] == 1:
                redundancy_graph.add_edge(awake_sensors[i],awake_sensors[j])
                s1 = awake_sensors[i]
                s2 = awake_sensors[j]
                max_throughput = max(sensor_effective_throughputs[s1], sensor_effective_throughputs[s2])
                sensor_effective_throughputs[s1] = sensor_effective_throughputs[s2] = max_throughput
    
    nx.set_node_attributes(redundancy_graph, {i: sensor_effective_throughputs[i] for i in range(num_sensors)}, "effective_throughput")

    return similarity, connectivity_graph, redundancy_graph 
