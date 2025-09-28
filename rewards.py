import networkx as nx
from networkx.algorithms import approximation as approx
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpBinary, PULP_CBC_CMD
from collections import OrderedDict
import numpy as np

bounded_log = lambda x: np.log2(min(1, max(0.01, x))) 

def _get_max_ind_set(redundancy_graph):
    LP = LpProblem("Weighted_Max_Independent_Set", LpMaximize)

    # binary variabls for each node: 1 if selected, 0 otherwise
    x = {v: LpVariable(f"x_{v}", cat=LpBinary) for v in redundancy_graph.nodes}

    # Objective: maximize total throughput 
    LP += lpSum(redundancy_graph.nodes[v]["throughput"] * x[v] for v in redundancy_graph.nodes)

    # Constraint: for each edge, at most one endpoint can be in the independent st
    for u,v in redundancy_graph.edges:
        LP += x[u] + x[v] <= 1

    # Solve the LP without any console output
    LP.solve(PULP_CBC_CMD(msg=0))

    independent_set = [v for v in redundancy_graph.nodes if x[v].varValue == 1]
    return independent_set

def throughput(sim_config, redundancy_graph):
    ind_set_with_max_throughput = _get_max_ind_set(redundancy_graph)
    ind_set_total_throughput = np.sum([redundancy_graph.nodes[v]["throughput"] for v in ind_set_with_max_throughput])
    total_throughput = np.sum([redundancy_graph.nodes[v]["throughput"] for v in redundancy_graph.nodes])
    redundant_throughput = total_throughput - ind_set_total_throughput
    
    maxF = np.max(sim_config.transmission_frequencies)
    mis = approx.maximum_independent_set(redundancy_graph)
    #total_throughput_bound = len(sim_config.sensor_ids) * maxF
    # TODO: divide by number of sensors, not 10
    throughput_reward = (ind_set_total_throughput - 0.1*redundant_throughput) / len(mis)
    #throughput_reward = (ind_set_total_throughput - 0.5*redundant_throughput) / len(sim_config.sensor_ids)


    #throughput_reward = bounded_log((ind_set_total_throughput - redundant_throughput) / total_throughput_bound) / 4

    return [throughput_reward, ind_set_with_max_throughput]

def similarity(sim_config, redundancy_graph):
    # Get similarity reward
    max_throughput = max(nx.get_node_attributes(redundancy_graph, "throughput"))
    num_sensors = len(sim_config.sensor_ids)
    sensor_effective_throughputs = [redundancy_graph.nodes[i]["effective_throughput"] for i in range(num_sensors)]
    similarity_reward = [bounded_log(sensor_effective_throughputs[i] / max_throughput) for i in range(num_sensors)]

    return [similarity_reward]

def get_rewards(reward_config, sim_config, redundancy_graph):
    rewards = OrderedDict()
    reward_output = {}

    for reward_name, reward in reward_config.items():
        reward_fn_output = reward["compute_fn"](sim_config, redundancy_graph)
        rewards[reward_name] = reward_fn_output[0]
        if len(reward_fn_output) > 1:
            reward_output[reward_name] = tuple((reward_fn_output[i] for i in range(1, len(reward_fn_output))))

    return rewards, reward_output 
