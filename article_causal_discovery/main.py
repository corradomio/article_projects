from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import IIDSimulation, DAG
from castle.algorithms import *

# Get a list of all attributes (including functions, classes, and variables) in the castle.algorithms module
all_algorithms = dir()

# Print the list, excluding special attributes and functions starting with "__"
print([algo for algo in all_algorithms if not algo.startswith("__")])

import numpy as np

def generate_random_dag_adjacency_matrix(n_nodes, edge_prob):
    """
    Generate a random adjacency matrix for a Directed Acyclic Graph (DAG).
    based on a graph akin to the Erdős-Rényi model (the G(n,p) variant)

    Parameters:
    - n_nodes: Number of nodes in the graph.
    - edge_prob: Probability of an edge being present between two nodes (0 <= edge_prob <= 1).

    Returns:
    - A numpy array representing the adjacency matrix of the DAG.
    """
    # Initialize an empty adjacency matrix
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)

    # Assign a fixed topological order to the nodes
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Ensure no backward edges
            if np.random.rand() < edge_prob:  # Randomly decide to add an edge based on edge_prob
                adjacency_matrix[i, j] = 1

    return adjacency_matrix

# Example usage
# n_nodes = 10  # Number of nodes
# edge_prob = 0.2  # Probability of an edge

# adj_matrix = generate_random_dag_adjacency_matrix(n_nodes, edge_prob)
# print(adj_matrix)

# Example usage for directed graph
gt_dag = generate_random_dag_adjacency_matrix(10, 0.70)
print(gt_dag)

# data simulation, simulate a true causal dag and train_data.

dataset = IIDSimulation(W=gt_dag, n=2000, method='linear',
                        sem_type='gauss')
true_causal_matrix, X = dataset.B, dataset.X

# GOLEM learn
golem = GOLEM(num_iter=1e3)
golem.learn(X)
print(golem.causal_matrix)

# plot predict_dag and true_dag
GraphDAG(golem.causal_matrix, true_causal_matrix, 'result')

# calculate metrics
mt = MetricsDAG(golem.causal_matrix, true_causal_matrix)
print(mt.metrics)


import numpy as np

def d_separation(adj_matrix):
    n = adj_matrix.shape[0]
    d_sep_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Marking all pairs as d-separated initially
                d_sep_matrix[i, j] = 1

                # Performing depth-first search from node i to check if node j is reachable
                visited = set()
                stack = [i]
                while stack:
                    node = stack.pop()
                    if node == j:
                        d_sep_matrix[i, j] = 0
                        break
                    visited.add(node)
                    parents = np.where(adj_matrix[:, node] != 0)[0]
                    for parent in parents:
                        if parent not in visited:
                            stack.append(parent)
    return d_sep_matrix

# Example adjacency matrix of a DAG
adj_matrix = np.array([[0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1],
                       [0, 0, 0, 0]])

d_sep_matrix = d_separation(adj_matrix)
print("D-separation matrix:")
print(d_sep_matrix)

