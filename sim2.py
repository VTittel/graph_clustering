from collections import defaultdict
from pprint import pprint

import numpy as np
import pandas as pd
import networkx as nx
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from graphrole import RecursiveFeatureExtractor, RoleExtractor
import math
import time

# Initialise graph
G = nx.Graph()

nodes = pd.read_csv('graph/nodes_new.csv', sep=',', header=None)
edges = pd.read_csv('graph/edges_new.csv', sep=',', header=None)
start = time.time()

# Sample nodes and save to new dataframe
sampled_nodes = nodes.sample(frac=0.001, replace=True, random_state=100)

# List of nodes we have sampled
nodes_list = list(sampled_nodes[0])
print("Number of sampled nodes : " + str(len(nodes_list)))

# For each samples nodes, only keep edges from that node
new_edges = edges.loc[edges[1].isin(nodes_list) | edges[2].isin(nodes_list)]
new_edges.reset_index(inplace=True)
print("Number of edges from sampled nodes : " + str(len(new_edges.index)))

# Printing data frame
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
   #print(sampled_nodes)

# Create a networkx graph
for i in range(0, len(new_edges.index)):
    fromNode = float(new_edges[1][i])
    fromNode = int(fromNode)
    toNode = int(new_edges[2][i])
    edgeWeight = new_edges[4][i]

    # Some conversions because of screwed up csv files
    if isinstance(edgeWeight, str) and edgeWeight == "nan":
        edgeWeight = int(1)

    if isinstance(edgeWeight, float) and math.isnan(edgeWeight):
        edgeWeight = int(1)

    edgeWeight = float(edgeWeight)
    edgeWeight = int(edgeWeight)

    G.add_edge(fromNode, toNode, weight=edgeWeight)

end = time.time()
print("Time to store graph :" + " " + str(end - start) + " seconds")
print(nx.info(G))

# Get the types of all sampled nodes
allsampled = nodes.loc[nodes[0].isin(nx.nodes(G))]
node_type = list(allsampled[1])

# Compute the largest connected graph
#Gp = nx.maximal_independent_set(G)

# Create an adjacency dictionary and populate it
adj_mat = nx.to_numpy_matrix(G)
df = pd.DataFrame(adj_mat)

# Do clustering. Affinity is euclidean by default
"""
algo = FuzzyKMeans(k=4, m=2)

print(df)
print(nx.info(G))

start = time.time()
algo.fit(df)
end = time.time()
print("Time to cluster :" + " " + str(end - start) + " seconds")

labels_array = np.array(algo.labels_)

# Print label counts
unique_elements, counts_elements = np.unique(labels_array, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))

# Assign each node a label
labelDict = {}
typeDict = {}
for node, label, type in zip(G, labels_array, node_type):
    labelDict[node] = str(label)
    typeDict[node] = str(type)

# Add labels as attributes to nodes of G, and Export graph
nx.set_node_attributes(G, labelDict, "labels")
nx.set_node_attributes(G, typeDict, "type")
print(nx.get_node_attributes(G, "labels"))
nx.write_gml(G, "test.gml")
"""

# RolX implementation. Pretty prints the roles
feature_extractor = RecursiveFeatureExtractor(G)
features = feature_extractor.extract_features()
role_extractor = RoleExtractor(n_roles=None)
role_extractor.extract_role_factors(features)
roles = role_extractor.roles
pprint(roles)