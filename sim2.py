from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import cluster
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import math
import time

# Initialise graph
G = nx.DiGraph()

nodes = pd.read_csv('graph/nodes.csv', sep=',', header=None)
edges = pd.read_csv('graph/edges.csv', sep=',', header=None)
start = time.time()

# Sample nodes and save to new dataframe
sampled_nodes = nodes.sample(frac=0.05, replace=True, random_state=1)

# List of nodes we have sampled
nodes_list = list(sampled_nodes[0])
print("Number of sampled nodes : " + str(len(nodes_list)))

# For each samples nodes, only keep edges from that node
new_edges = edges.loc[edges.index.isin(nodes_list)]

new_edges.reset_index(inplace=True)

# Printing data frame
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(test)

# Create a networkx graph
for i in range(2, len(new_edges.index)):
    fromNode = new_edges[1][i]
    toNode = new_edges[2][i]
    edgeWeight = new_edges[4][i]

    #if ((not fromNode == "") and (not toNode == "") and (not edgeWeight == "")):
    G.add_edge(fromNode, toNode, weight=edgeWeight)

end = time.time()
print("Time to store graph :" + " " + str(end - start) + " seconds")

# Export graph
nx.write_gml(G, "test.gml")

# Compute the largest connected graph
#Gp = nx.maximal_independent_set(G)

# Create an adjacency dictionary and populate it
graphDict = defaultdict(dict)

for node in G:
    for neighbor in G.neighbors(node):
        weight = int(G.get_edge_data(node, neighbor).get('weight'))
        graphDict[node][neighbor] = weight

df = pd.DataFrame.from_dict(graphDict)
df.fillna(0, inplace=True)

# Affinity is euclidean by default
algo = cluster.KMeans(n_clusters=10, n_init=50)

start = time.time()
algo.fit(df)
end = time.time()
print("Time to cluster :" + " " + str(end - start) + " seconds")

labels = np.array(algo.labels_)

# Print label counts
unique_elements, counts_elements = np.unique(labels, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))
