from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.approximation import maximum_independent_set
from sklearn import cluster
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import math
import time

# Should probably get this from csv and not hardcoded
numRepos = 124409
numUsers = 23755
numNodes = 148164

# Initialise graph
G = nx.Graph()

nodes = pd.read_csv('graph/nodes.csv', sep=',', header=None)
edges = pd.read_csv('graph/edges.csv', sep=',', header=None)
start = time.time()

sampled_nodes = nodes.sample(frac=0.01, replace=True, random_state=1)

# List of nodes we have sampled
nodes_list = list(sampled_nodes[0])

print(len(nodes_list))

new_edges = edges.loc[edges.index.isin(nodes_list)]

new_edges.reset_index(inplace=True)

# Printing data frame
#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(test)

for i in range(2, len(new_edges.index)):
    fromNode = new_edges[1][i]
    toNode = new_edges[2][i]
    edgeWeight = new_edges[4][i]

    #if ((not fromNode == "") and (not toNode == "") and (not edgeWeight == "")):
    G.add_edge(fromNode, toNode, weight=edgeWeight)

end = time.time()
print("Time to store graph :" + " " + str(end - start) + " seconds")

nx.write_gml(G, "test.gml")


Gp = maximum_independent_set(G)


print(Gp)

# Memory maps to store distance arrays for users and repositories on disk
# repoDistance = np.memmap('repo_distance.dat', dtype='int', mode='w+', shape=(124409,124409))
# userDistance = np.memmap('user_distance.dat', dtype='int', mode='w+', shape=(numUsers,numUsers))
#graphMatrix = np.memmap('nodes.dat', dtype='int', mode='w+', shape=(numNodes, numNodes))

"""
for node in G:
    if type(node) == float or type(node) == int:
        if math.isnan(node):
            continue

    for neighbor in G.neighbors(node):

        if neighbor == 'nan':
            continue

        if type(neighbor) == float:
            if math.isnan(neighbor):
                continue

        print("Node type : " + str(type(node)))
        print(node)

        weight = int(G.get_edge_data(node, neighbor).get('weight'))
        node2 = int(node)
        neighbor2 = 0

        if (not math.isnan(node2)) and (not math.isnan(neighbor2)):
            graphMatrix[node2][neighbor2] = weight

    graphMatrix[int(node)][int(node)] = 1

"""

"""
start = time.time()

sparse = csr_matrix(graphMatrix)

end = time.time()
print("Time to convert matrix to CSR :" + " " + str(end - start) + " seconds")

#algo = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=50000,
 #                      max_no_improvement=10, verbose=0)

# Affinity is euclidean by default
algo = cluster.KMeans(n_clusters=5, n_init=50)

start = time.time()
algo.fit(sparse)
end = time.time()
print("Time to cluster :" + " " + str(end - start) + " seconds")

print(list(algo.labels_))
"""