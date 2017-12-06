import networkx as nx
import numpy as np


def bomb_graph(S):
    G = nx.complete_graph(int(S/2))
    for i in range(int(S/2)):
        G.add_node(int(S/2 + i))
        G.add_edge(i, int(S/2+i))
    for e in G.edges():
        if e[0] >= S/2 or e[1] >= S/2:
            G[e[0]][e[1]]["weight"] = 10
        else:
            G[e[0]][e[1]]["weight"] = 0.1
    mat = nx.to_numpy_matrix(G)
    arrivals = [1/S for i in range(S)]
    departures = [0 for i in range(int(S/2))] + [1 for i in range(int(S/2), S)]
    return (mat, arrivals, departures)


def random_graph(S):
    G = nx.erdos_renyi_graph(S, 0.5)
    for e in G.edges():
                G[e[0]][e[1]]["weight"] = np.random.random()
    mat = nx.to_numpy_matrix(G)
    arrivals = np.random.random(S)
    departures = np.random.random(S)
    return (mat, arrivals, departures)
