import networkx as nx
import matplotlib.pyplot as plt
from parameters import *
import scipy.stats
import numpy as np
import functions as f


def make_graph(xbs, ybs, xpop, ypop, number_of_BS, number_of_users):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    for node in range(number_of_BS):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('b')
        nodesize.append(20)
    for node in range(number_of_users):
        G.add_node(node + number_of_BS, pos=(xpop[node], ypop[node]))
        colorlist.append('g')
        nodesize.append(3)
    return G, colorlist, nodesize

def draw_graph(G, colorlist, nodesize):
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist=G.nodes(), node_size=nodesize, node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray', alpha=0.3)
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])
    plt.show()

# --------------------- RANDOM FAILURES ------------------------------------------
# Every link in the network has a probability of p of breaking down
def random_failure(graph, p, number_of_BS):
    G = graph.copy()
    for bs in range(number_of_BS):
        if np.random.uniform(0, 1) <= p:
            G.remove_node(bs)
    return G

def random_failure_link(graph, p):
    G = graph.copy()
    for edge in G.edges():
        if np.random.uniform(0, 1) <= p:
            G.remove_edge(edge[0], edge[1])
    return G

# ------------------- LINK DISTANCE FAILURE -------------------------------------------
# Every link that is larger than rmax breaks down
def distance_failure(graph, rmax, xbs, ybs, xpop, ypop):
    G = graph.copy()
    for edge in G.edges:
        distance = f.find_distance(xpop[edge[1] - number_of_BS], ypop[edge[1] - number_of_BS], xbs[edge[0]], ybs[edge[0]])
        if distance >= rmax:
            G.remove_edge(edge[0], edge[1])
    return G

# ------------------------- LINE OF SIGHT FAILURE ---------------------------------------
def los_failure(graph, rlos, xbs, ybs, xpop, ypop):
    G = graph.copy()
    for edge in G.edges:
        distance = f.find_distance(xpop[edge[1] - number_of_BS], ypop[edge[1] - number_of_BS], xbs[edge[0]], ybs[edge[0]])
        if distance >= rlos:
            if np.random.uniform(0, 1) <= 1 - 1/distance * (rlos + distance * math.exp(-distance / (2*rlos)) - rlos * math.exp(-distance / (2*rlos))):
                G.remove_edge(edge[0], edge[1])
    return G
