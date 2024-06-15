import os
import pickle

import matplotlib
import numpy as np
import seaborn

import graph_functions as graph
from parameters import *

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 18 # using a size in points
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
markers = ['o', 's', 'p', 'd', '*']

colors = seaborn.color_palette('rocket')
colors.reverse()


def number_of_disconnected_users(G, number_of_BS):
    discon_list = [1 for u, v in G.degree if u >= number_of_BS and v == 0]
    return len(discon_list)


def find_bs(xpop, ypop, xbs, ybs, max_connections):
    indices = find_distance(xpop, ypop, xbs, ybs).argsort()
    return indices[:max_connections]


def average(x):
    if len(x) > 0:
        return sum(x) / len(x)
    else:
        return 0


def average_nonempty(x):
    X = [i for i in x if i != 0]
    if len(X) > 0:
        return sum(X) / len(X)
    else:
        return 0

def find_channel(G, Gnew, xbs, ybs, xpop, ypop, noise):
    number_of_users = len(xpop)
    number_of_BS = len(xbs)

    channel = [0 for i in range(number_of_users)]
    if SNR:
        for edge in Gnew.edges():
            u, v = edge[0], edge[1] - number_of_BS
            channel[v] += find_shannon_capacity_SNR(u, v, G, number_of_BS / (xDelta * yDelta), xbs, ybs, xpop, ypop, noise)
    else:
        for v in range(number_of_users):
            distances = find_squared_distance(xbs, ybs, xpop[v], ypop[v])
            # smallest_distances = np.sort(distances)
            # no_interferers = np.sum(np.power(smallest_distances, -alpha / 2))
            no_interferers = 0

            interference = np.sum(np.power(distances, -alpha / 2)) - no_interferers

            for e1, e2 in Gnew.edges(v + number_of_BS):
                bs = min(e1, e2)
                channel[v] += (total_bandwidth / (G.degree(bs) * bandwidth_split)) * math.log2(
                    1 + (ptx * distances[bs] ** (-alpha / 2)) / (noise + ptx * interference))

    return channel


def from_memory(filename):
    if os.path.exists(filename):
        file = pickle.load(open(filename, 'rb'))
        return file
    else:
        return None


def find_distance(x, y, xbs, ybs):
    return np.sqrt(find_squared_distance(x, y, xbs, ybs))


def find_squared_distance(x, y, xbs, ybs):
    x = np.minimum((x - np.array(xbs)) % xDelta, (np.array(xbs) - x) % xDelta)
    y = np.minimum((y - np.array(ybs)) % yDelta, (np.array(ybs) - y) % yDelta)
    return (x ** 2 + y ** 2)


def find_shannon_capacity_SNR(u, v, G, labdaBS, xbs, ybs, xpop, ypop, noise):
    snr = find_snr(u, v, xbs, ybs, xpop, ypop, noise)
    return (total_bandwidth / (G.degree(u) * labdaBS * (xDelta * yDelta))) * math.log2(1 + snr)


def find_snr(u, v, xbs, ybs, xpop, ypop, noise):
    dist = find_squared_distance(xbs[u], ybs[u], xpop[v], ypop[v])
    return ptx / noise * max(1, dist) ** (-alpha / 2)


def distance_pdf(x, lbs, k):
    return 2 * (lbs * pi * x ** 2) ** k / (x * math.factorial(k - 1)) * math.exp(-lbs * pi * x ** 2)


def find_channel_outage(filename, x_values, k, G, xbs, ybs, xpop, ypop, iteration, number_of_BS, noise=None):
    channel_sum = from_memory(str('MC/parts/channel_sum' + str(min_x) + str(max_x) + str(delta) + filename))
    discon_sum = from_memory(str('MC/parts/disconnected_sum' + str(min_x) + str(max_x) + str(delta) + filename))
    if channel_sum == None or discon_sum == None or True:
        channel_sum = []
        discon_sum = []
        for x in x_values:
            if OBJ == RandomFailure:
                Gnew = graph.random_failure(G, x, number_of_BS)
            elif OBJ == RandomFailureLINK:
                Gnew = graph.random_failure_link(G, x)
            elif OBJ == DistanceFailure:
                Gnew = graph.distance_failure(G, x, xbs, ybs, xpop, ypop)
            elif OBJ == LoSFailure:
                Gnew = graph.los_failure(G, x, xbs, ybs, xpop, ypop)
            elif OBJ == NoFailure:
                number_of_BS = int(x * (yDelta * xDelta))
                labdaBS = number_of_BS / (yDelta * xDelta)
                name = str(
                    'MC/Graphs/' + 'G_lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(xDelta) + 'k=' + str(
                        k) + 'iteration=' + str(iteration) + '.p')

                if os.path.exists(name):
                    G = pickle.load(open(name, 'rb'))
                    xbs, ybs = initialise_BSs(number_of_BS)
                else:
                    xbs, ybs = initialise_BSs(number_of_BS)
                    G, colorlist, nodesize = graph.make_graph(xbs, ybs, xpop, ypop, number_of_BS, number_of_users)
                    for node in range(number_of_users):
                        bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
                        for bs in bss:
                            G.add_edge(node + number_of_BS, bs)

                    pickle.dump(G, open(name, 'wb'), protocol=4)
                Gnew = G.copy()

            if Reallocation:
                channel = find_channel(Gnew, Gnew, xbs, ybs, xpop, ypop, noise)
            else:
                channel = find_channel(G, Gnew, xbs, ybs, xpop, ypop, noise)

            disconnected = number_of_disconnected_users(Gnew, number_of_BS) / number_of_users

            channel = average(channel)

            channel_sum.append(channel)
            discon_sum.append(disconnected)

        pickle.dump(channel_sum,
                    open(str('MC/parts/channel_sum' + str(min_x) + str(max_x) + str(delta) + filename), 'wb'),
                    protocol=4)
        pickle.dump(discon_sum,
                    open(str('MC/parts/disconnected_sum' + str(min_x) + str(max_x) + str(delta) + filename), 'wb'),
                    protocol=4)

    return channel_sum, discon_sum
