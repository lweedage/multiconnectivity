import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import networkx as nx
import math
import scipy.special
import failure_models as fm
import os
import pickle
import seaborn
import matplotlib

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
pi = math.pi

a = [3.5, 7.2, 11.1, 15.2, 21.2]

xMin, xMax = 0, 1000  # in meters
yMin, yMax = 0, 1000  # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

labdaBS = 1e-6  # = how many BS per square meter
labdaU = 1e-4  # = how many users per square meter

total_bandwidth = 20 * 10 ** 6

alpha = 2
power = 10 ** (3.5)
noise = 10 ** (-167.1 + 5)
c = power / noise


# --------- Functions --------------------
def initialise_graph(points):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((points, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(points)]
    y = [yy[i][0] for i in range(points)]

    return x, y


def make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop):
    G = nx.Graph()
    colorlist = list()
    nodesize = list()
    for node in range(pointsBS):
        G.add_node(node, pos=(xbs[node], ybs[node]))
        colorlist.append('b')
        nodesize.append(20)
    for node in range(pointsPop):
        G.add_node(node + pointsBS, pos=(xpop[node], ypop[node]))
        colorlist.append('g')
        nodesize.append(3)
    return G, colorlist, nodesize


def draw_graph(G, colorlist, nodesize):
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(G, nx.get_node_attributes(G, 'pos'), nodelist=G.nodes(), node_size=nodesize,
                           node_color=colorlist, ax=ax)
    nx.draw_networkx_edges(G, nx.get_node_attributes(G, 'pos'), edge_color='gray', alpha=0.3)
    ax.set_xlim([xMin, xMax]), ax.set_ylim([yMin, yMax])
    plt.show()


def find_distance(x, y, xbs, ybs):
    x = np.minimum((x - np.array(xbs)) % xDelta, (np.array(xbs) - x) % xDelta)
    y = np.minimum((y - np.array(ybs)) % yDelta, (np.array(ybs) - y) % yDelta)
    return np.sqrt(x ** 2 + y ** 2)


def find_shannon_capacity(u, v, G, labdaBS, xbs, ybs, xpop, ypop):
    SNR = find_snr(u, v, xbs, ybs, xpop, ypop)
    return (total_bandwidth / (G.degree(u) * labdaBS)) * math.log2(1 + SNR)


def find_snr(u, v, xbs, ybs, xpop, ypop):
    dist = find_distance(xbs[u], ybs[u], xpop[v], ypop[v])
    if dist <= 1:
        return c
    else:
        return c * (dist) ** (-alpha)


def distance_pdf(x, lbs, k):
    return 2 * (lbs * pi * x ** 2) ** k / (x * math.factorial(k - 1)) * math.exp(-lbs * pi * x ** 2)


# --------------------- RANDOM FAILURES ------------------------------------------
# Every link in the network has a probability of p of breaking down
def random_failure(graph, p):
    G = graph.copy()
    for edge in G.edges:
        if np.random.uniform(0, 1) <= p:
            G.remove_edge(edge[0], edge[1])
    return G


# -------------------- OVERLOAD FAILURE -----------------------------------------
# Every base station that has more than K connections will break down (together with all links)
def overload_failure(graph, beta, pointsBS):
    G = graph.copy()
    for node in range(pointsBS):
        if G.degree(node) > 0:
            if np.random.uniform(0, 1) <= 1 - graph.degree(node) ** (-beta):
                for nb in list(nx.neighbors(G, node)):
                    G.remove_edge(node, nb)
    return G


def overload_failure_det(graph, K, pointsBS):
    G = graph.copy()
    for node in range(pointsBS):
        if G.degree(node) >= K:
            for nb in list(nx.neighbors(G, node)):
                G.remove_edge(node, nb)
    return G


# ------------------- LINK DISTANCE FAILURE -------------------------------------------
# Every link has a probability of breaking down, which becomes larger when the length of that link is larger
def distance_failure(graph, rmax, xbs, ybs, xpop, ypop):
    G = graph.copy()
    for edge in G.edges:
        distance = find_distance(xpop[edge[1] - pointsBS], ypop[edge[1] - pointsBS], xbs[edge[0]], ybs[edge[0]])
        if distance >= rmax:
            G.remove_edge(edge[0], edge[1])
    return G


def distance_failure_prop(graph, beta, xbs, ybs, xpop, ypop, k, lbs):
    G = graph.copy()
    pointsBS = len(xbs)
    for edge in G.edges:
        distance = find_distance(xpop[edge[1] - pointsBS], ypop[edge[1] - pointsBS], xbs[edge[0]], ybs[edge[0]])
        if distance >= 1:
            if np.random.uniform(0, 1) <= 1 - distance ** (-beta):
                G.remove_edge(edge[0], edge[1])
    return G


# ------------------------- LINE OF SIGHT FAILURE ---------------------------------------
def los_failure(graph, xbs, ybs, xpop, ypop, rlos):
    G = graph.copy()
    for edge in G.edges:
        distance = find_distance(xpop[edge[1] - pointsBS], ypop[edge[1] - pointsBS], xbs[edge[0]], ybs[edge[0]])
        if distance >= rlos:
            if np.random.uniform(0, 1) <= 1 - 1 / distance * (
                    rlos + distance * math.exp(-distance / (2 * rlos)) - rlos * math.exp(-distance / (2 * rlos))):
                G.remove_edge(edge[0], edge[1])
    return G


# ----------------------- OTHER FUNCTIONS ---------------------------------------
def number_of_disconnected_users(G, pointsBS):
    discon_list = [1 for u, v in G.degree if u >= pointsBS and v == 0]
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


def find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop, total_bandwidth, k):
    channel = [0 for i in range(pointsPop)]
    logSNR = [0 for i in range(pointsPop)]
    W = [0 for i in range(pointsPop)]
    for edge in Gnew.edges():
        u, v = edge[0], edge[1] - pointsBS
        channel[v] += find_shannon_capacity(u, v, G, labdaBS, xbs, ybs, xpop, ypop)
        logSNR[v] += math.log2(1 + find_snr(u, v, xbs, ybs, xpop, ypop))
        W[v] += (total_bandwidth / (G.degree(u) * labdaBS))
    return channel, logSNR, W


def from_memory(filename):
    if os.path.exists(filename):
        file = pickle.load(open(filename, 'rb'))
        return file


# -------------------------- THE PROGRAM --------------------------------------

pointsBS = labdaBS * (yDelta * xDelta)
pointsPop = labdaU * (yDelta * xDelta)

pointsBS = int(pointsBS)
pointsPop = int(pointsPop)

labdaBS = pointsBS / (yDelta * xDelta)
labdaU = pointsPop / (yDelta * xDelta)

xbs, ybs = initialise_graph(pointsBS)
xpop, ypop = initialise_graph(pointsPop)

delta = 10


Overload_Failure = False
Distance_Failures = False
Random_Failure = False
LoS_Failure = False
No_Failure = True
Draw = False
Save = True
NoReallocation = True

if NoReallocation:
    name = str(
        'simulation' + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'alpha=' + str(alpha) + 'area=' + str(xDelta))
else:
    name = str('simulation_with_reallocation' + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'alpha=' + str(
        alpha) + 'area=' + str(xDelta))

if Overload_Failure:
    fig1, ax1 = plt.subplots()  # overload failure channel capacity
    fig2, ax2 = plt.subplots()  # overload failure outage probability
if Distance_Failures:
    fig3, ax3 = plt.subplots()  # distance failure channel capacity
    fig4, ax4 = plt.subplots()  # distance failure outage probability
if Random_Failure:
    fig7, ax7 = plt.subplots()  # random failure channel capacity
    fig8, ax8 = plt.subplots()  # random failure outage probability
if LoS_Failure:
    fig9, ax9 = plt.subplots()  # LoS failure channel capacity
    fig10, ax10 = plt.subplots()  # LoS failure outage probability
if No_Failure:
    fig11, ax11 = plt.subplots()  # no failure channel

graph, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)
lijstk = [1, 2, 3, 4, 5]

for k in lijstk:
    G = graph.copy()
    if os.path.exists(
            str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p')) and Save:
        print('The graph for k =', k, 'is already stored in memory')
        G = pickle.load(
            open(str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'), 'rb'))
        pointsPop = pickle.load(
            open(str('Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        pointsBS = pickle.load(
            open(str('Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                 'rb'))
        xbs = pickle.load(
            open(str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'), 'rb'))
        ybs = pickle.load(
            open(str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'), 'rb'))
        xpop = pickle.load(
            open(str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'), 'rb'))
        ypop = pickle.load(
            open(str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'), 'rb'))
    else:
        for node in range(pointsPop):
            bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
            for bs in bss:
                G.add_edge(node + pointsBS, bs)
        if Draw:
            draw_graph(G, colorlist, nodesize)
        pickle.dump(G,
                    open(str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                         'wb'), protocol=4)
        pickle.dump(pointsBS,
                    open(str('Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                        k) + '.p'), 'wb'), protocol=4)
        pickle.dump(pointsPop,
                    open(str('Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                        k) + '.p'), 'wb'), protocol=4)
        pickle.dump(xbs,
                    open(str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                         'wb'), protocol=4)
        pickle.dump(ybs,
                    open(str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                         'wb'), protocol=4)
        pickle.dump(xpop,
                    open(str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                         'wb'), protocol=4)
        pickle.dump(ypop,
                    open(str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                         'wb'), protocol=4)
    print('Number of base stations:', pointsBS, ', Number of users:', pointsPop)
    # --------------------- OVERLOAD FAILURE --------------------
    if Overload_Failure:
        betas = np.arange(0.00001, 2, 2 / delta)

        if NoReallocation:
            filename = str(name + 'overload' + str(k) + str(delta) + '.p')
        else:
            filename = str('reallocation' + name + 'overload' + str(k) + str(delta) + '.p')

        channel_sum = from_memory(str('channel_sum' + filename))
        discon = from_memory(str('discon' + filename))
        logSNR_sum = from_memory(str('logSNR_sum' + filename))

        if channel_sum == None or discon == None or logSNR_sum == None:
            channel_sum = list()
            logSNR_sum = list()
            discon = []
            for beta in betas:
                Gnew = overload_failure(G, beta, pointsBS)
                if NoReallocation:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                else:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, Gnew, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)

                discon.append(number_of_disconnected_users(Gnew, pointsBS) / pointsPop)
                channel_sum.append(average(channel))
                logSNR_sum.append(average(logSNR))
            pickle.dump(channel_sum, open(str('channel_sum' + filename), 'wb'), protocol=4)
            pickle.dump(discon, open(str('discon' + filename), 'wb'), protocol=4)
            pickle.dump(logSNR_sum, open(str('logSNR_sum' + filename), 'wb'), protocol=4)

        # Channel capacity - overload failure
        if NoReallocation:
            ax1.plot(betas,
                     [fm.channel_capacity_overload(k, labdaU, labdaBS, beta, c, alpha, total_bandwidth) for beta in
                      betas],
                     color=colors[k - 1])
            ax1.scatter(betas, channel_sum, color=colors[k - 1], marker=markers[k - 1], label=str('$k = $ ' + str(k)),
                )
        else:
            ax1.scatter(betas, channel_sum, color=colors[k - 1], marker=markers[k - 1], label=str('$k = $ ' + str(k)))
        if k == 1:
            channel_sum_benchmarkOVER = np.array(channel_sum)
        diff = (np.array(channel_sum) - channel_sum_benchmarkOVER) / channel_sum_benchmarkOVER * 100
        print('Overload: For k =', k, 'the difference in performance is max', max(diff), 'and min', min(diff))
        # print(diff)
        # Outage probability - overload failure
        if NoReallocation:
            ax2.plot(betas, [fm.outage_probability_overload(k, labdaU / labdaBS, beta) for beta in betas],
                     color=colors[k - 1])
            ax2.scatter(betas, discon, color=colors[k - 1], marker=markers[k - 1],
                        label=str('$k = $ ' + str(k)))

        print('Proportional overload failure for k =', k, "done")

    if Distance_Failures:
        discon = []
        distances = np.arange(5, 25, 25 / delta)

        if NoReallocation:
            filename = str(name + 'distance' + str(k) + str(delta) + '.p')
        else:
            filename = str('reallocation' + name + 'distance' + str(k) + str(delta) + '.p')

        channel_sum = from_memory(str('channel_sum' + filename))
        discon = from_memory(str('discon' + filename))
        logSNR_sum = from_memory(str('logSNR_sum' + filename))

        if channel_sum == None or discon == None or logSNR_sum == None:
            channel_sum = list()
            logSNR_sum = list()
            discon = []
            for rmax in distances:
                Gnew = distance_failure(G, rmax, xbs, ybs, xpop, ypop)
                if NoReallocation:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                else:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, Gnew, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                discon.append(number_of_disconnected_users(Gnew, pointsBS) / pointsPop)
                channel_sum.append(average(channel))
                logSNR_sum.append(average(logSNR))
            pickle.dump(channel_sum, open(str('channel_sum' + filename), 'wb'), protocol=4)
            pickle.dump(discon, open(str('discon' + filename), 'wb'), protocol=4)
            pickle.dump(logSNR_sum, open(str('logSNR_sum' + filename), 'wb'), protocol=4)

        # Channel capacity - distance failure
        if NoReallocation:
            ax3.plot(distances,
                     [fm.channel_capacity_dist(k, labdaU, labdaBS, rmax, c, alpha, total_bandwidth) for rmax in
                      distances],
                     color=colors[k - 1])
            ax3.scatter(distances, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))

        else:
            ax3.scatter(distances, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))
        if k == 1:
            channel_sum_benchmarkDIST = np.array(channel_sum)
        diff = (np.array(channel_sum) - channel_sum_benchmarkDIST) / channel_sum_benchmarkDIST * 100
        print('Distance: For k =', k, 'the difference in performance is max', max(diff), 'and min', min(diff))
#         print(diff)

        # Outage probability - distance failure
        if NoReallocation:
            ax4.plot(distances, [fm.outage_probability_deterministic_distance(labdaBS, rmax) for rmax in distances],
                     color=colors[k - 1])
            ax4.scatter(distances, discon, color=colors[k - 1], marker=markers[k - 1],
                        label=str('$k = $ ' + str(k)))

        print('Deterministic distance failure for k =', k, "done")

    if Random_Failure:
        discon = []
        probabilities = np.arange(0, 1, 1 / delta)

        if NoReallocation:
            filename = str(name + 'random' + str(k) + str(delta) + '.p')
        else:
            filename = str('reallocation' + name + 'random' + str(k) + str(delta) + '.p')

        channel_sum = from_memory(str('channel_sum' + filename))
        discon = from_memory(str('discon' + filename))
        logSNR_sum = from_memory(str('logSNR_sum' + filename))
        if logSNR_sum == None or channel_sum == None or discon == None:
            channel_sum = list()
            logSNR_sum = list()
            discon = []
            for p in probabilities:
                Gnew = random_failure(G, p)
                if NoReallocation:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                else:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, Gnew, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                discon.append(number_of_disconnected_users(Gnew, pointsBS) / pointsPop)
                channel_sum.append(average(channel))
                logSNR_sum.append(average(logSNR))

            pickle.dump(channel_sum, open(str('channel_sum' + filename), 'wb'), protocol=4)
            pickle.dump(discon, open(str('discon' + filename), 'wb'), protocol=4)
            pickle.dump(logSNR_sum, open(str('logSNR_sum' + filename), 'wb'), protocol=4)

        # Channel capacity - distance failure proportional
        if NoReallocation:
            ax7.plot(probabilities,
                     [fm.channel_capacity(k, labdaU, labdaBS, [p] * k, c, alpha, total_bandwidth) for p in
                      probabilities],
                     color=colors[k - 1])
            ax7.scatter(probabilities, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))
        else:
            ax7.scatter(probabilities, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))
        if k == 1:
            channel_sum_benchmarkRAND = np.array(channel_sum)
        diff = (np.array(channel_sum) - channel_sum_benchmarkRAND) / channel_sum_benchmarkRAND * 100
        print('Random: For k =', k, 'the difference in performance is max', max(diff), 'and min', min(diff))
#         print(diff)
        # Outage probability - distance failure proportional
        if NoReallocation:
            ax8.plot(probabilities, [p ** k for p in probabilities], color=colors[k - 1])
            ax8.scatter(probabilities, discon, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))

        print('Random failure for k =', k, "done")

    if LoS_Failure:
        discon = []
        distances = np.arange(1, 15, 15 / delta)

        if NoReallocation:
            filename = str(name + 'los' + str(k) + str(delta) + '.p')
        else:
            filename = str('reallocation' + name + 'los' + str(k) + str(delta) + '.p')

        channel_sum = from_memory(str('channel_sum' + filename))
        discon = from_memory(str('discon' + filename))
        logSNR_sum = from_memory(str('logSNR_sum' + filename))
        if logSNR_sum == None or channel_sum == None or discon == None or 3 == 3:
            channel_sum = list()
            logSNR_sum = list()
            discon = []
            for rlos in distances:
                Gnew = los_failure(G, xbs, ybs, xpop, ypop, rlos)
                if NoReallocation:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, G, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)
                else:
                    channel, logSNR, W = find_all(pointsPop, pointsBS, Gnew, Gnew, labdaBS, xbs, ybs, xpop, ypop,
                                                  total_bandwidth, k)

                discon.append(number_of_disconnected_users(Gnew, pointsBS) / pointsPop)
                channel_sum.append(average(channel))
                logSNR_sum.append(average(logSNR))
            pickle.dump(channel_sum, open(str('channel_sum' + filename), 'wb'), protocol=4)
            pickle.dump(discon, open(str('discon' + filename), 'wb'), protocol=4)
            pickle.dump(logSNR_sum, open(str('logSNR_sum' + filename), 'wb'), protocol=4)

        # Channel capacity
        if NoReallocation:
            ax9.plot(distances, [fm.channel_capacity_los(k, labdaU, labdaBS, c, alpha, total_bandwidth, rlos) for
                                 rlos in distances], color=colors[k - 1])
            ax9.scatter(distances, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))

        else:
            ax9.scatter(distances, channel_sum, marker=markers[k - 1], color=colors[k - 1],
                        label=str('$k = $ ' + str(k)))
        if k == 1:
            channel_sum_benchmarkLOS = np.array(channel_sum)
        diff = (np.array(channel_sum) - channel_sum_benchmarkLOS) / channel_sum_benchmarkLOS * 100
        print('LoS: For k =', k, 'the difference in performance is max', max(diff), 'and min', min(diff))
#         print(diff)
        # Outage probability
        if NoReallocation:
            ax10.plot(distances, [fm.outage_probability_LoS(labdaBS, k, rlos) for rlos in distances],
                      color=colors[k - 1])
            ax10.scatter(distances, discon, marker=markers[k - 1], color=colors[k - 1],
                         label=str('$k = $ ' + str(k)))

        print('LoS failure for k =', k, "done")


def find_fairness(list):
    return sum(list) ** 2 / (len(list) * sum([x ** 2 for x in list]))


for k in lijstk:
    if No_Failure:
        channel_sum = list()
        logSNR_sum = []
        W_sum = []
        fairness = []
        lbs = np.arange(1e-4, 1e-2, 1e-2 / delta)
        filename = str(name + 'nofailure' + str(k) + '.p')

        channel_sum = from_memory(str('channel_sum' + filename))
        if channel_sum == None:
            channel_sum = list()
            for labdaBS in lbs:
                pointsBS = labdaBS * (yDelta * xDelta)
                pointsBS = int(pointsBS)
                labdaBS = pointsBS / (yDelta * xDelta)
                if os.path.exists(
                        str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p')) and Save:
                    print('The graph for k =', k, 'is already stored in memory')
                    G = pickle.load(
                        open(str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'),
                             'rb'))
                    pointsPop = pickle.load(
                        open(str(
                            'Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                                k) + '.p'),
                            'rb'))
                    pointsBS = pickle.load(
                        open(str(
                            'Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                                k) + '.p'),
                            'rb'))
                    xbs = pickle.load(
                        open(str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'),
                             'rb'))
                    ybs = pickle.load(
                        open(str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'),
                             'rb'))
                    xpop = pickle.load(
                        open(str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'),
                             'rb'))
                    ypop = pickle.load(
                        open(str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'),
                             'rb'))
                else:
                    print(labdaBS)
                    xbs, ybs = initialise_graph(pointsBS)
                    G, colorlist, nodesize = make_graph(xbs, ybs, xpop, ypop, pointsBS, pointsPop)

                    for node in range(pointsPop):
                        bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
                        for bs in bss:
                            G.add_edge(node + pointsBS, bs)

                    pickle.dump(G, open(
                        str('Simulations/graph' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                        'wb'), protocol=4)
                    pickle.dump(pointsBS,
                                open(str('Simulations/pointsBS' + str(labdaBS) + str(labdaU) + str(alpha) + str(
                                    xMax) + str(
                                    k) + '.p'), 'wb'), protocol=4)
                    pickle.dump(pointsPop, open(
                        str('Simulations/pointsPop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(
                            k) + '.p'), 'wb'), protocol=4)
                    pickle.dump(xbs, open(
                        str('Simulations/xbs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                        'wb'), protocol=4)
                    pickle.dump(ybs, open(
                        str('Simulations/ybs' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                        'wb'), protocol=4)
                    pickle.dump(xpop, open(
                        str('Simulations/xpop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                        'wb'), protocol=4)
                    pickle.dump(ypop, open(
                        str('Simulations/ypop' + str(labdaBS) + str(labdaU) + str(alpha) + str(xMax) + str(k) + '.p'),
                        'wb'), protocol=4)

                channel, logSNR, W = find_all(pointsPop, pointsBS, G, G, labdaBS, xbs, ybs, xpop, ypop, total_bandwidth,
                                              k)
                logSNR_sum.append(average(logSNR))
                W_sum.append(average_nonempty(W))
                channel_sum.append(average(channel))
                fairness.append(find_fairness(channel))
                print(fairness)
            filename = str(name + 'nofailure' + str(delta) + '.p')
            pickle.dump(channel_sum, open(str('channel_sum' + filename), 'wb'), protocol=4)
            pickle.dump(fairness, open(str('fairness' + filename), 'wb'), protocol=4)

        ax11.plot(lbs, [fm.channel_capacity(k, labdaU, labdaBS, [0] * k, c, alpha, total_bandwidth) for labdaBS in lbs],
                  color=colors[k - 1])
        ax11.scatter(lbs, channel_sum, color=colors[k - 1], marker=markers[k - 1], label=str('k = ' + str(k)),
                )
        # ax11.plot(lbs, channel_sum, color=colors[k - 1])

if Overload_Failure:
    ax1.legend()
    ax1.set_ylabel("Channel capacity")
    ax1.set_xlabel('$\\beta$')
    # ax1.set_title('Overload Failure')
    fig1.savefig(str(name + 'overload' + 'capacity' + '.png'))

    if NoReallocation:
        ax2.legend()
        ax2.set_ylabel("Outage probability")
        ax2.set_xlabel('$\\beta$')
        # ax2.set_title('Overload Failure')
        fig2.savefig(str(name + 'overload' + 'outage' + '.png'))

if Distance_Failures:
    ax3.legend()
    ax3.set_ylabel("Channel capacity")
    ax3.set_xlabel('$r_{max}$')
    # ax3.set_title('Distance Failure')
    fig3.savefig(str(name + 'distance' + 'capacity' + '.png'))

    if NoReallocation:
        ax4.legend()
        ax4.set_ylabel("Outage probability")
        ax4.set_xlabel('$r_{max}$')
        # ax4.set_title('Distance Failure')
        fig4.savefig(str(name + 'distance' + 'outage' + '.png'))

if Random_Failure:
    ax7.legend()
    ax7.set_ylabel("Channel capacity")
    ax7.set_xlabel('$p$')
    # ax7.set_title('Random Failure')
    fig7.savefig(str(name + 'random' + 'capacity' + '.png'))

    if NoReallocation:
        ax8.legend()
        ax8.set_ylabel("Outage probability")
        ax8.set_xlabel('$p$')
        # ax8.set_title('Random Failure')
        fig8.savefig(str(name + 'random' + 'outage' + '.png'))

if LoS_Failure:
    ax9.legend()
    ax9.set_ylabel("Channel capacity")
    ax9.set_xlabel('$r_{LoS}$')
    # ax9.set_title('LoS Failure')
    fig9.savefig(str(name + 'LoS' + 'capacity' + '.png'))

    if NoReallocation:
        ax10.legend()
        ax10.set_ylabel("Outage probability")
        ax10.set_xlabel('$r_{LoS}$')
        # ax10.set_title('LoS Failure')
        fig10.savefig(str(name + 'LoS' + 'outage' + '.png'))

if No_Failure:
    ax11.legend()
    ax11.set_ylabel("Channel capacity")
    ax11.set_xlabel('$\lambda_{BS}$')
    # ax11.set_title('No Failure')

    ax11.set_yscale('log')
    ax11.set_xticks([0.0001, 0.002, 0.004, 0.006, 0.008, 0.010])
    fig11.savefig(str(name + 'capacity' + '.png'))

# plt.show()

fig, ax = plt.subplots()
lbs = np.arange(1e-4, 1e-2, 1e-2 / delta)
k1 = [0.44914349341472953, 0.42629911552411454, 0.5146280181544932, 0.43902251017588756, 0.4993766388450689,
      0.5234588027390966, 0.5287443538375828, 0.5198092190646557, 0.5038885381660926, 0.5108676174792246,
      0.5174753179018707, 0.522793014531818, 0.5243563643663222, 0.5240357497071884, 0.5039989842104989,
      0.5236183663977139, 0.5223402136349939, 0.511941592576615, 0.5178191509776288, 0.5171434302019504,
      0.5228786491791767, 0.5198138278404151, 0.5289894109069477, 0.5149905213676083, 0.523668584467515]
k2 = [0.5118968089050481, 0.6701107151417457, 0.6923121921419298, 0.7070633188948818, 0.7206889170433542,
      0.72896579697329, 0.7335081243929971, 0.7419266553640254, 0.74169749193286, 0.749423241647495, 0.7488968348911206,
      0.7493768086863947, 0.7437324901757515, 0.7563130109768677, 0.7544784987415226, 0.7611253478872315,
      0.7549071210859647, 0.7465633840967681, 0.7472035724798755, 0.7527745368665658, 0.7532958237710919,
      0.7462255747204636, 0.746541427725401, 0.7500089086151838, 0.7472010476288471]
k3 = [0.5709490224078269, 0.7277550919188773, 0.7459768435356736, 0.780059162728499, 0.798147126460036,
      0.7976483040832294, 0.794898008218014, 0.8078269160293172, 0.8034180340546256, 0.8162906726661474,
      0.8140765149184107, 0.819906167818684, 0.8182651617461341, 0.8183361128893134, 0.8289279501548658,
      0.824117320014148, 0.83082105770366, 0.823813354100575, 0.826001192857565, 0.8260221364106115, 0.8304129502583462,
      0.8293173207823684, 0.8240474255831186, 0.828853985953923, 0.8291707553133483]
k4 = [0.6853094575990409, 0.7864821757148724, 0.8049729331625785, 0.8276350204559219, 0.8316008175307986,
      0.8377263256026447, 0.8459139111887478, 0.8494023407934568, 0.8489130426209588, 0.8397117350745225,
      0.8561737680718134, 0.861924773310516, 0.855961141658236, 0.8606753536826769, 0.8628157641969849,
      0.8657435835080036, 0.860367320552564, 0.8688321397029255, 0.8550354660376561, 0.8600184816689006,
      0.8660286305248142, 0.8667060072411875, 0.8675285112454679, 0.8648474308943254, 0.8631534316041876]
k5 = [0.628232177720139, 0.7759739573154878, 0.8299636592365099, 0.8321862876548252, 0.8497649971485974,
      0.8608049230794947, 0.869163228475955, 0.8717945856924654, 0.8737947621971774, 0.879458010102488,
      0.8708826646725009, 0.8789142054018334, 0.8802192471758262, 0.8846239844531641, 0.8803052688055956,
      0.8887812659178653, 0.8818298679336207, 0.8807080707742176, 0.8847426651346721, 0.8884247068263884,
      0.890695185579052, 0.8886820083901626, 0.8921485137846353, 0.886714901223072, 0.885400283823729]

plt.plot(lbs, k1, color=colors[0], label='k = 1')
plt.plot(lbs, k2, color=colors[1], label='k = 2')
plt.plot(lbs, k3, color=colors[2], label='k = 3')
plt.plot(lbs, k4, color=colors[3], label='k = 4')
plt.plot(lbs, k5, color=colors[4], label='k = 5')

plt.xlabel('$\lambda_{BS}$')
plt.ylabel('Fairness')
plt.legend()
plt.show()