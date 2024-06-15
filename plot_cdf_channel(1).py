import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.special
import failure_models as fm
import os
import pickle
import seaborn
from parameters import *
import graph_functions as g
from functions import *
import functions as f
import time
import plot_ecdf_lists as ecdf


total_channel = np.zeros((5, number_of_iterations))
total_discon = np.zeros((5, number_of_iterations))


np.random.seed(0)
xbs, ybs = initialise_BSs(number_of_BS)
xpop, ypop = initialise_users(number_of_users)

total_channel = []

start = time.time()
for k in [1, 2, 3, 4, 5]:


    G, colorlist, nodesize = g.make_graph(xbs, ybs, xpop, ypop, number_of_BS, number_of_users)
    iteration = 1
    name = str('MC/Graphs/G_lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(xDelta) + 'k=' + str(k) + 'iteration=' + str(iteration) + '.p')

    if os.path.exists(name):
        G = pickle.load(open(name, 'rb'))
    else:
        for node in range(number_of_users):
            bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
            for bs in bss:
                G.add_edge(node + number_of_BS, bs)

        pickle.dump(G, open(name, 'wb'), protocol=4)

    channel = find_channel(G, G, xbs, ybs, xpop, ypop, noise)
    total_channel.append(channel)# channel = average(channel)
ecdf.plot_ecdf_lists(total_channel, 'cdf_channel')

