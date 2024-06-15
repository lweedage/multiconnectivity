import time
import graph_functions as g
from functions import *
from parameters import *

total_channel = np.zeros((5, number_of_iterations, delta))
total_discon = np.zeros((5, number_of_iterations, delta))

if Reallocation:
    name = str(
        'objective=' + str(OBJ) + 'reallocation' + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(
            xDelta))
else:
    name = str('objective=' + str(OBJ) + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(xDelta))

if not SNR:
    name = str('SINR_' + name)

if not os.path.exists(str('MC/total_channel' + str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(
    number_of_iterations) + name + '.p')):
    for iteration in range(number_of_iterations):
        print('Iteration ', iteration)
        start = time.time()
        np.random.seed(iteration)

        xbs, ybs = initialise_BSs(number_of_BS)
        xpop, ypop = initialise_users(number_of_users)

        labdaBS = len(xbs) / (yDelta * xDelta)
        labdaU = len(xpop) / (yDelta * xDelta)

        number_of_BS = len(xbs)
        graph, colorlist, nodesize = g.make_graph(xbs, ybs, xpop, ypop, number_of_BS, number_of_users)
        start = time.time()
        for k in [1, 2, 3, 4, 5]:
            graph_name = str(
                'MC/Graphs/' + 'G_lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(xDelta) + 'k=' + str(
                    k) + 'iteration=' + str(iteration) + '.p')

            if os.path.exists(graph_name):
                G = pickle.load(open(graph_name, 'rb'))
            else:
                G = graph.copy()
                for node in range(number_of_users):
                    bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
                    for bs in bss:
                        G.add_edge(node + number_of_BS, bs)
                pickle.dump(G, open(graph_name, 'wb'), protocol=4)

            if Draw:
                g.draw_graph(G, colorlist, nodesize)

            filename = str(name + 'iteration=' + str(iteration) + 'k=' + str(k) + '.p')

            thermal_noise = knoise * temperature * total_bandwidth / (labdaBS * xDelta * yDelta)
            thermal_noise_db = 10 * math.log10(thermal_noise) + 30
            noise = 10 ** (thermal_noise_db / 10 + 0.5)  # Noise power of 5 dB (in Mendeley, book Chapter 17)

            print('Finding channel capacity...')
            channel_sum, discon_sum = find_channel_outage(filename, x_values, k, G, xbs, ybs, xpop, ypop, iteration,
                                                          number_of_BS, noise)

            total_channel[k - 1, iteration] = channel_sum
            total_discon[k - 1, iteration] = discon_sum
            print(total_channel[k - 1, iteration])

        print('Iteration ', iteration, 'in ', time.time() - start, 'seconds')

pickle.dump(total_channel, open(str('MC/total_channel' + str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(
    number_of_iterations) + name + '.p'), 'wb'), protocol=4)
pickle.dump(total_discon, open(str('MC/total_discon' + str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(
    number_of_iterations) + name + '.p'), 'wb'), protocol=4)
