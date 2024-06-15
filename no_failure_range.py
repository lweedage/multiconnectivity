import time

import matplotlib.pyplot as plt

import graph_functions as g
from functions import *

total_channel = np.zeros((5, number_of_iterations, delta))
total_discon = np.zeros((5, number_of_iterations, delta))


name = str('lu=' + str(labdaU) + 'area=' + str(xDelta))

if not SNR:
    name = str('SINR_' + name)

# total_channel = pickle.load(open(
#     str('MC/total_channel_nofailure_iterations=' + str(number_of_iterations) + 'lbsrange=' + str(min(x_values)) + str(
#         max(x_values)) + name + '.p'), 'rb'))
total_channel = None
if total_channel is None:
    total_channel = dict()
    for iteration in range(number_of_iterations):
        print('Iteration ', iteration)
        np.random.seed(iteration)
        start = time.time()

        for k in [1, 2, 3, 4, 5]:
            channel_sum = []
            for labdaBS in x_values:
                thermal_noise = knoise * temperature * total_bandwidth / (labdaBS * xDelta * yDelta)
                thermal_noise_db = 10 * math.log10(thermal_noise) + 30
                noise = 10 ** (thermal_noise_db / 10 + 0.5)  # Noise power of 5 dB (in Mendeley, book Chapter 17)

                number_of_BS = int(xDelta * yDelta * labdaBS)

                xbs, ybs = initialise_BSs(number_of_BS)
                xpop, ypop = initialise_users(number_of_users)

                G, colorlist, nodesize = g.make_graph(xbs, ybs, xpop, ypop, number_of_BS, number_of_users)

                name = str('MC/Graphs/G_lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(xDelta) + 'k=' + str(
                    k) + 'iteration=' + str(iteration) + '.p')

                if os.path.exists(name):
                    G = pickle.load(open(name, 'rb'))
                else:
                    print('Graph does not exist yet')
                    for node in range(number_of_users):
                        bss = find_bs(xpop[node], ypop[node], xbs, ybs, k)
                        for bs in bss:
                            G.add_edge(node + number_of_BS, bs)

                    pickle.dump(G, open(name, 'wb'), protocol=4)

                channel = find_channel(G, G, xbs, ybs, xpop, ypop, noise)
                channel_sum.append(average(channel))

            total_channel[k - 1, iteration] = channel_sum
        print('Iteration ', iteration, 'in ', time.time() - start, 'seconds')

    name = str('lu=' + str(labdaU) + 'area=' + str(xDelta))

    if not SNR:
        name = str('SINR_' + name)

    pickle.dump(total_channel, open(
        str('MC/total_channel_nofailure_iterations=' + str(number_of_iterations) + 'lbsrange=' + str(min(x_values)) + str(
            max(x_values)) + name + '.p'), 'wb'), protocol=4)

channel_sum = total_channel
print(channel_sum)
print(x_values)

fig, ax1 = plt.subplots()
for k in [1, 2, 3, 4, 5]:
    channel_boxplot = np.zeros((delta, number_of_iterations))

    for iteration in range(number_of_iterations):
        for i in range(delta):
            channel_boxplot[i, iteration] = channel_sum[(k-1, iteration)][i]

    channel_avg = [sum(channel_boxplot[i]) / number_of_iterations for i in range(delta)]

    lower = math.floor(0.025 * number_of_iterations)
    upper = number_of_iterations - math.floor(0.025 * number_of_iterations)

    channel = np.zeros((delta, (upper - lower)))

    for i in range(delta):
        data = np.sort(channel_boxplot[i])
        channel[i, :] = data[lower:upper]

    channel_min = [min(channel[i]) for i in range(delta)]
    channel_max = [max(channel[i]) for i in range(delta)]

    print(channel_avg[1])

    ax1.scatter(x_values, channel_avg, label=f'$k = {k}$', color=colors[k - 1], marker=markers[k - 1],
                facecolors='none')

    ax1.errorbar(x_values, channel_avg,
                 yerr=[np.subtract(channel_avg, channel_min), np.subtract(channel_max, channel_avg)],
                 color=colors[k - 1])  # , marker = markers[k-1])

ax1.legend()

ax1.set_ylabel('Channel capacity (bps)')
ax1.set_xlabel('$\lambda_{BS}$')
# fig.savefig(str(name + 'channel.png'))
plt.show()
