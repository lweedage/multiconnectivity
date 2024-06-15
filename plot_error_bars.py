from parameters import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from functions import *
import failure_models as fm

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 18 # using a size in points
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
markers = ['o', 's', 'p', 'd', '*']

colors = seaborn.color_palette('rocket')
colors.reverse()
def find_name():
    if Reallocation:
        if not SNR:
            name = str(str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(number_of_iterations) +
                       'SINR_objective=' + str(OBJ) + 'reallocation' + 'lbs=' + str(labdaBS) + 'lu=' + str(
                labdaU) + 'area=' + str(
                xDelta))

            name_no_failure = str('iterations=' + str(number_of_iterations) +
                                  'SINR_' + 'lu=' + str(
                labdaU) + 'area=' + str(
                xDelta))
        else:
            name = str(str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(number_of_iterations) +
                       'objective=' + str(OBJ) + 'reallocation' + 'lbs=' + str(labdaBS) + 'lu=' + str(
                labdaU) + 'area=' + str(
                xDelta))

            name_no_failure = str('iterations=' + str(number_of_iterations)
                                  + 'lu=' + str(                labdaU) + 'area=' + str(
                xDelta))
    else:
        if not SNR:
            name = str(str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(number_of_iterations) +
                       'SINR_objective=' + str(OBJ) + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(
                xDelta))

            name_no_failure = str('iterations=' + str(number_of_iterations) +
                                  'SINR_' + 'lu=' + str(
                labdaU) + 'area=' + str(
                xDelta))
        else:
            name = str(str(min_x) + str(max_x) + str(delta) + 'iterations=' + str(number_of_iterations) +
                       'objective=' + str(OBJ) + 'lbs=' + str(labdaBS) + 'lu=' + str(labdaU) + 'area=' + str(
                xDelta))

            name_no_failure = str('iterations=' + str(number_of_iterations) +
                                  'lu=' + str(
                labdaU) + 'area=' + str(
                xDelta))

    return name, name_no_failure


def box_plot(data, edge_color, fill_color, ax):
    bp = ax.boxplot(data, patch_artist=True, showfliers=False, positions=x_values)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color, alpha=0.3)

    return bp

data = []

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig4, ax4 = plt.subplots()

name, name_no_failure = find_name()

channel_sum = from_memory(str('MC/total_channel' + name + '.p'))
discon = from_memory(str('MC/total_discon' + name + '.p'))

channel_no_failure = from_memory(str('MC/total_channel_nofailure_' + name_no_failure + '.p'))
print(name)
# print([average(channel_no_failure[i]) for i in range(5)])
#
# print(str('MC/total_channel_nofailure_' + name_no_failure + '.p'))

for k in [1, 2, 3, 4, 5]:
    channel_boxplot = np.zeros((delta, number_of_iterations))
    discon_boxplot = np.zeros((delta, number_of_iterations))

    for iteration in range(number_of_iterations):
        for i in range(delta):
            channel_boxplot[i, iteration] = channel_sum[k - 1, iteration, i]
            discon_boxplot[i, iteration] = discon[k - 1, iteration, i]

    channel_avg = [sum(channel_boxplot[i]) / number_of_iterations for i in range(delta)]
    discon_avg = [sum(discon_boxplot[i]) / number_of_iterations for i in range(delta)]
    #
    # if k == 1:
    #     channel_avgk1 = channel_avg
    # else:
    #     difference = np.divide(np.subtract(channel_avgk1, channel_avg), channel_avgk1)
    #     print('k = ', k, ', capacity loss:', max(difference) * 100, min(difference) * 100)
    # no_failure_avg = sum(channel_no_failure[k - 1]) / number_of_iterations
    #
    # lower = math.floor(0.025 * number_of_iterations)
    # upper = number_of_iterations - math.floor(0.025 * number_of_iterations)
    #
    # channel = np.zeros((delta, (upper - lower)))
    # disconnected = np.zeros((delta, (upper - lower)))
    #
    # for i in range(delta):
    #     data = np.sort(channel_boxplot[i])
    #     channel[i, :] = data[lower:upper]
    #     discon_data = np.sort(discon_boxplot[i])
    #     disconnected[i] = discon_data[lower:upper]
    #
    # channel_min = [min(channel[i]) for i in range(delta)]
    # channel_max = [max(channel[i]) for i in range(delta)]
    #
    # discon_min = [min(disconnected[i]) for i in range(delta)]
    # discon_max = [max(disconnected[i]) for i in range(delta)]
    #
    # ax1.scatter(x_values, channel_avg, label=f'$k = {k}$', color=colors[k - 1], marker=markers[k - 1],
    #             facecolors='none')
    #
    # ax1.errorbar(x_values, channel_avg,
    #              yerr=[np.subtract(channel_avg, channel_min), np.subtract(channel_max, channel_avg)],
    #              color=colors[k - 1])  # , marker = markers[k-1])
    #
    ax2.scatter(x_values, discon_avg, label=f'$k = {k}$', color=colors[k - 1], marker=markers[k - 1])

    # ax2.errorbar(x_values, discon_avg, yerr=[np.subtract(discon_avg, discon_min), np.subtract(discon_max, discon_avg)],
    #              color=colors[k - 1], ls='none')  # , marker = markers[k-1])
    #
    #
    ax4.scatter(x_values, channel_avg, color=colors[k - 1], marker=markers[k - 1],
                label=f'$k = {k}$')

    thermal_noise = knoise * temperature * total_bandwidth / (labdaBS * xDelta * yDelta)
    thermal_noise_db = 10 * math.log10(thermal_noise) + 30
    noise = 10 ** (thermal_noise_db / 10 + 0.5)  # Noise power of 5 dB (in Mendeley, book Chapter 17)

    if OBJ == DistanceFailure:
        ax4.plot(x_values,
                 [fm.channel_capacity_dist(k, labdaU, labdaBS, rmax, ptx / noise, alpha, total_bandwidth) for rmax
                  in
                  x_values], '--', color=colors[k - 1])#, label=f'$k = {k}$ - calculated')
        ax2.plot(x_values, [fm.outage_probability_distance(labdaBS, rmax) for rmax in x_values], '--',
                 color=colors[k - 1])#, label=f'$k = {k}$ - calculated')

    if OBJ == LoSFailure:
        ax4.plot(x_values,
                 [fm.channel_capacity_los(k, labdaU, labdaBS, ptx / noise, alpha, total_bandwidth, rlos) for rlos in
                  x_values], '--', color=colors[k - 1])#, label=f'$k = {k}$ - calculated')
        ax2.plot(x_values, [fm.outage_probability_LoS(labdaBS, k, rlos) for rlos in x_values], '--',
                 color=colors[k - 1])#, label=f'$k = {k}$ - calculated')

    if OBJ == RandomFailure or OBJ == RandomFailureLINK:
        ax4.plot(x_values, [fm.channel_capacity(k, labdaU, labdaBS, [p, p, p, p, p], ptx/noise, alpha, total_bandwidth) for p in x_values], '--', color=colors[k - 1])#, label=f'$k = {k}$ - calculated')
        ax2.plot(x_values, [p ** k for p in x_values], '--', color=colors[k - 1])#, label=f'$k = {k}$ - calculated')

    if OBJ == NoFailure:
        ax4.plot(x_values, [fm.channel_capacity(k, labdaU, labdaBS, [0, 0, 0, 0, 0], ptx/noise, alpha, total_bandwidth) for labdaBS in x_values], '--', color=colors[k - 1], label=f'$k = {k}$ - calculated')


ax1.legend()
ax2.legend()
ax4.legend()

ax1.set_ylabel('Channel capacity (bps)')
ax2.set_ylabel("Outage probability")
ax4.set_ylabel('Channel capacity (bps)')


if OBJ == DistanceFailure:
    ax1.set_xlabel('$r_{max}$')
    ax2.set_xlabel('$r_{max}$')
    ax4.set_xlabel('$r_{max}$')

if OBJ == RandomFailure or OBJ == RandomFailureLINK:
    ax1.set_xlabel('$p$')
    ax2.set_xlabel('$p$')
    ax4.set_xlabel('$p$')

if OBJ == LoSFailure:
    ax1.set_xlabel('$r_{LoS}$')
    ax2.set_xlabel('$r_{LoS}$')
    ax4.set_xlabel('$r_{LoS}$')

if OBJ == NoFailure:
    ax1.set_xlabel('$\lambda_{BS}$')
    ax2.set_xlabel('$\lambda_{BS}$')
    ax4.set_xlabel('$\lambda_{BS}$')

plt.show()

if OBJ == NoFailure:
    name = 'no_failure'
elif OBJ == DistanceFailure:
    name = 'distance'
elif OBJ == LoSFailure:
    name = 'los'
elif OBJ == RandomFailureLINK :
    name = 'random'
elif OBJ == RandomFailure:
    name = 'random_bs'


if not Reallocation:
    fig2.savefig(str(name + 'outage.png'))
    if SNR:
        fig4.savefig(str(name + 'channel_calculated.png'))
else:
    name = str(name + '_reallocation')

if not SNR:
    name = str(name + '_SINR')

fig1.savefig(str(name + 'channel.png'))
