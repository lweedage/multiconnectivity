from __future__ import division

import matplotlib as mp

mp.use('Agg')
from matplotlib import pyplot
# from scipy.stats.kde import gaussian_kde
import numpy as np
import seaborn
import matplotlib

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['legend.fontsize'] = 18  # using a size in points
matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
# matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
markers = ['o', 's', 'p', 'd', '*']

colors = seaborn.color_palette('rocket')
colors.reverse()


def get_color(i):
    colors = seaborn.color_palette('rocket')
    colors.reverse()
    return colors[i % len(colors)]


def get_marker(m):
    markers = ["d", "o", "s", "*", "p", 'H', '8', "v", (5, 2), "D", "<", ">", "x", 'o', 'v', '^', '<', '>', 's', 'p',
               ".",
               '*', 'h',
               'D', 'd']
    return markers[m % len(markers)]


def get_linestyle(ls, or_none=False):
    linest = ['-', '--', ':', '-.', '--']
    if or_none is True:
        return 'None'
    return linest[ls % len(linest)]


def plot_ecdf_lists(A, inpname, xlab="$C^k_{sum}$", ylab="CDF", labels="", fontscale=2, scale=1, legendpos="in",
                    main="",
                    lloc=4, out_style=[".png"], dim=[7, 5], grid_on=False, legendborderspacing=1, FRAMEON=True,
                    x_lim_values=(0, 1e6), LEGEND_X_POS=1.7):
    # A is a list of lists and we will plot the ecdf of all these lists in the same plot
    # inpname is the name of the figure file to save the plotted figure
    # example usage: plot_ecdf_lists(linkSNRecdf, "linkECDF", xlab="Link SNR", labels=["client1", "client-2"])
    markers = ['o', 's', 'p', 'd', '*']
    labels = ['$k = 1$', '$k = 2$', '$k = 3$', '$k = 4$', '$k = 5$']
    afont = {'fontname': 'serif'}  # 'Arial'
    all_nan = True
    # pyplot.figure(figsize=(dim[0], dim[1]))
    allelements = []
    for sub_list_i in A:
        for k in sub_list_i:
            allelements.append(k)

    if x_lim_values:
        range_values = x_lim_values[1] - x_lim_values[0]
        min_value = x_lim_values[0]
        max_value = x_lim_values[1]
    else:
        min_value = np.nanmin(allelements)
        max_value = np.nanmax(allelements)
        range_values = max_value - min_value

    pyplot.xlim(xmin=min_value - 100, xmax=2e5 + 0.01*range_values)  # max_value + range_values * 0.1)
    pyplot.ylim((-0.1, 1.1))
    ymin, ymax = pyplot.ylim()
    np.percentile_list = [25, 50, 75, 95]
    for i in range(0, len(A)):
        print(i)
        Asub = A[i]
        b = 1. * np.arange(len(Asub)) / (len(Asub) - 1)
        np.sorted_data = np.sort(Asub)

        np.percentile_points = np.percentile(np.sorted_data, np.percentile_list)
        y_values = [float(k / 100) for k in np.percentile_list]
        pyplot.plot(np.sorted_data, b, color=get_color(i))
        pyplot.plot(np.percentile_points, y_values, color=get_color(i), linestyle='None',
                    marker=markers[i], label=labels[i])
        if legendpos == "out":
            pyplot.legend(bbox_to_anchor=(LEGEND_X_POS, 1), borderaxespad=0., frameon=FRAMEON, numpoints=1)

        else:
            pyplot.legend(borderaxespad=legendborderspacing, loc=lloc, frameon=FRAMEON, numpoints=1)
        ax = pyplot.gca()
        if main != "":
            ax.set_title(main, y=1.0)

    pyplot.ylim((ymin, ymax))
    # pyplot.tick_params(axis='both', which='major', labelsize=20)
    pyplot.xlabel(xlab)  # , fontsize=20, **afont)
    pyplot.ylabel(ylab)  # , fontsize=20, **afont)
    # ax = pyplot.gca()
    # ax.grid(grid_on)
    # ax.set_facecolor((1, 1, 1))
    # ax.set_xticklabels(ax.get_xticks(), **afont)
    # ax.set_yticklabels(ax.get_yticks(), **afont)
    for os in out_style:
        pyplot.savefig(inpname + os)  # , bbox_inches='tight')
    pyplot.close(pyplot.gcf())
