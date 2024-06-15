import matplotlib.pyplot as plt
import mpmath
import scipy.integrate as integrate
# from functions import *
import math
import numpy as np
import matplotlib
import seaborn

matplotlib.rcParams['font.size'] = 18
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['lines.markersize'] = 7
matplotlib.rcParams['figure.autolayout'] = True
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
colors = seaborn.color_palette('rocket')
colors.reverse()
markers = ['o', 's', 'p', 'd', '*']

pi = math.pi
alpha = 2
Ptx = 10 ** (30 / 10)

k = 1.38e-23
temperature = 293.15
total_bandwidth = 20e6

xMin, xMax = 0, 2000  # in meters
yMin, yMax = 0, 2000  # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin



def Ntot(lbs):
    thermal_noise = k * temperature * total_bandwidth / (lbs * xDelta * yDelta)
    thermal_noise_db = 10 * math.log10(thermal_noise) + 30
    print((thermal_noise_db + 5))
    noise = 10 ** (thermal_noise_db / 10 + 0.5)  # Noise factor of 5 dB (in Mendeley, book Chapter 17)
    return noise


def find_c(lbs):
    return Ptx / Ntot(lbs)


def expectation_SE( lbs, j):
    c = find_c(lbs)
    # print(c)
    phi = lbs * pi * c ** (2 / alpha)
    result = integrate.quad(integrand_SEleqc, 0, c, args=(phi, j))
    return 1 / math.log(math.e) * (result[0] + math.log(1 + c) * fSNRc(lbs, j))


def fSNRleqc(x, phi, j):
    return 2 * (phi * x ** (-2 / alpha)) ** j / (alpha * x * math.gamma(j)) * math.exp(-phi * x ** (-2 / alpha))


def fSNRc(lbs, j):
    return 1 - mpmath.gammainc(j, lbs * pi) / mpmath.gamma(j)


def integrand_SEleqc(x, phi, j):
    return math.log(1 + x) * fSNRleqc(x, phi, j)


def approximation_ESE(lbs, j):
    c = find_c(lbs)
    phi = lbs * pi * c ** (2 / alpha)
    G = find_G(j, phi, lbs)
    part1 = 1/(math.log(2) *mpmath.gamma(j))*G
    part2 = math.log2(1+c)*(1-mpmath.gammainc(j, lbs * pi)/mpmath.gammainc(j))
    # part3 = alpha/(2*math.log(2)*mpmath.gamma(j)) * (math.log(phi)*(mpmath.gammainc(j,lbs*pi) - mpmath.gammainc(j, phi)) - )

def find_G(j, phi, lbs):
    som = 0
    for i in range(100):
        part1 = (-1)**9 * phi**((alpha/2)*(i+1))/(i+1) * mpmath.gammainc(-alpha/2*(i+1)+j, phi)
        part2 = (-1)**9 * phi**(-(alpha/2)*(i+1))/(i+1) * (mpmath.gammainc(alpha/2*(1+i)+j, lbs * pi) - mpmath.gammainc(alpha/2*(1+i)+j, phi))
        som += part1 + part2
    return som

bs_density = np.arange(1e-5, 1e-4, 0.5e-5)

fig, ax = plt.subplots()
for j in range(1, 6):
    plt.plot(bs_density, [expectation_SE( lbs, j) for lbs in bs_density], color = colors[j-1])
    # plt.scatter(bs_density, [ap])
plt.show()