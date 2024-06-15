import matplotlib.pyplot as plt
import mpmath
import scipy.stats
import numpy as np
import networkx as nx
import math
from scipy.special import gamma, loggamma
from scipy.integrate import quad
from mpmath import gammainc
from parameters import *
from functions import *

pi = math.pi

a = [3.527, 7.187, 11.062, 15.212, 21.166]
b = [3.527, 7.187, 11.062, 15.212, 21.166]


# ------------------- SNR and channel capacity ----------------------------------
def find_expectation_log_SNR_numerical(k, alpha, lbs, c):
    I = quad(integrand, 0, c, args=(k, alpha, lbs * pi * c ** (2 / alpha)))
    return I[0] + math.log2(1 + c) * (1 - gammainc(k, lbs * pi) / gamma(k))


def integrand(x, k, alpha, phi):
    return math.log2(1 + x) * 2 * (phi * x ** (-2 / alpha)) ** k / (alpha * x * gamma(k)) * math.exp(
        -phi * x ** (-2 / alpha))


def expected_w(lu, wtot, k):
    return wtot / (k * lu * (xDelta * yDelta))


def channel_capacity(k, lu, lbs, p, c, alpha, wtot):
    som = 0
    for j in range(1, k + 1):
        som += (1 - p[j - 1]) * find_expectation_log_SNR_numerical(j, alpha, lbs, c)
    W = expected_w(lu, wtot, k)
    return som * W


# ------------------- RANDOM FAILURE------------------------------------------
def find_channel_random(k, alpha, lbs, lu, c, p, wtot):
    som = 0
    for j in range(k):
        I = quad(integrand, 0, c, args=(j + 1, alpha, lbs * pi * c ** (2 / alpha)))
        som += (1 - p) * (I[0] + math.log2(1 + c) * (1 - mpmath.gammainc(j + 1, lbs * pi) / gamma(j)))
    return wtot /(k * lu * (xDelta * yDelta)) * som


def find_probability_random(k, p):
    return [p] * k


# ------------------ DETERMINISTIC DISTANCE FAILURE ----------------------------------
def channel_capacity_dist(k, lu, lbs, rmax, c, alpha, wtot):  # for deterministic distance failures
    som = 0
    for j in range(1, k + 1):
        som += find_expectation_log_SNR_numerical_dist(j, alpha, lbs, c, rmax)
    return expected_w(lu, wtot, k) * som


def find_expectation_log_SNR_numerical_dist(k, alpha, lbs, c, rmax):
    I = quad(integrand, c * rmax ** (-alpha), c, args=(k, alpha, lbs * pi * c ** (2 / alpha)))
    return I[0] + math.log2(1 + c) * (1 - mpmath.gammainc(k, lbs * pi) / math.factorial(k - 1))


def integrand_dist(x, lbs, k, c, alpha):
    return math.log2(1 + c * x ** (-alpha)) * 2 * (lbs * pi * x ** 2) ** k / (x * gamma(k)) * math.exp(
        -lbs * pi * x ** 2)


def find_probability_distance(k, lbs, rmax):
    p = np.zeros(k)
    for j in range(1, k + 1):
        p[j - 1] = scipy.special.gammaincc(j, lbs * pi * rmax ** 2)
    return p


def outage_probability_distance(lbs, rmax):
    return math.exp(-lbs * pi * rmax ** 2)


# -------------- LINE OF SIGHT FAILURE -------------------------------------------

def channel_capacity_los(k, lu, lbs, c, alpha, wtot, rlos):
    som = 0
    for j in range(1, k + 1):
        som += find_expectation_log_SNR_los(j, alpha, lbs, c, rlos)
    W = expected_w(lu, wtot, k)
    return som * W


def find_expectation_log_SNR_los(j, alpha, lbs, c, rlos):
    I1 = quad(integrand_los1, 1, rlos, args=(j, alpha, lbs, c))
    I2 = quad(integrand_los2, rlos, math.inf, args=(j, alpha, lbs, c, rlos))
    return (I1[0] + math.log2(1 + c) * (1 - mpmath.gammainc(j, lbs * pi) / gamma(j))) + I2[0]


def integrand_los1(x, k, alpha, lbs, c):
    return math.log2(1 + c * x ** (-alpha)) * 2 * (lbs * pi * x ** 2) ** k / (x * gamma(k)) * math.exp(
        -lbs * pi * x ** 2)


def integrand_los2(x, k, alpha, lbs, c, rlos):
    return (x ** (-1) * (rlos + x * math.exp(-x / (2 * rlos)) - rlos * math.exp(-x / (2 * rlos)))) * math.log2(
        1 + c * x ** (-alpha)) * 2 * (lbs * pi * x ** 2) ** k / (x * gamma(k)) * math.exp(-lbs * pi * x ** 2)


def find_probability_LoS(k, lbs, rlos):
    p = np.zeros(k)
    for j in range(1, k + 1):
        I = quad(integrand_los_outage, rlos, math.inf, args=(j, lbs, rlos))
        # p[j - 1] = mpmath.gammainc(j, lbs * pi * 18**2)/gamma(j) - I[0]
        p[j - 1] = I[0]
    return p


def integrand_los_outage(x, j, lbs, rlos):
    return (1 - x ** (-1) * (rlos + x * math.exp(-x / (2 * rlos)) - rlos * math.exp(-x / (rlos * 2)))) * 2 * (
                lbs * pi * x ** 2) ** j / (x * gamma(j)) * math.exp(-lbs * pi * x ** 2)


def outage_probability_LoS(lbs, k, rlos):
    prod = 1
    p = find_probability_LoS(k, lbs, rlos)
    for j in range(k):
        prod = prod * p[j]
    return prod


# ---------------- APPROXIMATIONS ---------------------------
def I3approx_with_R(k, lbs, alpha, c):
    pi = math.pi
    phi = lbs * pi * c ** (2 / alpha)
    if k == 1:
        return 1 / math.log(2) * (
                    math.log(c) + alpha / (2) * (-phi * math.exp(-phi) + math.log(lbs * pi) + mpmath.euler - lbs * pi))
    else:
        return 1 / math.log(2) * (
                    alpha * phi ** k / (2 * k ** 2 * math.factorial(k - 1)) * mpmath.hyp2f2(k, k, k + 1, k + 1, -phi))


def approximate_expected_w(lu, wtot, k):
    return wtot / (k * lu)


def approximate_channel_capacity(k, lu, lbs, p, c, alpha, wtot):
    som = 0
    labda = lu / lbs
    for j in range(1, k + 1):
        som += (1 - p[j - 1]) * I3approx_with_R(j, lbs, alpha, c)
    W = expected_w(lbs, labda, wtot, k)
    return som * W


def fsnr(x, c, j, alpha, labdaBS):
    x = 10**(x/10)
    phi = labdaBS * math.pi * c**(2/alpha)
    if x == 0:
        return 0
    elif x < c:
        return 2*(phi * x**(-2/alpha))**j / (alpha * x * math.gamma(j)) * math.exp(-phi* x**(-2/alpha))
    elif x == c:
        return 1 - mpmath.gammainc(j, labdaBS * pi, regularized = True)
    else:
        return 0

if __name__ == "__main__":
    delta = 20
    x_values = np.arange(0, 40, 0.1)

    labdaBS = 5e-5
    c = 10**6
    alpha = 2
    for k in [1, 2 ,3, 4, 5]:
        plt.plot(x_values, [fsnr(x, c, k, alpha, labdaBS) for x in x_values], label=f'$j = {k}$', color=colors[k - 1])

    # plt.ylim((0, 0.02))
    plt.ylabel('PDF')
    plt.xlabel('SNR$_j$ (dB)')
    plt.legend()
    plt.show()

    x_values = np.arange(1e-6, 1e-3, 1e-3/15)


    for k in [1, 2, 3, 4, 5]:
        plt.scatter(x_values, [I3approx_with_R(k, x, alpha, c) for x in x_values], label = f'$j = {k}$', color = colors[k-1], marker = markers[k-1], facecolor = 'None')
        plt.plot(x_values, [find_expectation_log_SNR_numerical(k, alpha, x, c) for x in x_values], color = colors[k-1])

    plt.xlabel('$\lambda_{BS}$')
    plt.ylabel('$E(\log_2(1+SNR_j))$')
    plt.legend()
    plt.show()