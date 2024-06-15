import math

import numpy as np
import scipy.stats

SNR = True
Reallocation = True
Draw = False

OBJ = int(input('Objective?'))
SNR = int(input('SNR?'))
Reallocation = int(input('Re-allocation?'))

bandwidth_split = 1


def initialise_BSs(number_of_BS):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((number_of_BS, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((number_of_BS, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(number_of_BS)]
    y = [yy[i][0] for i in range(number_of_BS)]
    return x, y


def initialise_users(number_of_users):
    xx = xDelta * scipy.stats.uniform.rvs(0, 1, ((number_of_users, 1))) + xMin  # x coordinates of Poisson points
    yy = yDelta * scipy.stats.uniform.rvs(0, 1, ((number_of_users, 1))) + yMin  # y coordinates of Poisson points

    x = [xx[i][0] for i in range(number_of_users)]
    y = [yy[i][0] for i in range(number_of_users)]

    return x, y


pi = math.pi

a = [3.5, 7.2, 11.1, 15.2, 21.2]

xMin, xMax = 0, 1500  # in meters
yMin, yMax = 0, 1500  # in meters

xDelta = xMax - xMin
yDelta = yMax - yMin

if SNR:
    labdaBS = 5e-5
else:
    labdaBS = 5e-6  # = how many BS per square meter
    # labdaBS = 5e-5  # = how many BS per square meter

labdaU = 500e-6  # = how many users per square meter

total_bandwidth = 20e6  # we use 20 MHz bandwidth
alpha = 2  # Path loss exponent, between 2 and 4.
ptx = 10 ** (3)  # Transmission power of a BS ranges between 24 and 36 dB

number_of_BS = int(labdaBS * (yDelta * xDelta))
number_of_users = int(labdaU * (yDelta * xDelta))

knoise = 1.38e-23
temperature = 293.15




RandomFailure = 0
DistanceFailure = 1
LoSFailure = 2
NoFailure = 3
RandomFailureLINK = 4

# OBJ = RandomFailure
# OBJ = DistanceFailure
# OBJ = LoSFailure
# OBJ = NoFailure
# OBJ = RandomFailureLINK

delta = 10  # number of timesteps

if OBJ == RandomFailureLINK:
    min_x, max_x = 0, 0.75
elif OBJ == DistanceFailure:
    min_x, max_x = 50, 250
elif OBJ == LoSFailure:
    min_x, max_x = 10, 150
elif OBJ == NoFailure:
    min_x, max_x = 50e-7, 50e-5
elif OBJ == RandomFailure:
    min_x, max_x = 0, 0.5

x_values = np.arange(min_x, max_x, (max_x - min_x) / delta)

number_of_iterations = 100

print('Objective = ', OBJ)
