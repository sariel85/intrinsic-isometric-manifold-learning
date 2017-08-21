# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from data_generation import BoundingShape, ItoGenerator, print_process, create_color_map
import numpy
import os
import math
from observation_modes import *
import matplotlib.pyplot as plt

sim_dir_name = "2D Small Room - Static - Camera - 1"
intrinsic_test_file_name = 'intrinsic_test.txt'
sim_dir = './' + sim_dir_name
intrinsic_process_file_name = 'intrinsic_process.npy'
intrinsic_process_file = sim_dir + '/' + intrinsic_process_file_name
intrinsic_simulated_process = numpy.load(sim_dir + '/' + intrinsic_process_file_name).astype(dtype=numpy.float64).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=numpy.float64)
dist_potential = numpy.load(sim_dir + '/' + 'dist_potential.npy').astype(dtype=numpy.float64)

#path = numpy.asarray([(3., 1.), (6., 1.), (7., 5.), (5., 5.), (5., 3.), (3., 3.)])
path = numpy.asarray([(3., 1.), (6.5, 1.), (6.5, 5.), (5., 5.), (5., 2.), (2., 2.), (2., 5.), (1., 5.), (1., 4.), (4, 2.5), (6, 3.5)])

dist_per_frame = 0.2
deg_per_frame = 20

curr_loc = path[0]
curr_angle = 0

i_frame = 0

x = []
y = []
theta_view = []

for i_lag in range(path.shape[0]-1):
    diff = path[i_lag + 1] - curr_loc
    lag_length = numpy.linalg.norm(diff)
    dir = diff/lag_length

    next_angle = math.atan2(diff[1], diff[0])/math.pi*180.

    turn_total = (abs(next_angle - curr_angle)) % 180
    turn_did = 0

    if ((curr_angle - next_angle + 180.) % 360. - 180.) > 0:
        while turn_total > turn_did:
            theta_view.append(curr_angle)
            x.append(curr_loc[0])
            y.append(curr_loc[1])
            curr_angle = (curr_angle - deg_per_frame + 180.) % 360. - 180.
            turn_did = turn_did + deg_per_frame
    else:

        while turn_total > turn_did:
            theta_view.append(curr_angle)
            x.append(curr_loc[0])
            y.append(curr_loc[1])
            curr_angle = (curr_angle + deg_per_frame + 180.) % 360. - 180.
            turn_did = turn_did + deg_per_frame


    curr_angle = next_angle
    x.append(curr_loc[0])
    y.append(curr_loc[1])
    theta_view.append(next_angle)

    d = 0
    d = d + dist_per_frame

    while d < lag_length:
        curr_loc[0] = curr_loc[0] + dist_per_frame*dir[0]
        curr_loc[1] = curr_loc[1] + dist_per_frame*dir[1]
        x.append(curr_loc[0])
        y.append(curr_loc[1])
        theta_view.append(curr_angle)
        d = d + dist_per_frame

    curr_loc = path[i_lag+1, :]
    x.append(curr_loc[0])
    y.append(curr_loc[1])
    theta_view.append(curr_angle)



intrinsic_process_test = numpy.asarray([x, y, theta_view]).T
intrinsic_process_test[:, -1] = 2*numpy.pi*intrinsic_process_test[:, -1]/360

numpy.savetxt(sim_dir + '/' + intrinsic_test_file_name, intrinsic_process_test, delimiter=',')

color_map = create_color_map(intrinsic_simulated_process)
ax = print_process(intrinsic_simulated_process, bounding_shape=None, titleStr="Intrinsic Space", color_map=color_map)

print_process(intrinsic_process_test[:, 0:2].T, titleStr="Intrinsic Test Process", ax=ax)

plt.show(block=True)
