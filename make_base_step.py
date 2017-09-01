# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from data_generation import print_process, create_color_map, BoundingShape
import numpy
import matplotlib.pyplot as plt
import pickle
import os
from observation_modes import *


sim_dir_name = "2D Apartment to Print"

process_mode = "Array"

if process_mode == "Array":
    n_points_used = 1000000
    intrinsic_variance = 0.10 ** 2
    n_observations_per_cluster = 2
elif process_mode == "Bursts":
    n_points_used = 1000000
    intrinsic_variance = 0.03 ** 2
    n_observations_per_cluster = 100
elif process_mode == "Diffusion":
    n_points_used = 100000
else:
    assert False

if process_mode == "Array":
    sim_dir_load = './' + sim_dir_name + ' - ' + "Static"
    sim_dir_save = './' + sim_dir_name + ' - ' + "Array"
elif process_mode == "Bursts":
    sim_dir_load = './' + sim_dir_name + ' - ' + "Static"
    sim_dir_save = './' + sim_dir_name + ' - ' + "Bursts"
elif process_mode == "Diffusion":
    sim_dir_load = './' + sim_dir_name + ' - ' + "Dynamic"
    sim_dir_save = './' + sim_dir_name + ' - ' + "Diffusion"
else:
    assert False

if not(os.path.isdir(sim_dir_save)):
    os.makedirs(sim_dir_save)

if process_mode == "Array":
    sim_dir_load = './' + sim_dir_name + ' - ' + "Static"
    sim_dir_save = './' + sim_dir_name + ' - ' + "Array"
elif process_mode == "Bursts":
    n_obs_in_cluster = numpy.save(sim_dir_save + '/' + 'n_obs_in_cluster.npy', n_observations_per_cluster)

intrinsic_process_file_name = 'intrinsic_process.npy'
intrinsic_variance_file_name = 'intrinsic_variance.npy'
bounding_shape_file_name = 'bounding_shape.npy'

intrinsic_simulated_process = numpy.load(sim_dir_load + '/' + intrinsic_process_file_name).astype(dtype=numpy.float64).T
dim_intrinsic = intrinsic_simulated_process.shape[0]
n_points = intrinsic_simulated_process.shape[1]

if process_mode == "Array":
    numpy.save(sim_dir_save + '/' + intrinsic_variance_file_name, intrinsic_variance)
elif process_mode == "Bursts":
    numpy.save(sim_dir_save + '/' + intrinsic_variance_file_name, intrinsic_variance)

f = open(sim_dir_load + '/' + 'bounding_shape.pckl', 'rb')
bounding_shape = pickle.load(f)
f.close()

if process_mode == "Dynamic":
    intrinsic_variance = numpy.load(sim_dir_load + '/' + bounding_shape_file_name).astype(dtype=numpy.float64)
if process_mode == "Array" or "Bursts":
    n_points_used = min(n_points, n_points_used)
    points_used_index = numpy.random.choice(intrinsic_simulated_process.shape[1], size=n_points_used, replace=False)
elif "Dynamic":
    n_points_used = min(n_points - 1, n_points_used)
    points_used_index = numpy.random.choice(intrinsic_simulated_process.shape[1]-1, size=n_points_used, replace=False)
else:
    assert(False)

if process_mode == "Array":
    sensor_array_matrix = numpy.random.randn(dim_intrinsic, n_observations_per_cluster)
    #angles = numpy.arange(0, 2*numpy.pi, 2*numpy.pi/n_observations_per_cluster)
    #angles = numpy.asarray([0, numpy.pi/2])
    #sensor_array_matrix[0, :] = numpy.cos(angles)
    #sensor_array_matrix[1, :] = numpy.sin(angles)
    sensor_array_matrix = numpy.eye(2)
    sensor_array_matrix = sensor_array_matrix

    ax=print_process(sensor_array_matrix, titleStr="Sensor Array")
    for i_sensor in range(n_observations_per_cluster):
        ax.plot([0, sensor_array_matrix[0, i_sensor]], [0, sensor_array_matrix[1, i_sensor]], c="k")
    ax.scatter(0, 0, c="r", s=80)

    intrinsic_process_to_measure = numpy.zeros((dim_intrinsic, n_points_used*(n_observations_per_cluster+1)))
    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_simulated_process = intrinsic_simulated_process.T

    for i_point in range(n_points_used):
        rand_mat_temp = numpy.random.randn(dim_intrinsic, dim_intrinsic)
        U, s, V = numpy.linalg.svd(rand_mat_temp)
        sensor_array_matrix_rotated = numpy.dot(V, sensor_array_matrix)
        intrinsic_process_to_measure[i_point*(n_observations_per_cluster+1)+0, :] = intrinsic_simulated_process[points_used_index[i_point], :]
        for i_obs in range(n_observations_per_cluster):
            intrinsic_process_to_measure[i_point * (n_observations_per_cluster + 1) + i_obs + 1, :] = intrinsic_simulated_process[points_used_index[i_point], :]
            intrinsic_process_to_measure[i_point * (n_observations_per_cluster + 1) + i_obs + 1, :] = intrinsic_process_to_measure[i_point * (n_observations_per_cluster + 1) + i_obs + 1, :] + sensor_array_matrix_rotated[:, i_obs]*numpy.sqrt(intrinsic_variance)

    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_simulated_process = intrinsic_simulated_process.T
    n_points_used_for_plot = 10
    obs_used_for_sensor_array_indexes = numpy.random.choice(n_points_used, size=n_points_used_for_plot, replace=False)
    indexes_to_print = numpy.kron(obs_used_for_sensor_array_indexes*(n_observations_per_cluster+1), numpy.ones((1,n_observations_per_cluster))) + numpy.kron(numpy.ones((1,n_points_used_for_plot)), numpy.arange(1, n_observations_per_cluster+1))
    indexes_to_print = indexes_to_print.astype(int)
    ax = print_process(intrinsic_process_to_measure[:, indexes_to_print], bounding_shape=bounding_shape, titleStr="Intrinsic Process to Measure")
    ax.scatter(intrinsic_simulated_process[0, points_used_index[obs_used_for_sensor_array_indexes]], intrinsic_simulated_process[1, points_used_index[obs_used_for_sensor_array_indexes]], c='r')

    numpy.save(sim_dir_save + '/' + 'sensor_array_matrix.npy', sensor_array_matrix)

elif process_mode == "Bursts":
    intrinsic_process_to_measure = numpy.zeros((dim_intrinsic, n_observations_per_cluster*n_points_used)).T
    intrinsic_simulated_process = intrinsic_simulated_process.T
    for i_point in range(n_points_used):
        intrinsic_process_to_measure[(i_point * n_observations_per_cluster):((i_point + 1) * n_observations_per_cluster), :] = intrinsic_simulated_process[points_used_index[i_point], :] + numpy.sqrt(intrinsic_variance)*numpy.random.randn(n_observations_per_cluster, dim_intrinsic)
    intrinsic_process_to_measure = intrinsic_process_to_measure.T
    intrinsic_simulated_process = intrinsic_simulated_process.T
    n_points_used_for_plot = 10
    obs_used_for_sensor_array_indexes = numpy.random.choice(n_points_used, size=n_points_used_for_plot, replace=False)
    indexes_to_print = numpy.kron(obs_used_for_sensor_array_indexes*n_observations_per_cluster, numpy.ones((1,n_observations_per_cluster))) + numpy.kron(numpy.ones((1,n_points_used_for_plot)), numpy.arange(0, n_observations_per_cluster))
    indexes_to_print = indexes_to_print.astype(int)
    ax = print_process(intrinsic_process_to_measure[:, indexes_to_print], bounding_shape=bounding_shape, titleStr="")
    ax.scatter(intrinsic_simulated_process[0, points_used_index[obs_used_for_sensor_array_indexes]], intrinsic_simulated_process[1, points_used_index[obs_used_for_sensor_array_indexes]], c='r')

    if False:
        observed_process_to_measure = whole_sphere(intrinsic_process_to_measure, k=5)
        observed_simulated_process = whole_sphere(intrinsic_simulated_process, k=5)
        ax = print_process(observed_process_to_measure[:, indexes_to_print], titleStr="")
        ax.scatter(observed_simulated_process[0, :], observed_simulated_process[1, :], observed_simulated_process[2, :], c='r', s=20)


elif process_mode == "Diffusion":
        intrinsic_process_base = intrinsic_simulated_process[:, points_used_index]
        intrinsic_process_step = intrinsic_simulated_process[:, points_used_index+1]
        intrinsic_process_to_measure = numpy.zeros((dim_intrinsic, 2*n_points_used))
        intrinsic_process_to_measure = intrinsic_process_to_measure.T
        intrinsic_process_base = intrinsic_process_base.T
        intrinsic_process_step = intrinsic_process_step.T
        for i_point in range(n_points_used):
            intrinsic_process_to_measure[i_point*2 + 0, :] = intrinsic_process_base[i_point, :]
            intrinsic_process_to_measure[i_point*2 + 1, :] = intrinsic_process_step[i_point, :]
        intrinsic_process_to_measure = intrinsic_process_to_measure.T
        intrinsic_process_base = intrinsic_process_base.T
        intrinsic_process_step = intrinsic_process_step.T
        print_process(intrinsic_process_base, bounding_shape=bounding_shape, titleStr="Intrinsic Base Process")
        print_process(intrinsic_process_step, bounding_shape=bounding_shape, titleStr="Intrinsic Step Process")
else:
    assert False

numpy.savetxt(sim_dir_save + '/' + 'intrinsic_process_to_measure.txt', intrinsic_process_to_measure.T, delimiter=',')

plt.show(block=True)
