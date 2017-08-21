# Generates a simulated intrinsic process and stores it in a specified directory

from __future__ import print_function
from __future__ import absolute_import
from data_generation import print_process, create_color_map
import matplotlib.pyplot as plt
import os
from observation_modes import *

n_plot_points = 5000
noise_variance = 0.00**2

#intrinsic_process_file_name = 'intrinsic_process.npy'
sim_dir_name_int = "2D Unit Square Punctured by Cross - Bursts"

if not(os.path.isdir(sim_dir_name_int)):
    assert False

process_mode = sim_dir_name_int.split(" - ")[1]
if (process_mode != "Array") and (process_mode != "Bursts") and (process_mode != "Diffusion"):
    assert False

intrinsic_states = numpy.loadtxt(sim_dir_name_int + '/' + 'intrinsic_process_to_measure.txt', delimiter=',').T
intrinsic_variance = numpy.load(sim_dir_name_int + '/' + 'intrinsic_variance.npy').astype(dtype=numpy.float64)

n_points = intrinsic_states.shape[1]

'''
obs_type = "S-Curve"
observed_states_exact = s_curve(intrinsic_states, k=0.95)
azi = -59
el = 18
'''

'''
obs_type = "Twin-Peaks"
observed_states_exact = twin_peaks(intrinsic_states, k=5)
azi = -22
el = 53
'''


obs_type = "Severed Sphere"
observed_states_exact = severed_sphere(intrinsic_states, k1=2.5, k2=1)
azi = -61
el = 25


'''
obs_type = "Swissroll"
intrinsic_states_flipped = numpy.empty(intrinsic_states.shape, dtype=numpy.float64)
intrinsic_states_flipped[0] = intrinsic_states[1]
intrinsic_states_flipped[1] = intrinsic_states[0]
observed_states_exact = swissroll(intrinsic_states_flipped, k_r=0.5, k_twist=7)
azi = -82
el = 5
'''

'''
obs_type = "Singers Mushroom"
observed_states_exact = singers_mushroom(intrinsic_states)/8
observed_states_exact = whole_sphere(observed_states_exact, k=5)
'''

'''
obs_type = "Fishbowl"
observed_states_exact = whole_sphere(intrinsic_states, k=5)
'''

'''
obs_type = "Antennas"
ant_1 = numpy.asarray([[0.0], [0.0]])
ant_2 = numpy.asarray([[12.0], [8.0]])
ant_3 = numpy.asarray([[3.0], [7.0]])
ant_4 = numpy.asarray([[8.0], [0.0]])
ant_5 = numpy.asarray([[6.0], [7.0]])
ant_6 = numpy.asarray([[0.0], [4.0]])

antennas = numpy.concatenate((ant_1, ant_2, ant_3, ant_4, ant_5, ant_6), axis=1)

range_factor = [[2.], [2.], [2.], [2.], [2.], [2.]]
width = numpy.asarray([1, 2, 3, 4, 5, 6])
angles = numpy.asarray([1, 2, 3, 4, 5, 6])

amplitudes = [[1],[1],[1],[1],[1],[1]]
observed_states_exact = antenna(intrinsic_states, centers=antennas, amplitudes=amplitudes, width=width, angles=angles, range_factor=range_factor, reg_fact=2)
'''
'''
obs_type = "Antennas"
ant_1 = numpy.asarray([[0.2], [1.5]])
ant_2 = numpy.asarray([[7.3], [1.5]])
ant_3 = numpy.asarray([[12.8], [3.2]])
ant_4 = numpy.asarray([[4.5], [6]])
ant_5 = numpy.asarray([[9.12], [8.35]])
ant_6 = numpy.asarray([[4.0], [0.]])

antennas = numpy.concatenate((ant_1, ant_2, ant_3, ant_4, ant_5, ant_6), axis=1)

range_factor = [[2.], [2.], [2.], [2.], [2.], [2.]]
width = numpy.asarray([0.2, 0.05, 1.5, 0.1, 1., 0.3])
angles = numpy.asarray([0.4, 2.3, 2.6, 6.0, 4.8, 1.7])
reg_fact = numpy.asarray([2., 2.0, 2.0, 2.0, 2.0, 2.0])
amplitudes = [[1],[1],[1],[1],[1],[1]]
observed_states_exact = antenna(intrinsic_states, centers=antennas, amplitudes=amplitudes, width=width, angles=angles, range_factor=range_factor, reg_fact=reg_fact)

if True:
    for i_dim_print in range(observed_states_exact.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.scatter(intrinsic_states[0, :], intrinsic_states[1, :], c=(observed_states_exact[i_dim_print, :])**(1/20), cmap='YlOrRd')
        plt.title("Measurement %d" % (i_dim_print+1))
        plt.show(block=False)
'''

'''
obs_type = "Distance"
ant_1 = numpy.asarray([[2.5], [1.3]])
ant_2 = numpy.asarray([[4.5], [6.2]])
ant_3 = numpy.asarray([[9.5], [3.7]])
antenas = numpy.concatenate((ant_1, ant_2, ant_3), axis=1)
observed_states_exact = numpy.zeros((antenas.shape[1], intrinsic_states.shape[1]), dtype=numpy.float64)
observed_states_exact[0, :] = (((intrinsic_states.T - antenas[:, 0].T).T)**2).sum(axis=0)
observed_states_exact[1, :] = (((intrinsic_states.T - antenas[:, 1].T).T)**2).sum(axis=0)
observed_states_exact[2, :] = (((intrinsic_states.T - antenas[:, 2].T).T)**2).sum(axis=0)
'''



# Add noise
observed_states_noisy = observed_states_exact + numpy.sqrt(noise_variance) * numpy.random.randn(observed_states_exact.shape[0], n_points)

sim_dir_name_save = sim_dir_name_int + " - " + obs_type

# Save data
sim_dir_name_save = './' + sim_dir_name_save

if not(os.path.isdir(sim_dir_name_save)):
    os.makedirs(sim_dir_name_save)

numpy.savetxt(sim_dir_name_save + '/' + 'observed_states_exact.txt', observed_states_exact.T, delimiter=',')

numpy.savetxt(sim_dir_name_save + '/' + 'observed_states_noisy.txt', observed_states_noisy.T, delimiter=',')

numpy.savetxt(sim_dir_name_save + '/' + 'intrinsic_states.txt', intrinsic_states, delimiter=',')

numpy.save(sim_dir_name_save + '/' + 'intrinsic_variance', intrinsic_variance)

numpy.save(sim_dir_name_save + '/' + 'noise_variance', noise_variance)

numpy.save(sim_dir_name_save + '/' + 'azi', azi)

numpy.save(sim_dir_name_save + '/' + 'el', el)

n_plot_points = min(n_points, n_plot_points)
points_plot_index = numpy.random.choice(n_points, size=n_plot_points, replace=False)

color_map = create_color_map(intrinsic_states)

if process_mode == "Array":
    sensor_array_matrix = numpy.load(sim_dir_name_int + '/' + 'sensor_array_matrix.npy').astype(dtype=numpy.float64)
    numpy.save(sim_dir_name_save + '/' + 'sensor_array_matrix.npy', sensor_array_matrix)
elif process_mode == "Bursts":
    n_obs_in_cluster = numpy.load(sim_dir_name_int + '/' + 'n_obs_in_cluster.npy').astype(dtype=numpy.float64)
    n_obs_in_cluster = numpy.save(sim_dir_name_save + '/' + 'n_obs_in_cluster.npy', n_obs_in_cluster)

print_process(intrinsic_states, indexes=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic States", el=el, azi=azi)
print_process(observed_states_exact, indexes=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Data", el=el, azi=azi)
print_process(observed_states_noisy, indexes=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Data + Noise", el=el, azi=azi)

plt.show(block=True)
