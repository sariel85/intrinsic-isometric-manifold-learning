from __future__ import print_function
from __future__ import absolute_import
from data_generation import print_process, create_color_map, print_dynamics
from util import *
import numpy
from TCIE_helpers import multiscale_isomaps

from non_local_tangent import non_local_tangent_net
###Settings#############################################################################################################
sim_dir_name = "2D Unit Square - Dynamic"  #Which dataset to run
process_mode = "Dynamic"

n_points_used_for_dynamics = 3000 #How many points are available from which to infer dynamics
n_points_used_for_plotting_dynamics = 2000
n_metrics_to_print = 200
n_points_used_for_clusters = 3000 #How many cluster to use in Kernal method

n_neighbors_cov = 60 #How neighboors to use from which to infer dynamics locally
n_neighbors_mds = 20 #How many short distances are kept for each cluster point
n_hidden_drift = 4 #How many nodes in hidden layer that learns intrinsic dynamics
n_hidden_tangent = 20 #How many nodes in hidden layer that learns tangent plane
n_hidden_int = 20 #How many nodes in hidden layer that learns intrinsic dynamics
########################################################################################################################

sim_dir = './' + sim_dir_name
dtype = numpy.float64


short_dist = short_dist + short_dist.T
short_dist_full = scipy.sparse.csgraph.shortest_path(short_dist, directed=False)

intrinsic_process = numpy.loadtxt(sim_dir + '/' + 'intrinsic_used.txt', delimiter=',', dtype=dtype).T
noisy_sensor_measured = numpy.loadtxt(sim_dir + '/' + 'sensor_noisy.txt', delimiter=',', dtype=dtype).T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
measurement_variance = numpy.load(sim_dir + '/' + 'measurement_variance.npy').astype(dtype=dtype)
#dist_potential = numpy.loadtxt(sim_dir + '/' + 'dist_potential_used.txt', delimiter=',', dtype=dtype)

dim_intrinsic = intrinsic_process.shape[0]
dim_measurement = noisy_sensor_measured.shape[0]
n_points = intrinsic_process.shape[1]


if process_mode == "Static":
    noisy_sensor = noisy_sensor_measured[:, ::(dim_intrinsic+1)]
elif process_mode == "Dynamic":
    noisy_sensor = noisy_sensor_measured[:, ::2]
else:
    assert 0

noisy_sensor = (noisy_sensor)

n_points_used_for_dynamics = min(n_points, n_points_used_for_dynamics)
points_used_dynamics_index = numpy.random.choice(n_points, size=n_points_used_for_dynamics, replace=False)

intrinsic_process = intrinsic_process[:, points_used_dynamics_index]
noisy_sensor = noisy_sensor[:, points_used_dynamics_index]
#dist_potential = dist_potential[points_used_dynamics_index]

if process_mode == "Static":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros((points_used_dynamics_index.shape[0]*(dim_intrinsic+1), dim_measurement))
    for i_index in range(points_used_dynamics_index.shape[0]):
        noisy_sensor_measured_new[i_index*(dim_intrinsic+1):(i_index+1)*(dim_intrinsic+1), :] = noisy_sensor_measured[points_used_dynamics_index[i_index]*(dim_intrinsic+1):(points_used_dynamics_index[i_index]+1)*(dim_intrinsic+1),:]
    noisy_sensor_measured = noisy_sensor_measured_new.T
elif process_mode == "Dynamic":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros((2*points_used_dynamics_index.shape[0], dim_measurement))
    for i_index in range(points_used_dynamics_index.shape[0]):
        noisy_sensor_measured_new[i_index * 2:(i_index + 1) * 2, :] = noisy_sensor_measured[points_used_dynamics_index[i_index] * 2:(points_used_dynamics_index[i_index] + 1) * 2, :]
    noisy_sensor_measured = noisy_sensor_measured_new.T
else:
    assert()

n_points = intrinsic_process.shape[1]

n_points_used_for_plotting_dynamics = min(n_points, n_points_used_for_plotting_dynamics)
points_dynamics_plot_index = numpy.random.choice(n_points, size=n_points_used_for_plotting_dynamics, replace=False)

color_map = create_color_map(intrinsic_process)

print_process(intrinsic_process, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Space")

print_process(noisy_sensor, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Space")

'''
fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[0, :])
plt.title("Sensor 1")

fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[1, :])
plt.title("Sensor 2")

fig = plt.figure()
ax = fig.gca()
ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c=noisy_sensor[2, :])
plt.title("Sensor 3")
'''

#plt.savefig(full_dir_name + '/' + 'sensor_base.png', bbox_inches='tight')

#print_dynamics(intrinsic_process_base, intrinsic_process_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Intrinsic Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'intrinsic_dynamics.png', bbox_inches='tight')

#print_dynamics(noisy_sensor_base, noisy_sensor_step, indexs=points_dynamics_plot_index, bounding_shape=None, color_map=color_map, titleStr="Observed Process Dynamics")
#plt.savefig(full_dir_name + '/' + 'sensor_dynamics.png', bbox_inches='tight')


#Testing and comparison with other methods##############################################################################
n_points_used_for_clusters = min(n_points, n_points_used_for_clusters)
points_used_for_clusters_indexs = numpy.random.choice(n_points, size=n_points_used_for_clusters, replace=False)

intrinsic_process_clusters = intrinsic_process[:, points_used_for_clusters_indexs]
noisy_sensor_clusters = noisy_sensor[:, points_used_for_clusters_indexs]
#dist_potential_clusters = dist_potential[points_used_for_clusters_indexs]

if process_mode == "Static":
    noisy_sensor_measured = noisy_sensor_measured.T
    noisy_sensor_measured_new = numpy.zeros((points_used_for_clusters_indexs.shape[0]*(dim_intrinsic+1),dim_measurement))
    for i_index in range(points_used_for_clusters_indexs.shape[0]):
        noisy_sensor_measured_new[i_index*(dim_intrinsic+1):(i_index+1)*(dim_intrinsic+1), :] = noisy_sensor_measured[points_used_for_clusters_indexs[i_index]*(dim_intrinsic+1):(points_used_for_clusters_indexs[i_index]+1)*(dim_intrinsic+1),:]
    noisy_sensor_measured = noisy_sensor_measured_new.T
elif process_mode == "Dynamic":
    noisy_sensor_measured = noisy_sensor_measured.T
    #noisy_sensor_measured_new = numpy.zeros((noisy_sensor_measured.shape[0], dim_measurement))
    #for i_index in range(points_used_for_clusters_indexs.shape[0]):
    #    noisy_sensor_measured_new[i_index * 2:(i_index + 1) * 2, :] = noisy_sensor_measured[points_used_for_clusters_indexs[i_index] * 2:(points_used_for_clusters_indexs[i_index] + 1) * 2, :]
    noisy_sensor_measured = noisy_sensor_measured_new.T
else:
    assert()


n_points_used_for_clusters = intrinsic_process_clusters.shape[1]

color_map_clusters = color_map[points_used_for_clusters_indexs, :]

#test_ml(noisy_sensor_clusters, intrinsic_process_clusters, n_neighbors=n_neighbors_mds, n_components=dim_intrinsic, color=color_map_clusters)

noisy_sensor_base = noisy_sensor_measured[:, ::2]
noisy_sensor_step = noisy_sensor_measured[:, 1::2]
metric_list_def, metric_list_full = get_metrics_from_points(noisy_sensor_clusters, noisy_sensor_base, noisy_sensor_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance, measurement_variance)

n_cluster_points = noisy_sensor_clusters.shape[1]
distance_metrix = scipy.spatial.distance.cdist(noisy_sensor_clusters.T, noisy_sensor_base.T, metric='sqeuclidean')
distance_metrix_full = scipy.spatial.distance.cdist(noisy_sensor_step.T, noisy_sensor_base.T, metric='sqeuclidean')

knn_indexes = numpy.argsort(distance_metrix, axis=1, kind='quicksort')
knn_indexes = knn_indexes[:, :n_neighbors_cov]

#ax = print_process(noisy_sensor_clusters, titleStr="Intrinsic Space")

process = noisy_sensor_clusters
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', aspect='equal')

ax.scatter(process[0, :2000], process[1, :2000], process[2, :2000], c='b', alpha=0.02)


process = noisy_sensor_step[:, knn_indexes[10, :]]
ax.scatter(process[0, :], process[1, :], process[2, :], c='r')
plt.show(block=False)

cluster_map = numpy.zeros([n_points_used_for_clusters, n_points])

for i_cluster in range(n_points_used_for_clusters):
    for j_point in range(n_neighbors_cov):
        cluster_map[i_cluster, knn_indexes[i_cluster, j_point]] = 1/n_neighbors_cov

sigma = numpy.median(distance_metrix_full)/50
distance_metrix_full_kernel = numpy.exp(-distance_metrix_full/(2*(sigma)))
row_sum = numpy.sum(distance_metrix_full_kernel, axis=1)
normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), distance_metrix_full_kernel)

cluster_map2 = numpy.dot(cluster_map, normlized_kernal)
cluster_map = cluster_map2
cluster_map = (cluster_map.T - numpy.mean(cluster_map, axis=1).T).T
fig = plt.figure()
plt.imshow(cluster_map)
plt.show(block=False)
diff_distance_metrix = scipy.spatial.distance.cdist(cluster_map, cluster_map)
diff_distance_metrix_trimmed = trim_distances(diff_distance_metrix, n_neighbors=n_neighbors_mds)

diff_distance_metrix_trimmed_geo = scipy.sparse.csgraph.shortest_path(diff_distance_metrix_trimmed, directed=False)

#kernel = numpy.dot(cluster_map, cluster_map.T)
#fig = plt.figure()
#plt.imshow(kernel)
#plt.show(block=False)

D_squared = diff_distance_metrix_trimmed_geo ** 2

# centering matrix
n = D_squared.shape[0]
J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

# perform double centering
B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

U, s, V = numpy.linalg.svd(B)
eigen_val = s
eigen_vect = U.T
eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])

eigen_vect = eigen_vect[eigen_val_sort_ind]
eigen_vect = eigen_vect[:dim_intrinsic].T
guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)/numpy.sqrt(intrinsic_variance)

print_process(guess.T, titleStr='Embeding', color_map=color_map_clusters)
print("Finish")
plt.show(block=True)
