from __future__ import print_function
from __future__ import absolute_import
from data_generation import print_process, create_color_map
from util import *
import numpy
from intrinsic_isometric_mapping import intrinsic_isometric_mapping
from intrinsic_metric_net import IntrinsicMetricNet
import os
import itertools

###Settings#############################################################################################################
net_flag = False
embed_flag = True

param_w = [0.0001]
param_nodes = [50]
#param_w = [0.00025]
#param_nodes = [20]

param_list = list(itertools.product(param_w, param_nodes))

#sim_dir_name = "2D Unit Square Punctured by Cross - Bursts - Severed Sphere"
sim_dir_name = "2D Apartment - Array - Color"

if not(os.path.isdir(sim_dir_name)):
    assert False

process_mode = sim_dir_name.split(" - ")[1]
if (process_mode != "Array") and (process_mode != "Bursts") and (process_mode != "Diffusion"):
    assert False

n_points_used_for_dynamics_true = 4000
n_obs_used_for_sensor_array = 4000
n_points_used_for_dynamics = 4000
n_points_used_for_clusters = 4000
n_points_used_for_clusters_2 = 500

n_obs_used_in_cluster = 2000
n_dim_used = 5

n_points_used_for_drift_plots = 300
n_points_used_for_metric_plots = 300

n_clusters_isometric_mapping = 1
size_patch_start = n_points_used_for_clusters_2
size_patch_step = 400
n_mds_iterations = 100
n_neighbors_cov_dense = 500
n_neighbors_cov = 5
n_neighbors_mds_1 = 10
n_neighbors_mds_2 = 10
mds_stop_threshold = 1e-9
n_net_initializations = 1
dtype = numpy.float64
cov_iter_factor = 800

########################################################################################################################
#Load files
########################################################################################################################
sim_dir = './' + sim_dir_name
intrinsic_process_total = numpy.loadtxt(sim_dir + '/' + 'intrinsic_states.txt', delimiter=',', dtype=dtype)
if intrinsic_process_total.shape[0] > intrinsic_process_total.shape[1]:
    intrinsic_process_total = intrinsic_process_total.T
noisy_sensor_measured_total = numpy.loadtxt(sim_dir + '/' + 'observed_states_noisy.txt', delimiter=',', dtype=dtype)
if noisy_sensor_measured_total.shape[0] > noisy_sensor_measured_total.shape[1]:
    noisy_sensor_measured_total = noisy_sensor_measured_total.T
intrinsic_variance = numpy.load(sim_dir + '/' + 'intrinsic_variance.npy').astype(dtype=dtype)
#noise_variance = numpy.load(sim_dir + '/' + 'noise_variance.npy').astype(dtype=dtype)
#azi = numpy.load(sim_dir + '/' + 'azi.npy').astype(dtype=dtype)
azi = 0
#el = numpy.load(sim_dir + '/' + 'el.npy').astype(dtype=dtype)
el = 0
if process_mode == "Array":
    sensor_array_matrix_dense = numpy.load(sim_dir + '/' + 'sensor_array_matrix.npy').astype(dtype=dtype)
    n_obs_in_sensor_array = sensor_array_matrix_dense.shape[1]
elif process_mode == "Bursts":
    n_obs_in_cluster = int(numpy.load(sim_dir + '/' + 'n_obs_in_cluster.npy'))

dim_intrinsic = intrinsic_process_total.shape[0]
dim_measurement = noisy_sensor_measured_total.shape[0]
########################################################################################################################

if process_mode == "Array":
    n_points_total = numpy.int(noisy_sensor_measured_total.shape[1]/(n_obs_in_sensor_array + 1))
    intrinsic_process_total_reshaped = numpy.reshape(intrinsic_process_total, [dim_intrinsic, n_points_total, n_obs_in_sensor_array + 1], order='C')
    noisy_sensor_measured_total_reshaped = numpy.reshape(noisy_sensor_measured_total, [dim_measurement, n_points_total, n_obs_in_sensor_array + 1], order='C')
    intrinsic_process_base_total = intrinsic_process_total_reshaped[:, :, 0]
    intrinsic_process_step_total = intrinsic_process_total_reshaped[:, :, 1:]
    noisy_sensor_base_total = noisy_sensor_measured_total_reshaped[:, :, 0]
    n_obs_used_in_each_cluster = min(n_obs_used_in_cluster, n_obs_in_sensor_array)
    obs_used_in_each_cluster_indexes = numpy.random.choice(n_obs_in_sensor_array, size=n_obs_used_in_each_cluster, replace=False)
    sensor_array_matrix = sensor_array_matrix_dense[:, obs_used_in_each_cluster_indexes]
    noisy_sensor_step_total = noisy_sensor_measured_total_reshaped[:, :, 1:]
elif process_mode == "Bursts":
    n_points_total = numpy.int(noisy_sensor_measured_total.shape[1] / n_obs_in_cluster)
    intrinsic_process_total_reshaped = numpy.reshape(intrinsic_process_total,
                                                     [dim_intrinsic, n_points_total, n_obs_in_cluster],
                                                     order='C')
    noisy_sensor_measured_total_reshaped = numpy.reshape(noisy_sensor_measured_total,
                                                         [dim_measurement, n_points_total,
                                                          n_obs_in_cluster], order='C')
    intrinsic_process_base_total = intrinsic_process_total_reshaped[:, :, :].mean(axis=2)
    intrinsic_process_step_total = intrinsic_process_total_reshaped[:, :, :]
    noisy_sensor_base_total = noisy_sensor_measured_total_reshaped[:, :, :].mean(axis=2)
    noisy_sensor_step_total = noisy_sensor_measured_total_reshaped[:, :, :]
    n_obs_used_in_each_cluster = min(n_obs_used_in_cluster, n_obs_in_cluster)
    obs_used_in_each_cluster_indexes = numpy.random.choice(n_obs_in_cluster, size=n_obs_used_in_each_cluster, replace=False)
elif process_mode == "Diffusion":
    intrinsic_process_base_total = intrinsic_process_total[:, ::2]
    intrinsic_process_step_total = intrinsic_process_total[:, 1::2]
    noisy_sensor_base_total = noisy_sensor_measured_total[:, ::2]
    noisy_sensor_step_total = noisy_sensor_measured_total[:, 1::2]
else:
    assert 0

n_points_total = intrinsic_process_base_total.shape[1]
n_points_used_for_dynamics_true = min(n_points_total, n_points_used_for_dynamics_true)
points_used_dynamics_indexes_true = numpy.random.choice(n_points_total, size=n_points_used_for_dynamics_true,
                                                        replace=False)
intrinsic_process_base_dynamics_true = intrinsic_process_base_total[:, points_used_dynamics_indexes_true]
intrinsic_process_step_dynamics_true = intrinsic_process_step_total[:, points_used_dynamics_indexes_true]
noisy_sensor_base_dynamics_true = noisy_sensor_base_total[:, points_used_dynamics_indexes_true]
noisy_sensor_step_dynamics_true = noisy_sensor_step_total[:, points_used_dynamics_indexes_true]

n_points_used_for_dynamics = min(n_points_used_for_dynamics_true, n_points_used_for_dynamics)
points_used_dynamics_indexes = numpy.random.choice(n_points_used_for_dynamics_true, size=n_points_used_for_dynamics,
                                                   replace=False)
intrinsic_process_base_dynamics = intrinsic_process_base_dynamics_true[:, points_used_dynamics_indexes]
intrinsic_process_step_dynamics = intrinsic_process_step_dynamics_true[:, points_used_dynamics_indexes]
noisy_sensor_base_dynamics = noisy_sensor_base_dynamics_true[:, points_used_dynamics_indexes]
noisy_sensor_step_dynamics = noisy_sensor_step_dynamics_true[:, points_used_dynamics_indexes]

color_map_dynamics = create_color_map(intrinsic_process_base_dynamics)

n_points_used_for_clusters = min(n_points_used_for_dynamics, n_points_used_for_clusters)
points_used_for_clusters_indexes = numpy.random.choice(n_points_used_for_dynamics, size=n_points_used_for_clusters,
                                                       replace=False)
intrinsic_process_base_clusters = intrinsic_process_base_dynamics[:, points_used_for_clusters_indexes]
intrinsic_process_step_clusters_dense = intrinsic_process_step_dynamics[:, points_used_for_clusters_indexes]
noisy_sensor_base_clusters = noisy_sensor_base_dynamics[:, points_used_for_clusters_indexes]
noisy_sensor_step_clusters_dense = noisy_sensor_step_dynamics[:, points_used_for_clusters_indexes]

intrinsic_process_step_clusters = intrinsic_process_step_dynamics[:, points_used_for_clusters_indexes][:, :, obs_used_in_each_cluster_indexes]
noisy_sensor_step_clusters = noisy_sensor_step_dynamics[:, points_used_for_clusters_indexes][:, :, obs_used_in_each_cluster_indexes]

color_map_clusters = color_map_dynamics[points_used_for_clusters_indexes, :]

# dist_potential_clusters = dist_potential[points_used_for_clusters_indexs]

n_points_used_for_metric_plots = min(n_points_used_for_clusters, n_points_used_for_metric_plots)
n_points_used_for_drift_plots = min(n_points_used_for_clusters, n_points_used_for_drift_plots)

points_used_for_metric_plots_indexes = numpy.random.choice(n_points_used_for_clusters,
                                                           size=n_points_used_for_metric_plots, replace=False)
points_used_for_drift_plots_indexes = numpy.random.choice(n_points_used_for_clusters,
                                                          size=n_points_used_for_drift_plots, replace=False)

if False:
    print_process(intrinsic_process_base_clusters, indexes=points_used_for_drift_plots_indexes, bounding_shape=None,
                  color_map=color_map_clusters, titleStr="Intrinsic Space")
    print_process(noisy_sensor_base_clusters, indexes=points_used_for_drift_plots_indexes, bounding_shape=None,
                  color_map=color_map_clusters, titleStr="Observed Space")

if False:
    for i_dim_print in range(dim_measurement):
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.scatter(intrinsic_process_base_clusters[0, :], intrinsic_process_base_clusters[1, :], c=(noisy_sensor_base_clusters[i_dim_print, :]), cmap='YlOrRd')
        plt.title("Measurement %d" % (i_dim_print+1))
        plt.show(block=False)


if False:
    test_ml(noisy_sensor_base_clusters, intrinsic_process_base_clusters, n_neighbors=n_neighbors_mds_1,
            n_components=dim_intrinsic, color=color_map_clusters)

if process_mode == "Array":
    cov_list_local_dense, metric_list_local_dense, metric_list_full_rank_local_dense, drift_list_local_dense, noise_cov_list_local_dense, est_noise_variance_local_dense = get_metrics_from_points_array(noisy_sensor_base_clusters[:n_dim_used, :], noisy_sensor_step_clusters_dense[:n_dim_used, :], dim_intrinsic, sensor_array_matrix_dense, intrinsic_variance)
    cov_list_local, metric_list_local, metric_list_full_rank_local, drift_list_local, noise_cov_list_local, est_noise_variance_local = get_metrics_from_points_array(noisy_sensor_base_clusters[:n_dim_used, :], noisy_sensor_step_clusters[:n_dim_used, :], dim_intrinsic, sensor_array_matrix, intrinsic_variance)
elif process_mode == "Bursts":
    cov_list_local_dense, metric_list_local_dense, metric_list_full_rank_local_dense, drift_list_local_dense, noise_cov_list_local_dense, est_noise_variance_local_dense = get_metrics_from_points_bursts(
        noisy_sensor_base_clusters[:n_dim_used, :], noisy_sensor_step_clusters_dense[:n_dim_used, :, :n_neighbors_cov_dense], dim_intrinsic,
        intrinsic_variance)
    cov_list_local, metric_list_local, metric_list_full_rank_local, drift_list_local, noise_cov_list_local, est_noise_variance_local = get_metrics_from_points_bursts(
        noisy_sensor_base_clusters[:n_dim_used, :], noisy_sensor_step_clusters[:n_dim_used, :, :n_neighbors_cov], dim_intrinsic,
        intrinsic_variance)
elif process_mode == "Diffusion":
    cov_list_local_dense, metric_list_local_dense, metric_list_full_rank_local_dense, drift_list_local_dense, noise_cov_list_local_dense, est_noise_variance_local_dense = get_metrics_from_points_diffusion(
        noisy_sensor_base_clusters, noisy_sensor_base_dynamics_true, noisy_sensor_step_dynamics_true,
        n_neighbors_cov_dense, dim_intrinsic, intrinsic_variance, None)
    cov_list_local, metric_list_local, metric_list_full_rank_local, drift_list_local, noise_cov_list_local, est_noise_variance_local = get_metrics_from_points_diffusion(
        noisy_sensor_base_clusters, noisy_sensor_base_dynamics, noisy_sensor_step_dynamics, n_neighbors_cov,
        dim_intrinsic, intrinsic_variance, None)
else:
    assert False

print_process(intrinsic_process_base_clusters, bounding_shape=None, color_map=color_map_clusters, titleStr="Intrinsic Space", azi=azi, el=el)
plt.savefig("plot_temp/intrinsic.png", bbox_inches='tight')

print_process(noisy_sensor_base_clusters, bounding_shape=None, color_map=color_map_clusters, titleStr="Observed Space", azi=azi, el=el)
plt.savefig("plot_temp/observed.png", bbox_inches='tight')

metric_print_scale = 5e7
metric_print_scale = 3

print_metrics(noisy_sensor_base_clusters, metric_list_local_dense, intrinsic_dim=dim_intrinsic, titleStr="Intrinsic metric - True", scale=3*metric_print_scale*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters, points_used_for_clusters_indexes=points_used_for_metric_plots_indexes, azi=azi, el=el)
plt.savefig("plot_temp/metric_local_dense.png", bbox_inches='tight')
print_metrics(noisy_sensor_base_clusters, metric_list_local, intrinsic_dim=dim_intrinsic, titleStr="Intrinsic metric - Local estimation", scale=3*metric_print_scale*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters, points_used_for_clusters_indexes=points_used_for_metric_plots_indexes, azi=azi, el=el)
plt.savefig("plot_temp/metric_local.png", bbox_inches='tight')

if net_flag:
    cov_local_dense = numpy.asarray(cov_list_local_dense)
    drift_local_dense = numpy.asarray(drift_list_local_dense)
    noise_cov_local_dense = numpy.asarray(noise_cov_list_local_dense)

    cov_local = numpy.asarray(cov_list_local)
    drift_local = numpy.asarray(drift_list_local)
    noise_cov_local = numpy.asarray(noise_cov_list_local)

    # print_drift(noisy_sensor_base_clusters[:n_dim_used, points_used_for_drift_plots_indexes], drift_local_dense.T[:, points_used_for_drift_plots_indexes], titleStr="Exact drift")
    # print_drift(noisy_sensor_base_clusters[:, points_used_for_drift_plots_indexes], drift_local.T[:, points_used_for_drift_plots_indexes], titleStr="Locally estimated drift")

    #if noise_variance > 0:
    #    est_noise_variance_local = noise_variance
    #else:
    est_noise_variance_local = numpy.sqrt(est_noise_variance_local*10)


    cross_validation_runs = 1
    per_valid = 0

    for i_run in range(cross_validation_runs):

        n_init_cluster_points_total = noisy_sensor_base_clusters.shape[1]
        n_init_cluster_points_valid = int(numpy.ceil(n_init_cluster_points_total * per_valid))
        n_init_cluster_points_train = n_init_cluster_points_total - n_init_cluster_points_valid
        indexes_valid = numpy.random.choice(n_init_cluster_points_total, size=n_init_cluster_points_valid, replace=False)
        indexes_train = numpy.setdiff1d(numpy.arange(n_init_cluster_points_total), indexes_valid, assume_unique=False)
        init_cluster_points_train = noisy_sensor_base_clusters[:n_dim_used, indexes_train]
        init_cluster_points_valid = noisy_sensor_base_clusters[:n_dim_used, indexes_valid]
        init_cluster_points_true = noisy_sensor_base_clusters[:n_dim_used, :]
        init_cov_list_train = cov_local[indexes_train, :]
        init_cov_list_valid = cov_local[indexes_valid, :]
        init_cov_list_true = cov_local_dense
        init_drift_list_train = drift_local[indexes_train]
        init_drift_list_valid = drift_local[indexes_valid]
        init_drift_list_true = drift_local_dense
        reg_list_train = noise_cov_local[indexes_train]
        reg_list_valid = noise_cov_local[indexes_valid]
        reg_list_true = noise_cov_local

        for i_param in range(param_list.__len__()):

            n_hidden_drift = 50
            n_hidden_tangent = param_list[i_param][1]
            batch_size = int(0.05*n_points_used_for_clusters)
            drift_iter = 0
            cov_iter = int(cov_iter_factor/2)
            learning_rate = 1e-1
            weight_decay = param_list[i_param][0]
            momentum = 0.8
            momentum2 = 0.999
            slowdown = 0.001

            non_local_tangent_net_instance_list = []
            valid_error_list = []
            test_error_list = []

            for i_net_trains in range(n_net_initializations):
                non_local_tangent_net_instance_temp = IntrinsicMetricNet(dim_measurements=dim_measurement, local_noise=est_noise_variance_local, dim_intrinsic=dim_intrinsic, n_hidden_tangent=n_hidden_tangent, n_hidden_drift=n_hidden_drift)
                non_local_tangent_net_instance_list.append(non_local_tangent_net_instance_temp)
                logs = non_local_tangent_net_instance_temp.train_net(intrinsic_variance, batch_size, init_cluster_points_train, init_cluster_points_valid, init_cluster_points_true, init_cov_list_train, init_cov_list_valid, init_cov_list_true, reg_list_train, reg_list_valid, reg_list_true, init_drift_list_train, init_drift_list_valid, init_drift_list_true, drift_iter=0, cov_init_iter=0, cov_iter=cov_iter, weight_decay=weight_decay, momentum=momentum, momentum2=momentum2, learning_rate=learning_rate, slowdown=slowdown, train_var=False)
                valid_error_list.append(logs[1, -1])
                test_error_list.append(logs[3, -1])

            non_local_tangent_net_instance = non_local_tangent_net_instance_list[numpy.argmin(numpy.asarray(valid_error_list))]

            #drift_list_net = get_drift_from_net(non_local_tangent_net_instance, noisy_sensor_base_clusters)
            #drift_net = numpy.asarray(drift_list_net)
            #print_drift(noisy_sensor_base_clusters[:, points_used_for_drift_plots_indexes], drift_net.T[:, points_used_for_drift_plots_indexes], titleStr="Net estimated drift")

            metric_list_net = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_base_clusters)
            print_metrics(noisy_sensor_base_clusters, metric_list_net, intrinsic_dim=dim_intrinsic, titleStr="Net estimated - Initial", scale=3*metric_print_scale*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters, points_used_for_clusters_indexes=points_used_for_metric_plots_indexes, azi=azi, el=el)
            plt.savefig("plot_temp/metric_net_init.png", bbox_inches='tight')

            n_hidden_tangent = param_list[i_param][1]
            batch_size = int(0.3*n_points_used_for_clusters)
            drift_iter = 0
            cov_iter = cov_iter_factor
            learning_rate = 1e-2
            weight_decay = param_list[i_param][0]
            momentum = 0.8
            momentum2 = 0.999
            slowdown = 0.1

            logs = non_local_tangent_net_instance.train_net(intrinsic_variance, batch_size, init_cluster_points_train, init_cluster_points_valid, init_cluster_points_true, init_cov_list_train, init_cov_list_valid, init_cov_list_true, reg_list_train, reg_list_valid, reg_list_true, init_drift_list_train, init_drift_list_valid, init_drift_list_true, drift_iter=0, cov_init_iter=0, cov_iter=cov_iter, weight_decay=weight_decay, momentum=momentum, momentum2=momentum2, learning_rate=learning_rate, slowdown=slowdown, train_var=True)


            metric_list_net = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_base_clusters)
            print_metrics(noisy_sensor_base_clusters, metric_list_net, intrinsic_dim=dim_intrinsic, titleStr="Net estimated - Mid", scale=3*metric_print_scale*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters, points_used_for_clusters_indexes=points_used_for_metric_plots_indexes, el=el, azi=azi)
            plt.savefig("plot_temp/metric_net_mid.png", bbox_inches='tight')


            n_hidden_tangent = param_list[i_param][1]
            batch_size = n_points_used_for_clusters
            drift_iter = 0
            cov_iter = int(cov_iter_factor)
            learning_rate = 1e-2
            weight_decay = param_list[i_param][0]
            momentum = 0.8
            momentum2 = 0.999
            slowdown = 0.01

            logs = non_local_tangent_net_instance.train_net(intrinsic_variance, batch_size, init_cluster_points_train, init_cluster_points_valid, init_cluster_points_true, init_cov_list_train, init_cov_list_valid, init_cov_list_true, reg_list_train, reg_list_valid, reg_list_true, init_drift_list_train, init_drift_list_valid, init_drift_list_true, drift_iter=0, cov_init_iter=0, cov_iter=cov_iter, weight_decay=weight_decay, momentum=momentum, momentum2=momentum2, learning_rate=learning_rate, slowdown=slowdown, train_var=True)

            metric_list_net = get_metrics_from_net(non_local_tangent_net_instance, noisy_sensor_base_clusters)
            print_metrics(noisy_sensor_base_clusters, metric_list_net, intrinsic_dim=dim_intrinsic, titleStr="Intrinsic metric - Globally estimated", scale=3*metric_print_scale*intrinsic_variance, space_mode=False, elipse=True, color_map=color_map_clusters, points_used_for_clusters_indexes=points_used_for_metric_plots_indexes, el=el, azi=azi)
            plt.savefig("plot_temp/metric_net.png", bbox_inches='tight')
            cross_dir = sim_dir + '/Cross2/' + 'n_' + str(n_hidden_tangent) + '_w_' + str(weight_decay) + '/'
            if not (os.path.isdir(cross_dir)):
                os.makedirs(cross_dir)
            rand_num = numpy.random.randint(1000)
            numpy.save(cross_dir + 'log_' + str(rand_num) + '.npy', logs.T)


dist_true = scipy.spatial.distance.cdist(intrinsic_process_base_clusters.T, intrinsic_process_base_clusters.T)
dist_observed = scipy.spatial.distance.cdist(noisy_sensor_base_clusters.T, noisy_sensor_base_clusters.T)
dist_local = numpy.sqrt(calc_dist(noisy_sensor_base_clusters[:n_dim_used, :], metric_list_local))
dist_local_dense = numpy.sqrt(calc_dist(noisy_sensor_base_clusters[:n_dim_used, :], metric_list_local_dense))

if net_flag:
    dist_net = numpy.sqrt(calc_dist(noisy_sensor_base_clusters, metric_list_net))

#Keep shortest distances
dist_trimmed_true = trim_distances(dist_true, n_neighbors=n_neighbors_mds_1)
dist_trimmed_observed = trim_distances(dist_observed, n_neighbors=n_neighbors_mds_1)
dist_trimmed_local = trim_distances(dist_local, dist_observed, n_neighbors=n_neighbors_mds_1)
dist_trimmed_local_dense = trim_distances(dist_local_dense, dist_observed, n_neighbors=n_neighbors_mds_1)

if net_flag:
    dist_trimmed_net = trim_distances(dist_net, dist_observed, n_neighbors=n_neighbors_mds_1)


#Geodesicly complete distances
dist_geo_true = scipy.sparse.csgraph.shortest_path(dist_trimmed_true, directed=False)
dist_geo_observed = scipy.sparse.csgraph.shortest_path(dist_trimmed_observed, directed=False)
dist_geo_local = scipy.sparse.csgraph.shortest_path(dist_trimmed_local, directed=False)
dist_geo_local_dense = scipy.sparse.csgraph.shortest_path(dist_trimmed_local_dense, directed=False)
if net_flag:
    dist_geo_net = scipy.sparse.csgraph.shortest_path(dist_trimmed_net, directed=False)


n_distances_plotted = 10000

dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_geo_true, 'True intrinsic Geodesic distances', 'True intrinsic Euclidean vs geodesic distances', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/true_geodesic_vs_intrinsic.png", bbox_inches='tight')

dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_local_dense, 'Approximated intrinsic Euclidean distances', 'Dense intrinsic distance approx', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_local_dense.png", bbox_inches='tight')

dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_trimmed_local_dense, 'Approximated intrinsic Euclidean distances', 'Dense intrinsic distance approx - KNN', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_local_dense_knn.png", bbox_inches='tight')

dist_scatter_plot(dist_geo_true, 'True intrinsic Geodesic distances', dist_geo_local_dense, 'Approximated intrinsic Geodesic distances', 'Dense Geodesic distance approx', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_geo_local_dense.png", bbox_inches='tight')


dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_local, 'Approximated intrinsic Euclidean distances', 'Sparse intrinsic distance approx', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_local.png", bbox_inches='tight')

dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_trimmed_local, 'Approximated intrinsic Euclidean distances', 'Sparse intrinsic distance approx - KNN', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_local_knn.png", bbox_inches='tight')

dist_scatter_plot(dist_geo_true, 'True intrinsic Geodesic distances', dist_geo_local, 'Approximated intrinsic Geodesic distances', 'Sparse Geodesic distance approx', n_dist=n_distances_plotted, flag=True)
plt.savefig("plot_temp/dist_geo_local.png", bbox_inches='tight')

if net_flag:
    dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_net, 'Approximated intrinsic Euclidean distances', 'Net intrinsic distance approx', n_dist=n_distances_plotted, flag=True)
    plt.savefig("plot_temp/dist_net.png", bbox_inches='tight')

    dist_scatter_plot(dist_true, 'True intrinsic Euclidean distances', dist_trimmed_net, 'Approximated intrinsic Euclidean distances', 'Net intrinsic distance approx - KNN', n_dist=n_distances_plotted, flag=True)
    plt.savefig("plot_temp/dist_net_knn.png", bbox_inches='tight')

    dist_scatter_plot(dist_geo_true, 'True intrinsic Geodesic distances', dist_geo_net, 'Approximated intrinsic Geodesic distances', 'Net Geodesic distance approx', n_dist=n_distances_plotted, flag=True)
    plt.savefig("plot_temp/dist_geo_net.png", bbox_inches='tight')


# Sub-sampling clusters for Kernel method
n_points_used_for_clusters_2 = min(n_points_used_for_clusters, n_points_used_for_clusters_2)
points_used_for_clusters_indexes_2 = numpy.random.choice(n_points_used_for_clusters, size=n_points_used_for_clusters_2, replace=False)

intrinsic_process_clusters_2 = intrinsic_process_base_clusters[:, points_used_for_clusters_indexes_2]
noisy_sensor_base_cluster_2 = noisy_sensor_base_clusters[:, points_used_for_clusters_indexes_2]
color_map_clusters_2 = color_map_clusters[points_used_for_clusters_indexes_2, :]

dist_2_true = dist_true[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_2_observed = dist_observed[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_2_local = dist_local[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_2_local_dense = dist_local_dense[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]

if net_flag:
    dist_2_net = dist_net[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]

dist_geo_2_true = dist_geo_true[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_geo_2_observed = dist_geo_observed[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_geo_2_local = dist_geo_local[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]
dist_geo_2_local_dense = dist_geo_local_dense[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]

if net_flag:
    dist_geo_2_net = dist_geo_net[points_used_for_clusters_indexes_2, :][:, points_used_for_clusters_indexes_2]

dist_trimmed_2_observed = trim_distances(dist_2_observed, dist_geo_2_observed, n_neighbors=n_neighbors_mds_2)
dist_trimmed_2_local = trim_distances(dist_geo_2_local, dist_geo_2_local, n_neighbors=n_neighbors_mds_2)
dist_trimmed_2_local_dense = trim_distances(dist_geo_2_local_dense, dist_geo_2_local_dense, n_neighbors=n_neighbors_mds_2)
dist_trimmed_2_true = trim_distances(dist_geo_2_true, dist_geo_2_true, n_neighbors=n_neighbors_mds_2)

if net_flag:
    dist_trimmed_2_net = trim_distances(dist_geo_2_net, dist_2_observed, n_neighbors=n_neighbors_mds_2)

if embed_flag:
    mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iterations, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
    embedding_standard_isomap = mds.fit(dist_geo_2_observed, dist_2_true).embedding_.T

    mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iterations, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
    embedding_intrinsic_isomap_dense = mds.fit(dist_geo_2_local_dense, dist_2_true).embedding_.T
    embedding_intrinsic_isometric_dense = intrinsic_isometric_mapping(approx_intrinsic_geo_dists=dist_geo_2_local_dense, approx_intrinsic_euc_dists=dist_2_local_dense,  approx_intrinsic_euc_dists_trimmed=dist_trimmed_2_local_dense, true_intrinsic_euc_dists=dist_2_true, intrinsic_points=intrinsic_process_clusters_2, dim_intrinsic=dim_intrinsic, n_mds_iters=n_mds_iterations, mds_stop_threshold=mds_stop_threshold, n_clusters=n_clusters_isometric_mapping, size_patch_start=size_patch_start, size_patch_step=size_patch_step)

    mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iterations, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
    embedding_intrinsic_isomap = mds.fit(dist_geo_2_local, dist_2_true).embedding_.T
    embedding_intrinsic_isometric = intrinsic_isometric_mapping(approx_intrinsic_geo_dists=dist_geo_2_local, approx_intrinsic_euc_dists=dist_2_local,  approx_intrinsic_euc_dists_trimmed=dist_trimmed_2_local, true_intrinsic_euc_dists=dist_2_true, intrinsic_points=intrinsic_process_clusters_2, dim_intrinsic=dim_intrinsic, n_mds_iters=n_mds_iterations, mds_stop_threshold=mds_stop_threshold, n_clusters=n_clusters_isometric_mapping, size_patch_start=size_patch_start, size_patch_step=size_patch_step)

    if net_flag:
        mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iterations, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
        embedding_intrinsic_isomap_net = mds.fit(dist_geo_2_net, dist_2_true).embedding_.T
        embedding_intrinsic_isometric_net = intrinsic_isometric_mapping(approx_intrinsic_geo_dists=dist_geo_2_net, approx_intrinsic_euc_dists=dist_2_net, approx_intrinsic_euc_dists_trimmed=dist_trimmed_2_net, true_intrinsic_euc_dists=dist_2_true, intrinsic_points=intrinsic_process_clusters_2, dim_intrinsic=dim_intrinsic, n_mds_iters=n_mds_iterations, mds_stop_threshold=mds_stop_threshold, n_clusters=n_clusters_isometric_mapping, size_patch_start=size_patch_start, size_patch_step=size_patch_step)

    n_points_for_stress_calc = 200

    print_process(embedding_standard_isomap, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Standard Isomap Embedding", align_points=intrinsic_process_clusters_2)
    plt.savefig("plot_temp/standard_isomap_embedding.png", bbox_inches='tight')
    stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_standard_isomap, titleStr="Standard Isomap Stress", n_points=n_points_for_stress_calc)
    print('standard_isomap:', stress_normalized)
    plt.savefig("plot_temp/standard_isomap_stress.png", bbox_inches='tight')

    print_process(embedding_intrinsic_isomap_dense, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isomap Embedding - True metric", align_points=intrinsic_process_clusters_2)
    plt.savefig("plot_temp/embedding_intrinsic_isomap_local_dense.png", bbox_inches='tight')
    stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isomap_dense, titleStr="Intrinsic Isomap Stress - True metric", n_points=n_points_for_stress_calc)
    plt.savefig("plot_temp/stress_intrinsic_isomap_local_dense.png", bbox_inches='tight')
    print('intrinsic_isomap_dense:', stress_normalized)

    print_process(embedding_intrinsic_isometric_dense, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isometric Embedding - True metric", align_points=intrinsic_process_clusters_2)
    plt.savefig("plot_temp/embedding_intrinsic_isometric_local_dense.png", bbox_inches='tight')
    stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isometric_dense, titleStr="Intrinsic Isometric Stress - True metric", n_points=n_points_for_stress_calc)
    plt.savefig("plot_temp/stress_intrinsic_isometric_local_dense.png", bbox_inches='tight')
    print('intrinsic_isometric_dense:', stress_normalized)

    print_process(embedding_intrinsic_isomap, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isomap Embedding - Local metric", align_points=intrinsic_process_clusters_2)
    plt.savefig("plot_temp/embedding_intrinsic_isomap_local.png", bbox_inches='tight')
    stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isomap, titleStr="Intrinsic Isomap Stress - Local metric", n_points=n_points_for_stress_calc)
    plt.savefig("plot_temp/stress_intrinsic_isomap_local.png", bbox_inches='tight')
    print('intrinsic_isomap:', stress_normalized)

    print_process(embedding_intrinsic_isometric, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isometric Embedding - Local metric", align_points=intrinsic_process_clusters_2)
    plt.savefig("plot_temp/embedding_intrinsic_isometric_local.png", bbox_inches='tight')
    stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isometric, titleStr="Intrinsic Isometric Embedding - Local metric", n_points=n_points_for_stress_calc)
    plt.savefig("plot_temp/stress_intrinsic_isometric_local.png", bbox_inches='tight')
    print('intrinsic_isometric:', stress_normalized)

    if net_flag:
        print_process(embedding_intrinsic_isomap_net, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isomap Embedding - Net metric", align_points=intrinsic_process_clusters_2)
        stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isomap_net, titleStr="Intrinsic Isomap Stress - Net metric", n_points=n_points_for_stress_calc)
        print('intrinsic_isomap_net:', stress_normalized)
        print_process(embedding_intrinsic_isometric_net, bounding_shape=None, color_map=color_map_clusters_2, titleStr="Intrinsic Isometric Embedding - Net metric", align_points=intrinsic_process_clusters_2)
        plt.savefig("plot_temp/embedding_intrinsic_isometric_net.png", bbox_inches='tight')
        stress_normalized = embedding_score(intrinsic_process_clusters_2, embedding_intrinsic_isometric_net, titleStr="Intrinsic Isometric Stress - Net metric", n_points=n_points_for_stress_calc)
        plt.savefig("plot_temp/stress_intrinsic_isometric_net.png", bbox_inches='tight')
        print('intrinsic_isometric_net:', stress_normalized)

print("Finish")

plt.show(block=True)
plt.close("all")