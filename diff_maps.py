from __future__ import print_function
from __future__ import absolute_import
from DataGeneration import print_process, print_drift, create_color_map
from Util import *
from sklearn import manifold
import numpy


def diff_maps(points_cluster_index, noisy_sensor_base, noisy_sensor_step, intrinsic_variance, intrinsic_process_base, intrinsic_process_step, dim_intrinsic = 0, n_neighbors_cov = 0, n_neighbors_mds = 0, ax_limits_inv=None, ax_limits_gen=None):

    color_map = create_color_map(intrinsic_process_base)

    n_cluster_points = points_cluster_index.shape[0]

    all_points = noisy_sensor_base
    cluster_points_int = intrinsic_process_base[:, points_cluster_index]
    cluster_points = noisy_sensor_base[:, points_cluster_index]
    #cluster_process(cluster_points, all_points, n_neighbors)
    distance_metrix = scipy.spatial.distance.cdist(cluster_points.T, all_points.T, metric='sqeuclidean')
    knn_indexes = numpy.argsort(distance_metrix, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_neighbors_cov+1]

    diff_clusters = noisy_sensor_step[:, knn_indexes] - noisy_sensor_base[:, knn_indexes]
    ''''
    ax_inv = print_process(intrinsic_process_base, indexs=points_cluster_index, bounding_shape=None,
                           color_map=color_map,
                           titleStr="Intrinsic Space")
    ax_inv.set_xlabel("$\displaystyle x_1$")
    ax_inv.set_ylabel("$\displaystyle x_2$")
    # plt.close()
    # print_process(exact_sensor_base, indexs=points_plot_index, bounding_shape=None, color_map=color_map, titleStr="Noiseless Sensor Process")
    ax_gen = print_process(noisy_sensor_base, indexs=points_cluster_index, bounding_shape=None, color_map=color_map,
                           titleStr="Measurement Space")
    ax_gen.set_xlabel("$\displaystyle y_1$")
    ax_gen.set_ylabel("$\displaystyle y_2$")
    plt.axis([-0.2, 2, -1, 1.2])

    ax_inv.scatter(
        intrinsic_process_base[0, points_cluster_index[:10]] + intrinsic_process_step[0, knn_indexes[:10, :]].T -
        intrinsic_process_base[0, knn_indexes[:10, :]].T,
        intrinsic_process_base[1, points_cluster_index[:10]] + intrinsic_process_step[1, knn_indexes[:10, :]].T -
        intrinsic_process_base[1, knn_indexes[:10, :]].T)
    ax_gen.scatter(noisy_sensor_base[0, points_cluster_index[:10]] + noisy_sensor_step[0, knn_indexes[:10, :]].T -
                   noisy_sensor_base[0, knn_indexes[:10, :]].T,
                   noisy_sensor_base[1, points_cluster_index[:10]] + intrinsic_process_step[1, knn_indexes[:10, :]].T -
                   intrinsic_process_base[1, knn_indexes[:10, :]].T)

    ax_gen.scatter(noisy_sensor_base[0, points_cluster_index[:10]] + noisy_sensor_step[0, knn_indexes[:10, :]].T -
                   noisy_sensor_base[0, knn_indexes[:10, :]].T,
                   noisy_sensor_base[1, points_cluster_index[:10]] + intrinsic_process_step[1, knn_indexes[:10, :]].T -
                   intrinsic_process_base[1, knn_indexes[:10, :]].T)

    print_process(intrinsic_process_base, indexs=points_cluster_index[:10], bounding_shape=None,
                  color_map=color_map,
                  titleStr="Intrinsic Space", ax=ax_inv)
    ax_inv.set_xlabel("$\displaystyle x_1$")
    ax_inv.set_ylabel("$\displaystyle x_2$")
    '''
    # tangent metric estimation
    cov_list_def, cov_list_full = get_metrics_from_points(intrinsic_process_base[:, points_cluster_index], input_base, input_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance)

    for x in range(0, knn_indexes.shape[0]):
        temp_cov = numpy.cov(diff_clusters[:, x, :])
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1 / s_def
        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float32).eps:
            s_full[dim_intrinsic:] = numpy.finfo(numpy.float32).eps

        s_full = 1 / s_full
        cov_list_def[x] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        cov_list_full[x] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))

    # distance estimation and training pair preparation


    dist_mat_true = calc_dist(intrinsic_process_base[:, points_cluster_index])
    dist_mat_measured = calc_dist(noisy_sensor_base[:, points_cluster_index])
    dist_mat_full = calc_dist(noisy_sensor_base[:, points_cluster_index], cov_list_full)
    dist_mat_def = calc_dist(noisy_sensor_base[:, points_cluster_index], cov_list_def)

    '''
    dist_mat_full = calc_dist
    for i_x in range(0, knn_indexes.shape[0]):
        for i_y in range(0, knn_indexes.shape[0]):
            if i_x != i_y:
                dif_vect = noisy_sensor_base[:, points_cluster_index[i_x]] - noisy_sensor_base[:, points_cluster_index[i_y]]
                dist_mat_full[i_x, i_y] = intrinsic_variance * 1 / 2 * (numpy.dot(dif_vect, numpy.dot(cov_list_full[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_full[i_y], dif_vect)))
                dist_mat_def[i_x, i_y] = intrinsic_variance * 1 / 2 * (numpy.dot(dif_vect, numpy.dot(cov_list_def[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_def[i_y], dif_vect)))
                dif_vect_real = intrinsic_process_base[:, points_cluster_index[i_x]] - intrinsic_process_base[:, points_cluster_index[i_y]]
                dist_mat_true[i_x, i_y] = numpy.dot(dif_vect_real, dif_vect_real)
                dist_mat_measured[i_x, i_y] = numpy.dot(dif_vect, dif_vect)

    '''
    '''
    ax_cov = print_process(noisy_sensor_base[:, points_cluster_index[:400]], color_map=color_map[points_cluster_index[:400], :], titleStr = "Empirically Estimated Covariances", covs=cov_list[:400])
    ax_cov.set_xlabel("$\displaystyle y_1$")
    ax_cov.set_ylabel("$\displaystyle y_2$")
    plt.axis([-0.2, 2, -1, 1.2])

    ax_cov = print_process(noisy_sensor_base[:, points_cluster_index[:400]], color_map=color_map[points_cluster_index[:400], :], titleStr = "Analytically Calculated Covariances", covs=cov_exact[:400])
    ax_cov.set_xlabel("$\displaystyle y_1$")
    ax_cov.set_ylabel("$\displaystyle y_2$")
    plt.axis([-0.2, 2, -1, 1.2])
    '''

    isomap_approx = trim_distances(dist_mat_def, dist_mat_measured, n_neighbors_mds)
    isomap_true = trim_distances(dist_mat_true, dist_mat_true, n_neighbors_mds)
    isomap_measured = trim_distances(dist_mat_measured, dist_mat_measured, n_neighbors_mds)

    isomap_approx = scipy.sparse.csgraph.shortest_path(isomap_approx, directed=False)
    isomap_true = scipy.sparse.csgraph.shortest_path(isomap_true, directed=False)
    isomap_measured = scipy.sparse.csgraph.shortest_path(isomap_measured, directed=False)

    #initial_guess_base = isomap_projection.fit_transform(noisy_sensor_base_for_guess.T)

    mds = manifold.MDS(n_components=2, max_iter=20000, eps=1e-9, dissimilarity="precomputed", n_jobs=1)
    pos_1 = mds.fit(isomap_approx).embedding_
    pos_2 = mds.fit(isomap_true).embedding_
    pos_3 = mds.fit(isomap_measured).embedding_

    diff_embedding = calc_diff_map(isomap_approx, dims=2, factor=2)

    diff_embedding_non_int = calc_diff_map(dist_mat_true, dims=2, factor=2)

    return pos_1, pos_2, pos_3, diff_embedding, diff_embedding_non_int
