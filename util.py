import scipy
from scipy.spatial.distance import cdist
import numpy
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets


def get_short_range_dist(sampled_process, ind_cluster_center_points, n_knn, intrinsic_process, approximated, k=5, process_var=1):
    n_cluster_points = ind_cluster_center_points.__len__()
    dim_sampled = sampled_process.shape[0]
    dim_intrinsic = intrinsic_process.shape[0]

    # cluster
    distance_metric = cdist(sampled_process[:, ind_cluster_center_points].T, sampled_process[:, :-1].T, 'sqeuclidean')
    knn_indexes = numpy.argsort(distance_metric, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_knn]

    diff_clusters = sampled_process[:, knn_indexes + 1] - sampled_process[:, knn_indexes]

    # tangent metric estimation
    cov_list_full = [None] * n_cluster_points
    cov_list_def = [None] * n_cluster_points

    for x in range(0, knn_indexes.shape[0]):
        temp_cov = numpy.cov(diff_clusters[:, x, :])
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1/s_def
        if s_def[dim_intrinsic:] < numpy.finfo(numpy.float32).eps:
            s_full[dim_intrinsic:] = numpy.finfo(numpy.float32).eps

        s_full = 1/s_full
        cov_list_def[x] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        cov_list_full[x] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))

    # distance estimation and training pair preparation

    approx_dist = numpy.zeros((n_cluster_points, n_cluster_points))

    distFull = numpy.zeros((n_cluster_points, n_cluster_points))
    distDef = numpy.zeros((n_cluster_points, n_cluster_points))

    for i_x in range(0, knn_indexes.shape[0]):
        for i_y in range(0, knn_indexes.shape[0]):
            if i_x != i_y:

                distFull[i_x, i_y] = 0
                distDef[i_x, i_y] = 0

                if approximated:

                    #temp_vect = numpy.dot(cov_list_full[i_x], sampled_process[:, ind_cluster_center_points[i_x]]) + numpy.dot(cov_list_full[i_y], sampled_process[:, ind_cluster_center_points[i_y]])
                    #mid_point = numpy.dot(numpy.linalg.inv(cov_list_full[i_x] + cov_list_full[i_y]), temp_vect)
                    #dif_vect1 = mid_point - sampled_process[:, ind_cluster_center_points[i_x]]
                    #dif_vect2 = mid_point - sampled_process[:, ind_cluster_center_points[i_y]]

                    #curr_dist = numpy.square(numpy.sqrt(numpy.dot(dif_vect1, numpy.dot(cov_list_full[i_x], dif_vect1))) + numpy.sqrt(numpy.dot(dif_vect2, numpy.dot(cov_list_full[i_y], dif_vect2))))
                    #curr_dist = numpy.dot(dif_vect1, numpy.dot(cov_list_def[i_x], dif_vect1)) + numpy.dot(dif_vect2, numpy.dot(cov_list_def[i_y], dif_vect2))

                    dif_vect = sampled_process[:, ind_cluster_center_points[i_x]] - sampled_process[:, ind_cluster_center_points[i_y]]

                    distFull[i_x, i_y] = process_var*1/2*(numpy.dot(dif_vect, numpy.dot(cov_list_full[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_full[i_y], dif_vect)))
                    distDef[i_x, i_y] = process_var*1/2*(numpy.dot(dif_vect, numpy.dot(cov_list_def[i_x], dif_vect)) + numpy.dot(dif_vect, numpy.dot(cov_list_def[i_y], dif_vect)))


                    '''dif_vect = intrinsic_process[:, ind_cluster_center_points[i_x]] - intrinsic_process[:, ind_cluster_center_points[i_y]]
                    curr_dist = numpy.dot(dif_vect, dif_vect)
                    labels += [curr_dist]
                    weights += [curr_dist]'''
                else:
                    dif_vect = intrinsic_process[:, ind_cluster_center_points[i_x]] - intrinsic_process[:, ind_cluster_center_points[i_y]]
                    curr_dist = numpy.dot(dif_vect, dif_vect)
                    distFull[i_x, i_y] = curr_dist
                    distDef[i_x, i_y] = curr_dist

    n_dist = distFull.shape[1]

    for i_x in range(0, knn_indexes.shape[0]):
        sortedDistances = numpy.sort(distFull[i_x, :], kind='quicksort')
        sigma = sortedDistances[k]
        ind_used = numpy.where(distFull[i_x, :] > sigma)
        distDef[i_x, ind_used] = 0


    return numpy.sqrt(distDef)


def print_potential(func, x_low=-0.25, x_high=0.25, y_low=-0.25, y_high=0.25, step=0.01):
    x_range = numpy.arange(x_low, x_high, step)
    n_x = x_range.shape[0]
    y_range = numpy.arange(y_low, y_high, step)
    n_y = y_range.shape[0]
    [X_grid, Y_grid] = numpy.meshgrid(x_range, y_range)
    potential = numpy.empty((n_x, n_y))
    for i in range(0, n_x):
        for j in range(0, n_y):
            potential[i, j] = func(numpy.asarray([X_grid[j, i], Y_grid[j, i]]))

    plt.figure()
    plt.contour(X_grid.T, Y_grid.T, potential, 15, linewidths=0.5, colors='k')
    plt.contourf(X_grid.T, Y_grid.T, potential, 15, cmap=plt.cm.rainbow, vmin=potential.min(), max=potential.max())
    plt.colorbar()  # draw colorbar
    ax = plt.gca()
    ax.set_title("Potential Level Contours")
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X_grid.T, Y_grid.T, potential, rstride=1, cstride=1, linewidth=0, antialiased=False)
    ax.set_title("Potential Surface")
    plt.show(block=False)


def embedding_score(ground_truth, embedding, titleStr, n_points=200):

    ground_truth = ground_truth[:, :n_points]
    embedding = embedding[:, :n_points]
    n_points = ground_truth.shape[1]
    dist_mat_ground_truth = scipy.spatial.distance.cdist(ground_truth.T, ground_truth.T)
    dist_mat_embedding = scipy.spatial.distance.cdist(embedding.T, embedding.T)
    dist_mat_ground_truth_squared = (dist_mat_ground_truth*dist_mat_ground_truth)
    weight_matrix = numpy.ones(dist_mat_ground_truth_squared.shape)
    #weight_matrix[numpy.where(dist_mat_ground_truth_squared != 0)] = 1 / dist_mat_ground_truth_squared[numpy.where(dist_mat_ground_truth_squared != 0)]
    stress_normalized = numpy.sqrt(numpy.sum(numpy.square(dist_mat_ground_truth-dist_mat_embedding))/numpy.sum(numpy.square(dist_mat_ground_truth)))
    #stress_normalized = numpy.sqrt(numpy.sum(numpy.square(dist_mat_ground_truth-dist_mat_embedding)*weight_matrix)/numpy.sum(weight_matrix))
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.scatter(dist_mat_ground_truth.reshape((1, n_points*n_points)), dist_mat_embedding[:].reshape((1, n_points*n_points)), s=1, c="k")
    ax.plot([0, numpy.max(dist_mat_ground_truth)], [0, numpy.max(dist_mat_ground_truth)], c='r')
    ax.set_xlim([0, numpy.max(dist_mat_ground_truth)*1.15])
    ax.set_ylim([0, numpy.max(dist_mat_ground_truth)*1.15])
    ax.set_xlabel('True distances')
    ax.set_ylabel('Distances in embedding')
    fig.canvas.set_window_title(titleStr)
    ax.set_title('Stress=%.4f' % stress_normalized)
    plt.show(block=False)
    return stress_normalized

def dist_scatter_plot(dist_1, titleStr_1, dist_2, titleStr_2, titleStr_main, n_dist = 200, flag=False):

    dist_1_flat = numpy.ndarray.flatten(dist_1)
    dist_2_flat = numpy.ndarray.flatten(dist_2)
    where_non_zero = numpy.nonzero(dist_2_flat)
    dist_1_flat = dist_1_flat[where_non_zero]
    dist_2_flat = dist_2_flat[where_non_zero]
    n_dist = min(n_dist, dist_1_flat.shape[0])
    inx_n_dist = numpy.random.choice(dist_1_flat.shape[0], size=n_dist, replace=False)
    dist_1_sub = dist_1_flat[inx_n_dist]
    dist_2_sub = dist_2_flat[inx_n_dist]
    fig = plt.figure()
    fig.canvas.set_window_title(titleStr_main)
    ax = fig.add_subplot(111)
    ax.scatter(dist_1_sub, dist_2_sub, s=1, c="k")
    ax.set_xlim([0, numpy.max(dist_1_sub)*1.15])
    ax.set_ylim([0, numpy.max(dist_2_sub)*1.15])
    ax.set_xlabel(titleStr_1)
    ax.set_ylabel(titleStr_2)
    if flag:
        ax.set_xlim([0, numpy.max(dist_1_sub) * 1.15])
        ax.set_ylim([0, numpy.max(dist_1_sub) * 1.15])
        ax.plot([0, numpy.max(dist_1)], [0, numpy.max(dist_1)], c='r')

    plt.show(block=False)


# Input: expects Nx3 matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform(A, B):
    dim = A.shape[0]

    assert len(A) == len(B)

    if dim == 2:
        A = numpy.pad(A, pad_width=((0, 1), (0, 1)), mode='constant')
        B = numpy.pad(B, pad_width=((0, 1), (0, 1)), mode='constant')

    N = A.shape[1]  # total points

    A = A.T
    B = B.T

    centroid_A = numpy.mean(A, axis=0)

    centroid_B = numpy.mean(B, axis=0)

    # centre the points
    AA = A - numpy.tile(centroid_A, (N, 1))
    BB = B - numpy.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = numpy.dot(numpy.transpose(AA), BB)

    U, S, Vt = numpy.linalg.svd(H)

    R = numpy.dot(Vt.T, U.T)

    # special reflection case
    if numpy.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = numpy.dot(Vt.T, U.T)
    t = -numpy.dot(R, centroid_A.T) + centroid_B.T

    if dim == 2:
        R = R[0:2, 0:2]
        t = t[0:2]

    return R, t

def calc_dist(points, metrics=None):

    dim_points = points.shape[0]
    n_points = points.shape[1]
    dist_mat = numpy.zeros((n_points, n_points))

    if metrics is None:
        metrics = [None] * n_points
        for i_x in range(0, n_points):
            metrics[i_x] = numpy.eye(dim_points)

    for i_x in range(0, n_points):
            tmp1 = numpy.dot(metrics[i_x], points[:, i_x])
            a2 = numpy.dot(points[:, i_x].T, tmp1)
            b2 = sum(points * numpy.dot(metrics[i_x], points), 0)
            ab = numpy.dot(points.T, tmp1)
            dist_mat[:, i_x] = numpy.real((numpy.tile(a2, (n_points, 1)) + b2.T.reshape((n_points, 1)) - 2 * ab.reshape((n_points, 1))).reshape(n_points))

    dist_mat = numpy.abs((dist_mat + dist_mat.T)/2)

    return dist_mat


def get_metrics_from_points_diffusion(cluster_centers, input_base, input_step, n_neighbors_cov, dim_intrinsic, intrinsic_variance, pre_calc_drift=None):

    n_cluster_points = cluster_centers.shape[1]
    distance_matrix = scipy.spatial.distance.cdist(cluster_centers.T, input_base.T, metric='sqeuclidean')
    knn_indexes = numpy.argsort(distance_matrix, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_neighbors_cov+1]

    diff_clusters = input_step[:, knn_indexes] - input_base[:, knn_indexes]

    if True:
        max_range = numpy.array([cluster_centers[0].max() - cluster_centers[0].min(), cluster_centers[1].max() - cluster_centers[1].min(), cluster_centers[2].max() - cluster_centers[2].min()]).max() / 2.0
        mid_x = (cluster_centers[0].max() + cluster_centers[0].min()) * 0.5
        mid_y = (cluster_centers[1].max() + cluster_centers[1].min()) * 0.5
        mid_z = (cluster_centers[2].max() + cluster_centers[2].min()) * 0.5
        i_point = 1
        fig = plt.figure()
        fig.canvas.set_window_title("Clustered process jumps")
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        ax.scatter(input_base[0, :], input_base[1, :], input_base[2, :], s=1)
        ax.scatter(input_base[0, knn_indexes[i_point, :]], input_base[1, knn_indexes[i_point, :]], input_base[2, knn_indexes[i_point, :]], c="r")
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show(block=False)
        plt.axis('equal')
        fig = plt.figure()
        fig.canvas.set_window_title("Clustered process jumps")
        ax = fig.add_subplot(111, projection='3d', aspect='equal')
        ax.scatter(cluster_centers[0, :], cluster_centers[1, :], cluster_centers[2, :], s=1)
        for i_knn in range(0, n_neighbors_cov):
            jump_vact = [input_step[0, knn_indexes[i_point, i_knn]] - input_base[0, knn_indexes[i_point, i_knn]], input_step[1, knn_indexes[i_point, i_knn]] - input_base[1, knn_indexes[i_point, i_knn]], input_step[2, knn_indexes[i_point, i_knn]] - input_base[2, knn_indexes[i_point, i_knn]]]
            ax.plot([cluster_centers[0, i_point], cluster_centers[0, i_point]+jump_vact[0]], [cluster_centers[1, i_point], cluster_centers[1, i_point]+jump_vact[1]], [cluster_centers[2, i_point], cluster_centers[2, i_point]+jump_vact[2]], c='g')
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        plt.show(block=False)
        plt.axis('equal')

    cov_list = [None] * n_cluster_points
    metric_list_def = [None] * n_cluster_points
    metric_list_full = [None] * n_cluster_points
    drift_list = [None] * n_cluster_points
    reg_list = [None] * n_cluster_points
    noise_list = [None] * n_cluster_points
    for x in range(0, knn_indexes.shape[0]):
        if pre_calc_drift is None:
            drift_list[x] = numpy.mean(diff_clusters[:, x, :], axis=1)
            temp_cov = (diff_clusters[:, x, :].T-drift_list[x].T).T
        else:
            temp_cov = (diff_clusters[:, x, :].T-pre_calc_drift[x].T).T

        temp_cov = numpy.dot(temp_cov, temp_cov.T)/(n_neighbors_cov-1)
        U, s, V = numpy.linalg.svd(temp_cov)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        noise_list[x] = s[dim_intrinsic:].sum()
        noise_level = s[dim_intrinsic:].mean()
        s_def = 1 / (s_def-noise_level)
        s_full = 1 / (s_full-noise_level)
        cov_list[x] = temp_cov
        metric_list_def[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        metric_list_full[x] = intrinsic_variance*numpy.dot(U, numpy.dot(numpy.diag(s_full), V))
        reg_list[x] = numpy.eye(N=temp_cov.shape[0])

    noise_array = numpy.asarray(noise_list)

    return cov_list, metric_list_def, metric_list_full, drift_list, reg_list, noise_array.mean()


def get_metrics_from_points_array(noisy_sensor_base_clusters, noisy_sensor_step_clusters, dim_intrinsic, sensor_array_matrix, intrinsic_variance):

    dim_measured = noisy_sensor_base_clusters.shape[0]
    n_cluster_points = noisy_sensor_base_clusters.shape[1]

    metric_list_def = [None] * n_cluster_points
    metric_list_full = [None] * n_cluster_points
    cov_list = [None] * n_cluster_points
    drift_list = [None] * n_cluster_points
    reg_list = [None] * n_cluster_points
    noise_list = [None] * n_cluster_points

    sensor_array_matrix_cov_inv = numpy.linalg.pinv(numpy.dot(sensor_array_matrix.T, sensor_array_matrix))/intrinsic_variance

    n_sensor_array_matrix_obs = sensor_array_matrix.shape[1]

    for i_cluster in range(n_cluster_points):
        directional_derivatives = numpy.zeros((dim_measured, n_sensor_array_matrix_obs))
        for i_dir in range(n_sensor_array_matrix_obs):
            directional_derivatives[:, i_dir] = (noisy_sensor_step_clusters[:, i_cluster, i_dir] - noisy_sensor_base_clusters[:, i_cluster])

        drift_list[i_cluster] = numpy.mean((noisy_sensor_step_clusters[:, i_cluster, :].T-noisy_sensor_base_clusters[:, i_cluster].T).T, axis=1)*0
        temp_cov = intrinsic_variance*numpy.dot(directional_derivatives, numpy.dot(sensor_array_matrix_cov_inv, directional_derivatives.T))
        cov_list[i_cluster] = temp_cov
        U, s, V = numpy.linalg.svd(temp_cov/intrinsic_variance)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        noise_list[i_cluster] = s[dim_intrinsic:].sum()
        s_def = 1 / (s_def)
        s_full = 1 / (s_full)
        cov_list[i_cluster] = temp_cov
        metric_list_def[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        metric_list_full[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))
        reg_list[i_cluster] = numpy.eye(N=temp_cov.shape[0])

    noise_array = numpy.asarray(noise_list)

    return cov_list, metric_list_def, metric_list_full, drift_list, reg_list, noise_array.mean()


def get_metrics_from_points_bursts(noisy_sensor_base_clusters, noisy_sensor_step_clusters, dim_intrinsic, intrinsic_variance):

    n_cluster_points = noisy_sensor_base_clusters.shape[1]

    metric_list_def = [None] * n_cluster_points
    metric_list_full = [None] * n_cluster_points
    cov_list = [None] * n_cluster_points
    drift_list = [None] * n_cluster_points
    reg_list = [None] * n_cluster_points
    noise_list = [None] * n_cluster_points

    n_sensor_array_matrix_obs = noisy_sensor_step_clusters.shape[2]

    for i_cluster in range(n_cluster_points):
        temp_mean = noisy_sensor_step_clusters[:, i_cluster, :].mean(axis=1)
        directional_derivatives = (noisy_sensor_step_clusters[:, i_cluster, :].T - temp_mean.T).T
        temp_cov = numpy.dot(directional_derivatives, directional_derivatives.T)/(n_sensor_array_matrix_obs-1)
        cov_list[i_cluster] = temp_cov
        U, s, V = numpy.linalg.svd(temp_cov/intrinsic_variance)
        s_full = numpy.copy(s)
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        noise_list[i_cluster] = intrinsic_variance*s[dim_intrinsic:].mean()
        noise_level = s[dim_intrinsic:].mean()
        s_def = 1 / (s_def-noise_level)
        s_full = 1 / (s_full-noise_level)
        metric_list_def[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
        metric_list_full[i_cluster] = numpy.dot(U, numpy.dot(numpy.diag(s_full), V))
        reg_list[i_cluster] = numpy.eye(N=temp_cov.shape[0])

    noise_array = numpy.asarray(noise_list)

    return cov_list, metric_list_def, metric_list_full, drift_list, reg_list, noise_array.mean()


def get_metrics_from_net(non_local_tangent_net, noisy_sensor_clusters):

    dim_measurement = noisy_sensor_clusters.shape[0]
    n_points = noisy_sensor_clusters.shape[1]
    metric_list_net = [None] * n_points
    for i_point in range(0, n_points):
        jacobian = non_local_tangent_net.get_jacobian_val(noisy_sensor_clusters[:, i_point].reshape((dim_measurement, 1)))[0, :, :]
        dim_intrinsic = jacobian.shape[1]
        U, s, V = numpy.linalg.svd(numpy.dot(jacobian, jacobian.T))
        s_def = numpy.copy(s)
        s_def[dim_intrinsic:] = float('Inf')
        s_def = 1 / s_def
        metric_list_net[i_point] = numpy.dot(U, numpy.dot(numpy.diag(s_def), V))
    return metric_list_net

def get_drift_from_net(non_local_tangent_net, noisy_sensor_clusters):
    drift = non_local_tangent_net.get_drift_val(noisy_sensor_clusters)
    drift_list = [None] * drift.shape[0]
    for x in range(0, drift.shape[0]):
        drift_list[x] = drift[x, :]
    return drift

def trim_distances(dist_mat, dist_mat_criteria=None, n_neighbors=10):

    n_points = dist_mat.shape[0]

    if dist_mat_criteria is None:
        dist_mat_criteria = numpy.copy(dist_mat)
    if n_neighbors is None:
        n_neighbors = numpy.ceil(n_points*0.1)

    n_points = dist_mat.shape[0]
    knn_indexes = numpy.argsort(dist_mat_criteria, axis=1, kind='quicksort')
    knn_indexes = knn_indexes[:, 1:n_neighbors + 1]
    dist_mat_trim = numpy.zeros((n_points, n_points))
    for i_x in range(0, n_points):
        for i_y in range(0, n_neighbors):
            dist_mat_trim[i_x, knn_indexes[i_x, i_y]] = dist_mat[i_x, knn_indexes[i_x, i_y]]
            dist_mat_trim[knn_indexes[i_x, i_y], i_x] = dist_mat[i_x, knn_indexes[i_x, i_y]]
    return dist_mat_trim

def trim_distances_topo(dist_mat, dist_potential, radius_trim, intrinsic_process):

    n_points = dist_mat.shape[0]
    dist_mat_trim = numpy.zeros(dist_mat.shape)

    [dist_mat, predecessors]  = scipy.sparse.csgraph.shortest_path(dist_mat, directed=False, return_predecessors=True, method='D')

    #fig = plt.figure()
    #x = fig.gca()
    #ax.scatter(intrinsic_process[0, numpy.where(dist_potential<radius_trim)], intrinsic_process[1, numpy.where(dist_potential<radius_trim)], c="r")

    #fig = plt.figure()
    #ax = fig.gca()
    #ax.scatter(intrinsic_process[0, :], intrinsic_process[1, :], c="k")

    for i_x in range(0, n_points):
        for i_y in range(0, n_points):
            pre_temp = i_y
            edge_list = []
            while pre_temp != -9999:
                edge_list.append(dist_potential[pre_temp]<radius_trim)
                pre_temp = predecessors[i_x, pre_temp]
            if all(edge == False for edge in edge_list):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif all(edge == True for edge in edge_list) and dist_mat[i_x, i_y]<2*radius_trim:
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[0]==True) and all(edge == False for edge in edge_list[1:]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[-1]==True) and all(edge == False for edge in edge_list[:-1]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]
            elif (edge_list[1]==True) and (edge_list[-1]== True) and all(edge == False for edge in edge_list[1:-1]):
                dist_mat_trim[i_x, i_y] = dist_mat[i_x, i_y]


    return dist_mat_trim

def calc_diff_map(dist_mat, dims=2, factor=2):
    sigma = numpy.median(dist_mat)/factor
    diff_kernal = numpy.exp(-(dist_mat ** 2) / (2 * sigma ** 2))
    row_sum = numpy.sum(diff_kernal, axis=1)
    normlized_kernal = numpy.dot(numpy.diag(1 / row_sum), diff_kernal)
    U, S, V = numpy.linalg.svd(normlized_kernal)
    return U[:, 1:dims+1].T

def print_metrics(noisy_sensor_clusters, metric_list_full, intrinsic_dim, titleStr, scale, space_mode, elipse, color_map, points_used_for_clusters_indexes, azi, el):
    if noisy_sensor_clusters.shape[0] == 3:
        metric_list = []

        noisy_sensor_clusters = numpy.copy(noisy_sensor_clusters[:, points_used_for_clusters_indexes])
        for i_point in range(points_used_for_clusters_indexes.shape[0]):
            metric_list.append(metric_list_full[points_used_for_clusters_indexes[i_point]])

        color_map = color_map[points_used_for_clusters_indexes, :]
        fig = plt.figure()
        ax = fig.gca(projection='3d', aspect='equal')

        n_points = noisy_sensor_clusters.shape[1]

        if space_mode is True:
            for i_point in range(0, n_points):
                metric = metric_list[i_point]
                center = noisy_sensor_clusters[:, i_point]

                if elipse:
                    U, s, rotation = numpy.linalg.svd(numpy.linalg.pinv(metric))
                    # now carry on with EOL's answer
                    u = numpy.linspace(0.0, 2.0 * numpy.pi, 12)
                    v = numpy.linspace(0.0, numpy.pi, 6)
                    x = int(s[0]>1e-10)*numpy.outer(numpy.cos(u), numpy.sin(v))/3
                    y = int(s[1]>1e-10)*numpy.outer(numpy.sin(u), numpy.sin(v))/3
                    z = 0*numpy.outer(numpy.ones_like(u), numpy.cos(v))/3
                    xp=numpy.zeros(shape=x.shape)
                    yp=numpy.zeros(shape=y.shape)
                    zp=numpy.zeros(shape=z.shape)

                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]):
                            [xp[i, j], yp[i, j], zp[i, j]] = numpy.dot(numpy.asarray([x[i, j], y[i, j], z[i, j]]), numpy.sqrt(scale)*rotation) + center

                    ax.plot_surface(xp, yp, zp, rstride=1, cstride=1, color=color_map[i_point], alpha=1)
                    #ax.quiver(center[0], center[1], center[2], 100*rotation[0, 0], 100*rotation[0, 1], 100*rotation[0, 2], length=0.3, pivot='tail')
                    #ax.quiver(center[0], center[1], center[2], 100*rotation[1, 0], 100*rotation[1, 1], 100*rotation[1, 2], length=0.3, pivot='tail')

                else:
                    [u, s, v] = numpy.linalg.svd(numpy.linalg.pinv(metric))
                    u = numpy.dot(u[:, 0:intrinsic_dim], numpy.diag(numpy.sqrt(s[:intrinsic_dim])))
                    sign = numpy.sign(u[0, 0])
                    ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                              noisy_sensor_clusters[2, i_point], sign*u[0, 0], sign*u[1, 0], sign*u[2, 0],
                              length=3*numpy.sqrt(scale), pivot='tail')
                    sign = numpy.sign(u[0, 1])
                    ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                              noisy_sensor_clusters[2, i_point], sign*u[0, 1], sign*u[1, 1], sign*u[2, 1],
                              length=3*numpy.sqrt(scale), pivot='tail')

        else:

            for i_point in range(0, n_points):
                metric = metric_list[i_point]
                center = noisy_sensor_clusters[:, i_point]

                if elipse:
                    U, s, rotation = numpy.linalg.svd(numpy.linalg.pinv(metric))
                    radii = (numpy.sqrt(s))/3
                    radii[intrinsic_dim:] = 0
                    # now carry on with EOL's answer
                    u = numpy.linspace(0.0, 2.0 * numpy.pi, 20)
                    v = numpy.linspace(0.0, numpy.pi, 3)

                    x = radii[0] * numpy.outer(numpy.cos(u), numpy.sin(v))
                    y = radii[1] * numpy.outer(numpy.sin(u), numpy.sin(v))
                    z = 0 * numpy.outer(numpy.ones_like(u), numpy.cos(v))

                    x_outline = radii[0] * numpy.cos(numpy.linspace(0.0, 2.0 * numpy.pi, 20))
                    y_outline = radii[1] * numpy.sin(numpy.linspace(0.0, 2.0 * numpy.pi, 20))
                    z_outline = 0 * y_outline

                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]):
                            [x[i, j], y[i, j], z[i, j]] = numpy.dot([x[i, j], y[i, j], z[i, j]], numpy.sqrt(scale)*rotation) + center

                    for i in range(x_outline.shape[0]):
                            [x_outline[i], y_outline[i], z_outline[i]] = numpy.dot([x_outline[i], y_outline[i], z_outline[i]], numpy.sqrt(scale)*rotation) + center

                    ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color_map[i_point], alpha=1, linewidth=0.0, shade=False)
                    ax.plot(x_outline, y_outline, z_outline, color='k')

                else:
                    [u, s, v] = numpy.linalg.svd(numpy.linalg.pinv(metric))
                    u = numpy.dot(u[:, 0:intrinsic_dim], numpy.sqrt(scale)*numpy.diag(numpy.sqrt(s[:intrinsic_dim])))
                    sign = numpy.sign(u[0, 0])
                    ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                              noisy_sensor_clusters[2, i_point], sign*u[0, 0], sign*u[1, 0], sign*u[2, 0],
                              length=numpy.linalg.norm(u[:, 0]), pivot='tail')
                    sign = numpy.sign(u[0, 1])
                    ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                              noisy_sensor_clusters[2, i_point], sign*u[0, 1], sign*u[1, 1], sign*u[2, 1],
                              length=numpy.linalg.norm(u[:, 1]), pivot='tail')

        max_range = numpy.array([noisy_sensor_clusters[0].max() - noisy_sensor_clusters[0].min(), noisy_sensor_clusters[1].max() - noisy_sensor_clusters[1].min(), noisy_sensor_clusters[2].max() - noisy_sensor_clusters[2].min()]).max() / 2.0
        mid_x = (noisy_sensor_clusters[0].max() + noisy_sensor_clusters[0].min()) * 0.5
        mid_y = (noisy_sensor_clusters[1].max() + noisy_sensor_clusters[1].min()) * 0.5
        mid_z = (noisy_sensor_clusters[2].max() + noisy_sensor_clusters[2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        fig.canvas.set_window_title(titleStr)
        ax.view_init(el, azi)

    plt.show(block=False)


def print_drift(noisy_sensor_clusters, drift, titleStr):

    if noisy_sensor_clusters.shape[0] == 3:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        drift = drift.T
        n_points = noisy_sensor_clusters.shape[1]
        for i_point in range(0, n_points):
            ax.quiver(noisy_sensor_clusters[0, i_point], noisy_sensor_clusters[1, i_point],
                      noisy_sensor_clusters[2, i_point], drift[i_point][0], drift[i_point][1], drift[i_point][2], length=numpy.linalg.norm(drift[i_point][:]), pivot='tail')
        ax.set_title(titleStr)
        drift = drift.T
        plt.show(block=False)

def trim_non_euc(dist_mat_fill, dist_mat_trust, dim_intrinsic, intrinsic_process_clusters):

    n_points = dist_mat_trust.shape[0]
    dist_mat_trimmed = numpy.zeros((n_points, n_points))
    dist_mat_trimmed_wgt = numpy.zeros((n_points, n_points))
    #indexs_balls = numpy.random.choice(n_points, size=n_points, replace=False)

    for i_point in range(15):
        dist_mat_trust_temp = numpy.array(dist_mat_trust, copy=True)

        knn_indexes = numpy.argsort(dist_mat_fill[i_point], kind='quicksort')
        D_sub_trust_original = dist_mat_fill[knn_indexes, :][:, knn_indexes]

        plt.figure()
        plt.imshow(D_sub_trust_original, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

        n_neighbors_start = 30
        n_neighbors_step = 30

        n_neighbors = n_neighbors_start

        flat = True
        check_list = []
        check_list_X = []
        while flat:
            knn_indexes_sub = knn_indexes[0:n_neighbors]

            numpy.fill_diagonal(dist_mat_fill, 0)

            D_fill_sub = dist_mat_fill[knn_indexes_sub, :][:, knn_indexes_sub]

            #plt.figure()
            #plt.imshow(D_fill_sub, vmin=numpy.min(D_sub_trust_original), vmax=numpy.max(D_sub_trust_original))

            # square it
            D_squared = D_fill_sub ** 2

            # centering matrix
            n = D_squared.shape[0]
            J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

            # perform double centering
            B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

            # find eigenvalues and eigenvectors
            U, eigen_val, V = numpy.linalg.svd(B)
            eigen_vect = V
            eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
            eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])


            expl = numpy.sum(eigen_val[:dim_intrinsic])
            res = numpy.sum(eigen_val[dim_intrinsic:])
            #dis = (D_sub_trust_original*wgt).sum()
            #check = (stress2/dis)
            check_list.append(dim_intrinsic*eigen_val[dim_intrinsic]/expl)

            check_list_X.append(n_neighbors)

            if n_neighbors == n_points:
                break

            n_neighbors = min(numpy.ceil(n_neighbors+n_neighbors_step), n_points)

        plt.figure()
        plt.plot(numpy.asarray(check_list_X), numpy.asarray(check_list))
        plt.show(block=False)

        min_rank = numpy.asarray(check_list).min()

        if (min_rank>0.05):
            break

        min_rank_flex = min_rank*1.2

        ind_neigboors = numpy.where(check_list<min_rank_flex)[-1]

        radius = check_list_X[ind_neigboors[-1]]

        print(i_point)

        knn_indexes_sub = knn_indexes[0:radius]

        numpy.fill_diagonal(dist_mat_fill, 0)

        D_fill_sub = dist_mat_fill[knn_indexes_sub, :][:, knn_indexes_sub]

        # square it
        D_squared = D_fill_sub ** 2

        # centering matrix
        n = D_squared.shape[0]
        J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

        # perform double centering
        B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

        # find eigenvalues and eigenvectors
        U, eigen_val, V = numpy.linalg.svd(B)
        eigen_vect = V
        eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
        eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])

        eigen_vect = eigen_vect[eigen_val_sort_ind]
        eigen_vect = eigen_vect[:dim_intrinsic].T
        guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)


        mds = manifold.MDS(n_components=dim_intrinsic, max_iter=2000, eps=1e-7, dissimilarity="precomputed", n_jobs=1, n_init=1)
        flat_local = mds.fit(D_fill_sub, init=guess[:, 0:dim_intrinsic,]).embedding_

        flat_local = flat_local.T

        embedding_dist = numpy.sqrt(calc_dist(flat_local))

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
        for j_point in knn_indexes_sub:
            ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
        ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]], intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')
        plt.axis('equal')

        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(flat_local[0, :], flat_local[1, :], c="k")
        for j_point in range(flat_local.shape[1]):
            ax.scatter(flat_local[0, j_point], flat_local[1, j_point], c='r')
        ax.scatter(flat_local[0, knn_indexes_sub[0]], flat_local[1, knn_indexes_sub[0]], c='g')
        plt.axis('equal')
        #dist_mat_trimmed = dist_mat_trimmed + dist_mat_trust_temp

        #dist_mat_trimmed_wgt = dist_mat_trimmed_wgt +1
        for i_row in range(knn_indexes_sub.shape[0]):
            for i_col in range(knn_indexes_sub.shape[0]):
                dist_mat_trimmed[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dist_mat_trimmed[i_row, i_col] + embedding_dist[i_row, i_col]
                dist_mat_trimmed_wgt[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dist_mat_trimmed_wgt[i_row, i_col] + 1


    dist_mat_trimmed = dist_mat_trimmed/numpy.maximum(dist_mat_trimmed_wgt, numpy.ones(dist_mat_trimmed_wgt.shape))
    return dist_mat_trimmed, dist_mat_trimmed_wgt

def intrinsic_isomaps(dist_mat_geo, dist_mat_short, dim_intrinsic, noisy_sensor_clusters_2):

    dist_mat_corrected, dist_mat_local_flat_wgt = trim_non_euc(dist_mat_geo, dist_mat_short,  dim_intrinsic, noisy_sensor_clusters_2)
    dist_mat_corrected, dist_mat_local_flat_wgt = trim_non_euc2(dist_mat_corrected,  dim_intrinsic, noisy_sensor_clusters_2)

    D_squared = dist_mat_corrected ** 2

    # centering matrix
    n = D_squared.shape[0]
    J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

    # perform double centering
    B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

    # find eigenvalues and eigenvectors
    U, eigen_val, V = numpy.linalg.svd(B)
    eigen_vect = V
    eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
    eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
    eigen_vect = eigen_vect[eigen_val_sort_ind]
    eigen_vect = eigen_vect[:dim_intrinsic].T

    mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)
    iso_embedding = mds.fit(dist_mat_geo, init=eigen_vect).embedding_
    #print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2,
    #              titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
    # stress, stress_normlized = embbeding_score(intrinsic_process_clusters, iso_embedding_local.T, titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
    mds = manifold.MDS(n_components=2, max_iter=2000, eps=1e-5, dissimilarity="precomputed", n_jobs=1, n_init=1)

    iso_embedding = mds.fit(dist_mat_short, weight=(dist_mat_short != 0).astype(int),
                                  init=iso_embedding).embedding_
    #print_process(iso_embedding_local.T, bounding_shape=None, color_map=color_map_clusters_2,
    #              titleStr="Isomap with Locally Learned Intrinsic Metric", align_points=intrinsic_process_clusters)
    #stress, stress_normlized = embbeding_score(intrinsic_process_clusters_2, iso_embedding_local.T,
    #                                           titleStr="Diffusion Maps with Locally Learned Intrinsic Metric")
    return iso_embedding

def trim_non_euc2(dist_mat_trust, dim_intrinsic, intrinsic_process_clusters):
    dist_mat_fill = scipy.sparse.csgraph.shortest_path(dist_mat_trust, directed=False)

    n_points = dist_mat_trust.shape[0]
    dist_mat_trimmed = numpy.zeros((n_points, n_points))
    dist_mat_trimmed_wgt = numpy.zeros((n_points, n_points))
    #indexs_balls = numpy.random.choice(n_points, size=n_points, replace=False)

    for i_point in range(1):
        dist_mat_trust_temp = numpy.array(dist_mat_trust, copy=True)

        D_fill = scipy.sparse.csgraph.shortest_path(dist_mat_trust_temp, directed=False)

        knn_indexes = numpy.argsort(dist_mat_fill[i_point], kind='quicksort')
        n_neighbors = 40
        flat = True
        check_list = []
        while flat:
            knn_indexes_sub = knn_indexes[0:n_neighbors]

            D_sub_trust_original = dist_mat_trust[knn_indexes_sub, :][:, knn_indexes_sub]


            numpy.fill_diagonal(D_fill, 0)

            D_fill_sub = D_fill[knn_indexes_sub, :][:, knn_indexes_sub]

            # square it
            D_squared = D_fill_sub ** 2

            # centering matrix
            n = D_squared.shape[0]
            J_c = 1. / n * (numpy.eye(n) - 1 + (n - 1) * numpy.eye(n))

            # perform double centering
            B = -0.5 * (J_c.dot(D_squared)).dot(J_c)

            # find eigenvalues and eigenvectors
            U, eigen_val, V = numpy.linalg.svd(B)
            eigen_vect = V
            eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
            eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
            eigen_vect = eigen_vect[eigen_val_sort_ind]
            eigen_vect = eigen_vect[:dim_intrinsic].T

            guess = numpy.real(numpy.dot(numpy.diag(numpy.sqrt(numpy.abs(eigen_val[:dim_intrinsic]))), eigen_vect.T).T)

            wgt = (D_sub_trust_original != 0).astype(int)

            mds = manifold.MDS(n_components=dim_intrinsic, max_iter=1000, eps=1e-9, dissimilarity="precomputed", n_jobs=1, n_init=1)
            #flat_local = mds.fit(D_fill_sub, init=guess).embedding_
            #stress1 = mds.stress_
            flat_local = mds.fit((D_sub_trust_original+D_sub_trust_original.T)/2, weight=wgt, init=guess).embedding_
            stress2 = mds.stress_

            flat_local = flat_local.T
            #fig = plt.figure()
            #ax = fig.gca()
            #ax.scatter(flat_local[0, :], flat_local[1, :], c="k")

            #expl = numpy.sum(eigen_val[:dim_intrinsic])
            #res = numpy.sum(eigen_val[dim_intrinsic:])
            dis = (D_sub_trust_original*wgt).sum()
            check = (stress2/dis)
            check_list.append(check)
            #flat = (check < 0.05)

            dis = numpy.sqrt(calc_dist(flat_local))
            for i_row in range(knn_indexes_sub.shape[0]):
                for i_col in range(knn_indexes_sub.shape[0]):
                    dist_mat_trust_temp[knn_indexes_sub[i_row], knn_indexes_sub[i_col]] = dis[i_row, i_col]

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(intrinsic_process_clusters[0, :], intrinsic_process_clusters[1, :], c="k")
            for j_point in knn_indexes_sub:
                ax.scatter(intrinsic_process_clusters[0, j_point], intrinsic_process_clusters[1, j_point], c='r')
            ax.scatter(intrinsic_process_clusters[0, knn_indexes_sub[0]],
                       intrinsic_process_clusters[1, knn_indexes_sub[0]], c='g')
            plt.axis('equal')

            fig = plt.figure()
            ax = fig.gca()
            ax.scatter(flat_local[0, 0], flat_local[1, 0], c="g")
            ax.scatter(flat_local[0, 1:], flat_local[1, 1:], c="r")
            plt.axis('equal')
            plt.show(block=False)

            if n_neighbors == n_points:
                break

            n_neighbors = min(numpy.ceil(n_neighbors*1.5), n_points)

        print(i_point)

        #dist_mat_trimmed = dist_mat_trimmed + dist_mat_trust_temp

        #dist_mat_trimmed_wgt = dist_mat_trimmed_wgt +1
        for i_row in knn_indexes_sub:
            for i_col in knn_indexes_sub:
                dist_mat_trimmed[i_row, i_col] = dist_mat_trimmed[i_row, i_col] + dist_mat_trust_temp[i_row, i_col]
                dist_mat_trimmed_wgt[i_row, i_col] = dist_mat_trimmed_wgt[i_row, i_col] + 1

    dist_mat_trimmed = dist_mat_trimmed/numpy.maximum(dist_mat_trimmed_wgt, numpy.ones(dist_mat_trimmed_wgt.shape))
    return dist_mat_trimmed, dist_mat_trimmed_wgt

def test_ml(Y, X, n_neighbors, n_components, color):

    X = X.T
    Y = Y.T

    fig = plt.figure(figsize=(15, 8))

    if Y.shape[1] == 2:
        try:

            # compatibility matplotlib < 1.0
            ax = fig.add_subplot(251)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

            # ax._axis3don = False
            ax.scatter(Y[:, 0], Y[:, 1], c=color)
            process = Y.T

            max_range = numpy.array([process[0].max() - process[0].min(), process[1].max() - process[1].min()]).max() / 2.0

            mid_x = (process[0].max() + process[0].min()) * 0.5
            mid_y = (process[1].max() + process[1].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)

            plt.title("Observed")
        except:
            ax = fig.add_subplot(251, projection='3d')
            plt.scatter(Y[:, 0], Y[:, 2], c=color)


    if Y.shape[1]==3:
        try:

            # compatibility matplotlib < 1.0
            ax = fig.add_subplot(251, projection='3d')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            #ax._axis3don = False
            ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=color)
            process = Y.T

            max_range = numpy.array([process[0].max() - process[0].min(), process[1].max() - process[1].min(), process[2].max() - process[2].min()]).max() / 2.0

            mid_x = (process[0].max() + process[0].min()) * 0.5
            mid_y = (process[1].max() + process[1].min()) * 0.5
            mid_z = (process[2].max() + process[2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

            ax.view_init(10, 60)
            plt.title("Observed")
        except:
            ax = fig.add_subplot(251, projection='3d')
            plt.scatter(Y[:, 0], Y[:, 2], c=color)

    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

    for i, method in enumerate(methods):
        Z = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                            eigen_solver='auto',
                                            method=method).fit_transform(Y)

        ax = fig.add_subplot(252 + i)
        plt.scatter(Z[:, 0], Z[:, 1], c=color)
        plt.title("%s" % (labels[i]))
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')

    ax = fig.add_subplot(256)

    plt.scatter(X[:, 0], X[:, 1], c=color)
    plt.title("Latent")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    Z = manifold.Isomap(n_neighbors, n_components).fit_transform(Y)
    ax = fig.add_subplot(257)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("Isomap")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    mds = manifold.MDS(n_components, max_iter=100, n_init=1)
    Z = mds.fit_transform(Y)
    ax = fig.add_subplot(258)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("MDS")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')

    se = manifold.SpectralEmbedding(n_components=n_components,
                                    n_neighbors=n_neighbors)
    Z = se.fit_transform(Y)
    ax = fig.add_subplot(259)
    plt.scatter(Z[:, 0], Z[:, 1], c=color)
    plt.title("Spectral Embedding")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())

    plt.axis('tight')

    plt.show()
