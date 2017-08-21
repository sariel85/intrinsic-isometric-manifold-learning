import numpy
import matplotlib.pyplot as plt
import scipy
from sklearn import manifold
from data_generation import print_process


def intrinsic_isometric_mapping(approx_intrinsic_geo_dists, approx_intrinsic_euc_dists, approx_intrinsic_euc_dists_trimmed, true_intrinsic_euc_dists, intrinsic_points, dim_intrinsic, n_mds_iters, mds_stop_threshold, n_clusters, size_patch_start, size_patch_step):
    plot_flag = False
    dim_clustering_embedding = 30

    n_points = approx_intrinsic_geo_dists.shape[0]

    # Embed so that geodesic distances are respected and find cluster points
    dist_mat_squared = approx_intrinsic_geo_dists ** 2
    J_c = 1. / n_points * (numpy.eye(n_points) - 1 + (n_points - 1) * numpy.eye(n_points))
    B = -0.5 * (J_c.dot(dist_mat_squared)).dot(J_c)
    eigen_val, U = scipy.sparse.linalg.eigs(B, k=dim_clustering_embedding, ncv=None, tol=0, which='LM', v0=None)
    eigen_val = numpy.real(eigen_val)
    U = numpy.real(U)
    eigen_vect = U
    eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
    eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
    eigen_vect = eigen_vect[:, eigen_val_sort_ind]
    eigen_vect = numpy.dot(eigen_vect, numpy.diag(numpy.sqrt(eigen_val[:dim_clustering_embedding])))
    (mu, clusters) = find_centers(eigen_vect, n_clusters)

    if plot_flag:
        # Plot clusters
        fig = plt.figure()
        ax = fig.gca()
        ax.scatter(intrinsic_points[0, :], intrinsic_points[1, :], c="k")
        color_list = numpy.zeros((3, mu.shape[0]))
        for i_cluster in range(mu.shape[0]):
            color_list[:, i_cluster] = numpy.random.rand(3, 1).T
            ax.scatter(intrinsic_points[0, clusters[i_cluster]], intrinsic_points[1, clusters[i_cluster]], color=color_list[:, i_cluster])
            ax.scatter(intrinsic_points[0, mu[i_cluster]], intrinsic_points[1, mu[i_cluster]], c='r', s=80)
        #plt.title('Clustering based on approximated geodesic distances')
        plt.axis('equal')
        plt.show(block=False)

    size_patch = size_patch_start

    dist_mat_intrinsic = numpy.zeros((n_clusters, n_points, n_points))

    for i_cluster in range(mu.shape[0]):
        dist_mat_intrinsic[i_cluster] = numpy.copy(approx_intrinsic_geo_dists, order='C')

    dist_mat_intrinsic_geo = numpy.zeros((n_clusters, n_points, n_points))

    dist_point_from_mu_sorted_arg = numpy.argsort(approx_intrinsic_euc_dists[mu, :], axis=1)

    while size_patch <= n_points:
        #n_neighbors_start = clusters_ind.__len__()
        #rank_check_list = []
        #n_neighbors_list = []

        for i_cluster in range(mu.shape[0]):
            dist_mat_intrinsic_geo[i_cluster] = scipy.sparse.csgraph.shortest_path(dist_mat_intrinsic[i_cluster], directed=False)
            clusters_indexes = numpy.ndarray.tolist(dist_point_from_mu_sorted_arg[i_cluster, :size_patch])
            dist_mat_sub_geo = dist_mat_intrinsic_geo[i_cluster][clusters_indexes, :][:, clusters_indexes]
            dist_mat_sub = approx_intrinsic_geo_dists[clusters_indexes, :][:, clusters_indexes]
            dist_mat_trimmed_sub = approx_intrinsic_euc_dists_trimmed[clusters_indexes, :][:, clusters_indexes]
            dist_mat_true_sub = true_intrinsic_euc_dists[clusters_indexes, :][:, clusters_indexes]
            dist_mat_sub_squared = dist_mat_sub_geo ** 2
            J_c = 1. / size_patch * (numpy.eye(size_patch) - 1 + (size_patch - 1) * numpy.eye(size_patch))
            B = -0.5 * (J_c.dot(dist_mat_sub_squared)).dot(J_c)
            eigen_val, U = scipy.sparse.linalg.eigs(B, k=dim_intrinsic, ncv=None, tol=0, which='LM', v0=None)
            eigen_val = numpy.real(eigen_val)
            U = numpy.real(U)
            eigen_vect = U
            eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
            eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
            eigen_vect = eigen_vect[:, eigen_val_sort_ind]
            eigen_vect = numpy.dot(eigen_vect, numpy.diag(numpy.sqrt(eigen_val[:dim_intrinsic]))).T
            mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iters, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
            iso_embedding = mds.fit(dist_mat_sub, dist_2_true=dist_mat_true_sub, init=eigen_vect.T).embedding_.T
            mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iters, eps=mds_stop_threshold, dissimilarity="precomputed", n_jobs=1, n_init=1)
            weight_matrix = numpy.zeros(dist_mat_trimmed_sub.shape)
            weight_matrix[numpy.where(dist_mat_trimmed_sub != 0)] = 1
            weight_matrix = weight_matrix
            iso_embedding_corrected = mds.fit(dist_mat_trimmed_sub, dist_2_true=dist_mat_true_sub, init=iso_embedding.T, weight=weight_matrix).embedding_.T
            #expl = numpy.sum(eigen_val[:dim_intrinsic])
            #res = numpy.sum(eigen_val[dim_intrinsic:])
            # dis = (D_sub_trust_original*wgt).sum()
            # check = (stress2/dis)
            #rank_check_list.append(dim_intrinsic * eigen_val[:dim_intrinsic] / expl)
            #n_neighbors_list.append(n_neighbors)

            if plot_flag:
                fig = plt.figure()
                ax = fig.gca()
                ax.scatter(intrinsic_points[0, :], intrinsic_points[1, :], c="silver")
                ax.scatter(intrinsic_points[0, clusters_indexes], intrinsic_points[1, clusters_indexes], c=color_list[:, i_cluster])
                ax.scatter(intrinsic_points[0, mu[i_cluster]], intrinsic_points[1, mu[i_cluster]], c='r', s=80)
                plt.axis('equal')
                #plt.title('Patch in intrinsic space')

                print_process(intrinsic_points[:, clusters_indexes], bounding_shape=None, color_map=color_list[:, i_cluster], titleStr="Embedding - Intrinsic Isomap")
                plt.axis('equal')
                plt.title('Patch in intrinsic space')

                print_process(eigen_vect, bounding_shape=None, color_map=color_list[:, i_cluster], titleStr="Embedding - Intrinsic Isomap", align_points=intrinsic_points[:, clusters_indexes])
                plt.axis('equal')
                plt.title('Embedding - Intrinsic Isomap - Linear')

                print_process(iso_embedding, bounding_shape=None, color_map=color_list[:, i_cluster], titleStr="Embedding - Intrinsic Isomap", align_points=intrinsic_points[:, clusters_indexes])
                plt.axis('equal')
                plt.title('Embedding - Intrinsic Isomap')

                print_process(iso_embedding_corrected, bounding_shape=None, color_map=color_list[:, i_cluster], titleStr="Embedding - Intrinsic - Isometric", align_points=intrinsic_points[:, clusters_indexes])
                plt.axis('equal')
                plt.title('Embedding - Intrinsic - Isometric')

                fig = plt.figure()
                ax = fig.gca()
                ax.plot(mds.stress_norm_log, c="green", label='Observed Stress')
                ax.plot(mds.stress_norm_real_log, c="red", label='Real Stress')
                ax.set_xlabel('SMACOF iterations')
                ax.set_ylabel('Stress')
                plt.legend()
                #plt.figure()
                #plt.plot(numpy.asarray(n_neighbors_list), numpy.asarray(rank_check_list))
                plt.show(block=False)

            dist_new_sub = scipy.spatial.distance.cdist(iso_embedding_corrected.T, iso_embedding_corrected.T, metric='euclidean', p=2, V=None, VI=None, w=None)
            X, Y = numpy.meshgrid(clusters_indexes, clusters_indexes)
            X = numpy.ndarray.flatten(X)
            Y = numpy.ndarray.flatten(Y)
            X_1, Y_1 = numpy.meshgrid(range(clusters_indexes.__len__()), range(clusters_indexes.__len__()))
            X_1 = numpy.ndarray.flatten(X_1)
            Y_1 = numpy.ndarray.flatten(Y_1)
            dist_mat_intrinsic[i_cluster][X, Y] = dist_new_sub[X_1, Y_1]

        if size_patch == n_points:
            break
        else:
            size_patch = numpy.min([size_patch + size_patch_step, n_points])


    dist_mat_intrinsic = scipy.sparse.csgraph.shortest_path(numpy.mean(dist_mat_intrinsic, axis=0), directed=False)
    dist_mat_intrinsic_squared = dist_mat_intrinsic ** 2

    n_neighbors = dist_mat_intrinsic_squared.shape[0]

    # centering matrix
    J_c = 1. / n_neighbors * (numpy.eye(n_neighbors) - 1 + (n_neighbors - 1) * numpy.eye(n_neighbors))

    # perform double centering
    B = -0.5 * (J_c.dot(dist_mat_intrinsic_squared)).dot(J_c)

    # find eigenvalues and eigenvectors
    eigen_val, U = scipy.sparse.linalg.eigs(B, k=dim_intrinsic, ncv=None, tol=0, which='LM', v0=None)
    eigen_val = numpy.real(eigen_val)
    U = numpy.real(U)
    eigen_vect = U
    eigen_val_sort_ind = numpy.argsort(-numpy.abs(eigen_val))
    eigen_val = numpy.abs(eigen_val[eigen_val_sort_ind])
    eigen_vect = eigen_vect[:, eigen_val_sort_ind]
    eigen_vect = numpy.dot(eigen_vect, numpy.diag(numpy.sqrt(eigen_val[:dim_intrinsic]))).T
    mds = manifold.MDS(n_components=dim_intrinsic, max_iter=numpy.int(n_mds_iters), eps=mds_stop_threshold,
                       dissimilarity="precomputed", n_jobs=1, n_init=1)
    iso_embedding = mds.fit(dist_mat_intrinsic, dist_2_true=true_intrinsic_euc_dists, init=eigen_vect.T).embedding_.T
    mds = manifold.MDS(n_components=dim_intrinsic, max_iter=n_mds_iters, eps=mds_stop_threshold,
                       dissimilarity="precomputed", n_jobs=1, n_init=1)
    weight_matrix = numpy.zeros(approx_intrinsic_euc_dists_trimmed.shape)
    weight_matrix[numpy.where(approx_intrinsic_euc_dists_trimmed != 0)] = 1
    iso_embedding_corrected = mds.fit(approx_intrinsic_euc_dists_trimmed, dist_2_true=true_intrinsic_euc_dists, init=iso_embedding.T,
                                      weight=weight_matrix).embedding_.T

    #expl = numpy.sum(eigen_val[:dim_intrinsic])

    #rank_check_list.append(dim_intrinsic * eigen_val[dim_intrinsic] / expl)

    #n_neighbors_list.append(n_neighbors)

    if plot_flag:
        print_process(eigen_vect, bounding_shape=None, color_map=color_list[:, i_cluster],
                      titleStr="Embedding - Intrinsic Isomap", align_points=intrinsic_points)
        plt.axis('equal')
        plt.title('Embedding - Intrinsic Isomap')

        print_process(iso_embedding, bounding_shape=None, color_map=color_list[:, i_cluster],
                      titleStr="Embedding - Intrinsic Isomap", align_points=intrinsic_points)
        plt.axis('equal')
        plt.title('Embedding - Intrinsic Isomap')

        print_process(iso_embedding_corrected, bounding_shape=None, color_map=color_list[:, i_cluster],
                      titleStr="Embedding - Intrinsic - Isometric", align_points=intrinsic_points)
        plt.axis('equal')
        plt.title('Embedding - Intrinsic - Isometric')
        plt.show(block=False)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(mds.stress_norm_log, c="green", label='Observed Stress')
        ax.plot(mds.stress_norm_real_log, c="red", label='Real Stress')
        ax.set_xlabel('SMACOF iterations')
        ax.set_ylabel('Stress')
        plt.legend()
        plt.show(block=False)


    return iso_embedding_corrected


def cluster_points(X, mu_ind):
    cluster_sizes = numpy.zeros(mu_ind.shape)
    clusters = {}
    dist = scipy.spatial.distance.cdist(X, X[mu_ind, :], metric='euclidean', p=2, V=None, VI=None, w=None)
    knn_indexes = numpy.argsort(dist, kind='quicksort', axis=1)

    for i_x in range(X.shape[0]):
        bestmukey = knn_indexes[i_x, 0]

        try:
            clusters[bestmukey].append(i_x)
        except KeyError:
            clusters[bestmukey] = [i_x]

        cluster_sizes[bestmukey] = cluster_sizes[bestmukey] + 1

    return clusters, cluster_sizes


def reevaluate_centers(mu_ind, clusters_ind, X):
    center = numpy.empty((mu_ind.shape[0], X.shape[1]))
    center[:] = numpy.NAN
    for i_center in range(mu_ind.size):
        center[i_center, :] = numpy.mean(X[clusters_ind[i_center], :], axis=0)

    dist = scipy.spatial.distance.cdist(center, X, metric='euclidean', p=2, V=None, VI=None, w=None)
    knn_indexes = numpy.argsort(dist, kind='quicksort')

    return knn_indexes[:, 0]


def has_converged(mu_ind, oldmu_ind, X):
    return (set(mu_ind) == set(oldmu_ind))


def find_centers(X, k_Final):

    k = k_Final*4
    # Initialize to K random centers

    mu_ind = numpy.random.choice(X.shape[0], size=k, replace=False)

    while mu_ind.shape[0] > k_Final:

        oldmu_ind = numpy.random.choice(X.shape[0], size=mu_ind.shape[0], replace=False)

        while not has_converged(mu_ind, oldmu_ind, X):
            oldmu_ind = mu_ind
            # Assign all points in X to clusters
            clusters, clusters_sizes = cluster_points(X, mu_ind)
            # Reevaluate centers
            mu_ind = reevaluate_centers(mu_ind, clusters, X)

        clusters, clusters_sizes = cluster_points(X, mu_ind)
        cluster_to_remove = numpy.argmin(clusters_sizes)
        mu_ind = mu_ind[numpy.where(numpy.arange(mu_ind.shape[0]) != cluster_to_remove)]

        clusters, clusters_sizes = cluster_points(X, mu_ind)
        mu_ind = reevaluate_centers(mu_ind, clusters, X)


    oldmu_ind = numpy.random.choice(X.shape[0], size=mu_ind.shape[0], replace=False)

    #while not has_converged(mu_ind, oldmu_ind, X):
    #    oldmu_ind = mu_ind
    #    # Assign all points in X to clusters
    #    clusters, clusters_sizes = cluster_points(X, mu_ind)
    #    # Reevaluate centers
    #    mu_ind = reevaluate_centers(mu_ind, clusters, X)

    '''
    X = X.T
    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(X[0, :], X[1, :], c="k")
    for i_point in range(mu_ind.shape[0]):
        ax.scatter(X[0, clusters[i_point]], X[1, clusters[i_point]],
                   color=numpy.random.rand(3, 1))
        ax.scatter(X[0, mu_ind[i_point]], X[1, mu_ind[i_point]], c='r')
    plt.axis('equal')
    plt.show(block=False)
    X = X.T
    '''

    return mu_ind, clusters


def init_board(N):
    X = numpy.array([(numpy.random.uniform(-1, 1), numpy.uniform(-1, 1)) for i in range(N)])
    return X