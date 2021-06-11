import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist


def run_kmeans_clustering(data, max_clusters):
    """
    This function will run kmeans clustering up to the max_profile number, and select an optimal cluster number.
    The function removes any observations assigned to independant cluster's; considering them outliers
    inhibitive to the intention of the model.

    It will then return a kmeans model fitted to the training data.

    Args:
        - data: df, training data
    Returns:
        - clusterer: fitted KMeans model
    """
    # perform an initial check to ensure we have more obs than possible clusters
    n_rows = len(data)
    if n_rows < max_clusters:
        max_clusters = n_rows
    else:
        max_clusters = max_clusters

    # create containers for pertinent information
    cluster_number = []
    distortions = []
    rocs = []
    cluster_rocs = []
    rate_of_change = 0

    # iterate through the number of clusters
    n_clusters = 0

    for i in range(max_clusters):
        # set min_obs_threshold, min observations to retain a cluster
        min_obs_threshold = len(data) * 0.05
        _pass = False

        # increment the number of clusters and fit
        n_clusters += 1
        clusterer = KMeans(n_clusters=n_clusters)
        clusters = clusterer.fit_predict(data)

        # check if our clusters have <= min_obs_threshold observation, if so this is inviable
        assignment_list = clusters.tolist()
        cluster_instances = [0] * n_clusters
        for x in assignment_list:
            cluster_instances[x] += 1

        for ind in range(n_clusters):
            # if the cluster has < min_obs_threshold, discard the cluster and the observations assigned to it
            if cluster_instances[ind] <= max(min_obs_threshold, 1):
                data = data[[member != ind for member in assignment_list]]
                assignment_list = (filter((ind).__ne__, assignment_list))
                _pass = True
        # if _pass has been switched to true, we must re-iterate with the same cluster number,
        # albeit with discarded observations
        if _pass:
            n_clusters -= 1
            continue

        # calculate cluster centers, and distortion
        centers = []
        for cluster in set(clusters):
            center = np.mean(data[clusters == cluster])
            centers.append(center)
        # calculate the distortion
        distortion_new = np.sum(
            np.min(cdist(data, centers, 'euclidean'), axis=1) / data.shape[0]) / n_clusters

        # store information to ID optimal cluster
        if n_clusters >= 2:
            rate_of_change = distortion_new - distortion
            clust_rate_of_change = n_clusters / (n_clusters - 1)
            distortion = distortion_new
        if n_clusters == 1:
            distortion = distortion_new
            clust_rate_of_change = 0

        # append to the containers
        cluster_number.append(n_clusters)
        distortions.append(distortion)
        rocs.append(rate_of_change)
        cluster_rocs.append(clust_rate_of_change)

    # calculate the strength of information gain
    strengths = np.divide(rocs, cluster_rocs)
    strengths = np.nan_to_num(strengths, 0)

    # keep either then optimal cluster based on strength, or max_clusters
    if min(strengths) < 0:
        optimal_clusters = min([index for index, strength in enumerate(strengths) if strength < 0]) + 1
    else:
        optimal_clusters = max_clusters

    # fit a final model on the optimal_cluster and set it as the number of profiles
    clusterer = KMeans(n_clusters=optimal_clusters)
    clusterer.fit(data)
    n_clusters = optimal_clusters

    return clusterer, n_clusters