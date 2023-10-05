import numpy as np


# def compute_greedy_index(betas: np.ndarray, clusters: np.ndarray):
#     betas = 10 ** (betas / 10)
#     max_beta_indices = np.argmax(betas, axis=0)
#     indices = [[user, ap] for (ap, user) in enumerate(max_beta_indices)]
#     return np.sum(clusters[range(num_users), max_beta_indices] * betas[range(num_users), max_beta_indices]) / np.sum(betas[:, max_beta_indices])
#

def compute_greedy_index(betas: np.ndarray, clusters: np.ndarray, num_pilots: int):
    num_users, num_aps = betas.shape
    betas = 10 ** (betas / 10)
    sorted_betas_indices = np.argsort(betas, axis=0)
    mask = np.zeros((num_users, num_aps))
    for ap in range(num_aps):
        tau_p_largest_users = sorted_betas_indices[-num_pilots:, ap]
        mask[tau_p_largest_users, ap] = 1
    betas_x_clusters = betas * clusters
    tau_p_largest_betas = betas * mask
    return np.sum(mask * clusters) / np.sum(mask)
    # return np.mean(np.sum(betas_x_clusters, axis=0) / np.sum(tau_p_largest_betas,0))


def compute_cluster_sizes(clusters: np.ndarray):
    return np.sum(clusters, axis=1)


def compute_abandoned_users(clusters: np.ndarray):
    return np.where(np.sum(clusters, axis=1) == 0)[0]


def compute_one_users(clusters: np.ndarray):
    return np.where(np.sum(clusters, axis=1) == 0)[0]


def jain_fairness(SE: np.ndarray):
    num_algs, num_layouts, num_users = SE.shape
    jain_values = np.zeros((num_algs,))
    for alg in range(num_algs):
        jain_values[alg] = np.mean(np.sum(SE[alg], axis=1) ** 2 / (num_users * np.sum(SE[alg] ** 2, axis=1)))
    return jain_values


# num_users = 12
# num_aps = 30
# betas = np.random.random((num_users, num_aps))
# clusters = np.round(np.random.random((num_users, num_aps)))
#
# test = compute_greedy_index(betas, clusters, 4)
