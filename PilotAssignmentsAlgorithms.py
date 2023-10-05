import matplotlib.axes
import numpy as np
from numpy import sum, zeros, ceil, tile, sqrt, array, min, argmin, copy, max, argsort
from numpy.random import random
from CellFreeNetwork import CellFreeNetwork
from SettingParams import *
from ClusteringAlgorithms import emil_heuristic_dcc_pilot_assignment
import matplotlib.pyplot as plt
import math
from collections import deque


def compute_wrap_around_locs(locs, coverage_area_len):
    wrap_around_locs = array([
        [0, 0],
        [coverage_area_len, 0],
        [0, coverage_area_len],
        [coverage_area_len, coverage_area_len],
        [coverage_area_len, -coverage_area_len],
        [-coverage_area_len, coverage_area_len],
        [-coverage_area_len, -coverage_area_len],
        [-coverage_area_len, 0],
        [0, -coverage_area_len]
    ]).T.reshape((2, 1, 9))
    locs_mat = tile(locs.reshape((locs.shape[0], locs.shape[1], 1)), (1, 1, 9))
    # return locs_mat
    return locs_mat + wrap_around_locs


def GB_KM(network: CellFreeNetwork,
          num_points: int = 100_000,
          epsilon: float = 1e-3):
    user_ap_distance = network.user_ap_distance
    num_pilots = network.pilot_len
    coverage_area_len = network.coverage_area_len
    num_users, num_aps = user_ap_distance.shape
    num_km_clusters = int(ceil(num_users / num_pilots))

    # initialize centroids and sample points
    sample_points = random((2, num_points)) * coverage_area_len - coverage_area_len / 2
    wrapped_sample_points = compute_wrap_around_locs(sample_points, coverage_area_len)
    centroids = random((2, num_km_clusters)) * coverage_area_len - coverage_area_len / 2
    # compute kmeans clusters centroids
    C_all = [set() for _ in range(num_km_clusters)]
    while True:
        # choose cluster for each sample point
        for i in range(num_points):
            wrapped_point = tile(wrapped_sample_points[:, i, :].reshape((2, 1, 9)), (1, num_km_clusters, 1))
            rep_centroids = tile(centroids.reshape((2, num_km_clusters, 1)), (1, 1, 9))
            chosen_cluster = argmin(min(sum((wrapped_point - rep_centroids) ** 2,  axis=0), axis=1))
            C_all[chosen_cluster].add(i)

        # update centroids
        old_centroids = copy(centroids)
        for i in range(num_km_clusters):
            members = list(C_all[i])
            centroids[:, i] = sum(sample_points[:, members], axis=1) / len(members)
        if max(sum((centroids - old_centroids) ** 2, axis=1)) < epsilon:
            break

    # assign pilots to UEs
    # compute distance between all UEs and centroids
    wrapped_users_loc = compute_wrap_around_locs(network.users_loc, coverage_area_len)
    d_ue_centroid = zeros((num_users, num_km_clusters))
    for i in range(num_km_clusters):
        rep_centroid = tile(centroids[:, i].reshape((2, 1, 1)), (1, num_users, 9))
        d_ue_centroid[:, i] = min(sum((rep_centroid - wrapped_users_loc) ** 2, axis=0), axis=1)

    pilot_groups = [list() for _ in range(num_km_clusters)]
    user_list = list(range(num_users))
    while user_list:
        for i in range(num_km_clusters):
            if len(pilot_groups[i]) == num_pilots:
                continue
            user_order = argsort(d_ue_centroid, axis=0)
            for j in range(num_users):
                if len(pilot_groups[i]) == num_pilots:
                    d_ue_centroid[:, i] = np.inf
                    break
                cand_user = user_order[j, i]
                if cand_user not in user_list:
                    continue
                if min(d_ue_centroid[cand_user]) == d_ue_centroid[cand_user, i]:
                    pilot_groups[i].append(cand_user)
                    user_list.remove(cand_user)
            d_ue_centroid[pilot_groups[i]] = np.inf
    pilot_alloc = zeros((num_users,))
    for group in pilot_groups:
        for i, user in enumerate(group):
            pilot_alloc[user] = i

    return pilot_alloc


def IB_KM(network: CellFreeNetwork,
          num_points: int = 100_000,
          epsilon: float = 1e-3):
    user_ap_distance = network.user_ap_distance
    num_pilots = network.pilot_len
    coverage_area_len = network.coverage_area_len
    aps_loc = network.aps_loc
    user_centric_clusters = network.clusters

    num_users, num_aps = user_ap_distance.shape
    num_km_clusters = int(ceil(num_users / num_pilots))
    sample_points = random((2, num_points)) * coverage_area_len - coverage_area_len / 2
    centroids = random((2, num_km_clusters)) * coverage_area_len - coverage_area_len / 2
    aps_wrapped_loc = compute_wrap_around_locs(aps_loc, coverage_area_len)

    d_all = np.zeros((num_points, num_aps))
    mu_all = np.zeros((num_km_clusters, num_aps))

    # compute distance between APs and centroids
    for i in range(num_km_clusters):
        mu_i = tile(centroids[:, i].reshape((2, 1, 1)), (1, num_aps, 9))
        mu_all[i] = sqrt(min(sum((aps_wrapped_loc - mu_i) ** 2, axis=0), axis=1))
    # compute distance between APs and sample points
    for i in range(num_points):
        point = tile(sample_points[:, i].reshape((2, 1, 1)), (1, num_aps, 9))
        d_all[i] = sqrt(min(sum((aps_wrapped_loc - point) ** 2, axis=0), axis=1))

    # compute kmeans clusters centroids
    C_all = [set() for _ in range(num_km_clusters)]
    while True:
        # choose cluster for each sample point
        for i in range(num_points):
            chosen_cluster = argmin(
                sum((tile(d_all[i, :].reshape((1, num_aps)), (num_km_clusters, 1)) - mu_all) ** 2, axis=1))
            C_all[chosen_cluster].add(i)

        # update centroids
        old_mu_all = copy(mu_all)
        for i in range(num_km_clusters):
            members = list(C_all[i])
            centroids[:, i] = sum(sample_points[:, members], axis=1) / len(members)
            mu_all[i, :] = sum(d_all[members, :], axis=0) / len(members)
        if max(sum((mu_all - old_mu_all) ** 2, axis=1)) < epsilon:
            break

    # assign the UEs to pilot clusters
    C_all = [set() for _ in range(num_km_clusters)]
    dx_clusters = user_centric_clusters * user_ap_distance
    dis = -np.ones((num_users, num_km_clusters))
    L_users = set(range(num_users))
    blacklist_users = [set() for _ in range(num_km_clusters)]
    while L_users:
        user = L_users.pop()
        d_user = tile(dx_clusters[user, :].reshape((1, -1)), (num_km_clusters, 1))
        dis[user, :] = sum((d_user - mu_all) ** 2, axis=1)
        user_prefs = argsort(dis[user])
        for pref in user_prefs:
            if user in blacklist_users[pref]:
                continue
            else:
                break
        if len(C_all[pref]) < num_pilots:
            C_all[pref].add(user)
        else:
            # competition mechanism
            members = C_all[pref].copy()
            members.add(user)
            members = list(members)
            members.sort()
            dis_members = dis[members, pref]
            sorted_members = [members[idx] for idx in argsort(dis_members)]
            C_all[pref] = set(sorted_members[:num_pilots])
            worst_user = sorted_members[-1]
            L_users.add(worst_user)
            blacklist_users[pref].add(worst_user)



    # plot_layer(network.users_loc[:, list(C_all[0])], 'r', 'o', 'Cluster 1')
    # plot_layer(network.users_loc[:, list(C_all[1])], 'b', 'o', 'Cluster 2')
    # plot_layer(aps_loc, 'g', 'd', 'APs')
    # plt.show()

    # assign orthogonal pilots to members of the first pilot cluster
    pilot_alloc = zeros((num_users,), dtype=int)
    cluster_lens = [len(C_all[i]) for i in range(num_km_clusters)]
    i_zero = cluster_lens.index(num_pilots)
    C_all[0], C_all[i_zero] = C_all[i_zero], C_all[0]
    for i, user in enumerate(C_all[0]):
        assert len(C_all[0]) == num_pilots
        pilot_alloc[user] = i

    # assign pilots to the remaining users
    for C_i in C_all[1:]:
        C_1_temp = C_all[0].copy()
        C_i_temp = C_i.copy()

        while C_i_temp:
            L_all = [set() for _ in range(num_users)]
            for user in C_1_temp:
                sorted_users = argsort(sum((tile(dx_clusters[user].reshape((1, -1)),
                                                 (num_users, 1)) - dx_clusters) ** 2, axis=1))
                best_user = [i for i in sorted_users if i in C_i_temp][-1]
                L_all[best_user].add(user)

            assigned_users = set()
            for user in C_i_temp:
                if not L_all[user]:
                    continue
                if len(L_all[user]) == 1:
                    sharing_user = list(L_all[user])[0]
                    pilot_alloc[user] = pilot_alloc[sharing_user]
                    C_1_temp.remove(sharing_user)
                else:
                    sorted_users = argsort(sum((tile(dx_clusters[user].reshape((1, -1)),
                                                     (num_users, 1)) - dx_clusters) ** 2, axis=1))
                    best_user = [k for k in sorted_users if k in L_all[user]][-1]
                    pilot_alloc[user] = pilot_alloc[best_user]
                    C_1_temp.remove(best_user)
                assigned_users.add(user)
            C_i_temp.difference_update(assigned_users)

    return pilot_alloc


def user_group(path_loss_shadowing: np.ndarray, pilot_len: int, clusters: np.ndarray):
    # local variables
    num_users, num_aps = path_loss_shadowing.shape
    delta_range = [0.01, 0.99]
    delta = 0.5
    betax_clusters = path_loss_shadowing*clusters
    mat_S = zeros((num_users, num_aps))
    num_serving_rels = len(np.where(clusters == 1)[0])
    sorted_betas_linear_indices = argsort(betax_clusters.reshape((-1, )))[::-1]
    sorted_betas_indices = [(index // num_aps, index % num_aps) for index in sorted_betas_linear_indices]
    betas = betax_clusters.reshape((-1,))[sorted_betas_linear_indices]

    # main loop
    trials = 0
    pilot_alloc = zeros((num_users,), dtype=int)
    while True:
        trials += 1
        user_list = list(range(num_users))
        curr_group = -1
        groups = list()
        threshold_beta = betas[math.ceil(delta * num_serving_rels)]
        # construct S
        for cand_user, j in sorted_betas_indices[0:math.ceil(delta * num_serving_rels)]:
            assert betax_clusters[cand_user, j] >= threshold_beta
            mat_S[cand_user, j] = 1
        # construct T
        mat_T = mat_S @ mat_S.T
        # construct G
        mat_G = -np.ones((num_users - 1, num_users - 1))
        for cand_user in range(num_users):
            k = 0
            for j in range(cand_user + 1, num_users):
                if mat_T[cand_user, j] == 0:
                    mat_G[cand_user, k] = j
                    k += 1

        R_all = [[int(k) for k in list(mat_G[user, :]) if k != -1] for user in range(num_users - 1)]

        # start allocation
        while user_list:
            user = user_list[0]
            user_list.remove(user)
            curr_group += 1
            groups.append(list())
            groups[curr_group].append(user)
            for R_k in R_all:
                if user in R_k:
                    R_k.remove(user)

            if user == num_users - 1:
                break
            else:
                while R_all[user]:
                    cand_user = R_all[user][0]
                    groups[curr_group].append(cand_user)
                    if cand_user != num_users - 1:
                        R_all[user] = [k for k in R_all[cand_user] if k in R_all[user]]
                    for R_k in R_all:
                        if cand_user in R_k:
                            R_k.remove(cand_user)
                    user_list.remove(cand_user)

        num_groups = curr_group + 1
        if num_groups == pilot_len or trials == 10:
            for cand_user in range(num_groups):
                for k in groups[cand_user]:
                    pilot_alloc[k] = cand_user
            break
        elif num_groups < pilot_len:
            delta_range[0] = delta
        else:
            delta_range[1] = delta
        delta = sum(delta_range) / 2

    return pilot_alloc



def plot_layer(points: np.ndarray, color, marker, label, ax: matplotlib.axes.Axes = None):
    if ax is None:
        ax = plt.subplot(111)
    ax.scatter(points[0, :], points[1, :], c=color, marker=marker, label=label)
    return ax


params = {'num_aps': 30,
          'num_users': 25,
          'num_antennas': 4,
          'ap_dist': 'Uniform',
          'users_dist': 'Uniform',
          'coverage_area_len': 707,  # ~0.5 km ** 2 area
          'channel_model': 'Correlated Rayleigh',
          'block_len': 200,
          'pilot_len': 5,
          'pilot_alloc_alg': 'random',
          'pilot_power_control_alg': 'max',
          'uplink_power_control_alg': 'max',
          'downlink_power_alloc_alg': 'fractional',
          'user_max_power': 100,
          'ap_max_power': 200,
          'uplink_noise_power': -94,
          'downlink_noise_power': -94,
          'clustering_alg': 'emil'
          }

network = CellFreeNetwork(**params)
network.generate_snapshot()
user_group(network.channel_model.path_loss_shadowing, network.pilot_len, network.clusters)
# pilots_km = GB_KM(network)
# _, pilots_emil = emil_heuristic_dcc_pilot_assignment(network.channel_model.path_loss_shadowing, network.pilot_len)

# fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
# colors = ['r', 'b', 'g', 'm', 'k', 'c', 'tab:brown']
# for i in range(len(set(pilots_km))):
#     plot_layer(network.users_loc[:, pilots_km == i], colors[i], 'o', f"pilot {i}", axes[0])
#
# for i in range(len(set(pilots_emil))):
#     plot_layer(network.users_loc[:, pilots_emil == i], colors[i], 'o', f"pilot {i}", axes[1])
# plt.show()
test = []
