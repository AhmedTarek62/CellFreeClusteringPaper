import numpy as np
from numpy import sum, inf, zeros, argmax, where, ones, argsort
def utility_function_one(beta_vector: np.ndarray, user_clusters: list, ap):
    num_users, num_aps = beta_vector.shape
    utilities = zeros((num_users,))
    sum_betas_ap = sum(beta_vector[:, ap])
    w_1, w_2, w_3 = 1, -0.5, -0.5
    user_clusters_arr = zeros((num_users, num_aps))
    for user in range(num_users):
        for ap in user_clusters[user]:
            user_clusters_arr[user, ap] = 1

    for user in range(num_users):
        user_channel_quality = beta_vector[user, ap]/sum_betas_ap
        user_cluster_quality = sum(user_clusters_arr[user] * beta_vector[user])/sum(beta_vector[user])
        other_users_cluster_quality = 0
        for other_user in range(num_users):
            if user != other_user:
                other_users_cluster_quality += sum(user_clusters_arr[other_user] * beta_vector[other_user])\
                                               /sum(beta_vector[other_user])
        utilities[user] = w_1 * user_channel_quality + w_2 * user_cluster_quality + w_3 * other_users_cluster_quality
    return utilities


def general_chen(beta_vector: np.ndarray, num_pilots: int, utility_function=utility_function_one):
    num_users, num_aps = beta_vector.shape

    # initialization
    all_aps = set(range(num_aps))
    user_clusters = [set() for _ in range(num_users)]
    aps_clusters = [set() for _ in range(num_aps)]
    blacklists = [set() for _ in range(num_users)]
    abandoned_users = set()

    for user in range(num_users):
        while True:
            candidate_aps = all_aps.difference(user_clusters[user].union(blacklists[user]))
            if not candidate_aps:
                break
            else:
                if user in abandoned_users:
                    user_clusters[user] = user_clusters[user].union(candidate_aps)
                    ap = list(candidate_aps)[0]
                    aps_clusters[ap].add(user)
                    if len(aps_clusters[ap]) == num_pilots + 1:
                        users_sorted = argsort(utility_function(10 ** (beta_vector/10), user_clusters, ap))
                        candidate_users = aps_clusters[ap].difference(abandoned_users)
                        worst_user = [k for k in users_sorted if k in candidate_users][0]
                        user_clusters[worst_user].remove(ap)
                        aps_clusters[ap].remove(worst_user)
                    break
                else:
                    aps_sorted = np.argsort(beta_vector[user,:])
                    best_ap = [l for l in aps_sorted if l in candidate_aps][-1]
                    user_clusters[user].add(best_ap)
                    aps_clusters[best_ap].add(user)
                    if len(aps_clusters[best_ap]) == num_pilots + 1:
                        users_sorted = argsort(utility_function(10 ** (beta_vector/10), user_clusters, best_ap))
                        candidate_users = aps_clusters[best_ap].difference(abandoned_users)
                        worst_user = [k for k in users_sorted if k in candidate_users][0]
                        blacklists[worst_user].add(best_ap)
                        if len(blacklists[worst_user]) == num_aps - 1:
                            abandoned_users.add(worst_user)
                        user_clusters[worst_user].remove(best_ap)
                        aps_clusters[best_ap].remove(worst_user)

    clusters = np.zeros((num_users, num_aps))
    for user in range(num_users):
        for ap in range(num_aps):
            if ap in user_clusters[user]:
                clusters[user, ap] = 1
    return clusters