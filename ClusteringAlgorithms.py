import numpy as np
from numpy import sum, inf, zeros, argmax, where, ones


def emil_heuristic_dcc_pilot_assignment(beta_vector: np.ndarray, num_pilots: int):
    num_users, num_aps = beta_vector.shape
    beta_vector = 10 ** (beta_vector/10)
    pilot_alloc = -ones((num_users,), dtype=int)
    clusters = zeros((num_users, num_aps), dtype=int)
    # allocate the orthogonal pilots to the first UEs
    for i in range(num_pilots):
        pilot_alloc[i] = i
    # allocate pilots with least interference to other UEs according to the best ap criterion
    for user in range(num_pilots, num_users):
        best_ap = argmax(beta_vector[user, :])
        best_pilot = -1
        interference = inf
        for pilot in range(num_pilots):
            temp = sum([beta_vector[user, best_ap] for user in range(num_users) if pilot_alloc[user] == pilot])
            if temp < interference:
                interference = temp
                best_pilot = pilot
        pilot_alloc[user] = best_pilot
    # assign the clusters
    for ap in range(num_aps):
        for pilot in range(num_pilots):
            candidate_users = where(pilot_alloc == pilot)[0]
            best_user = candidate_users[argmax(beta_vector[candidate_users, ap])]
            clusters[best_user, ap] = 1
    return clusters, pilot_alloc


def massive_access_clustering(beta_vector: np.ndarray, num_pilots: int):
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
                        users_sorted = np.argsort(beta_vector[:, ap])
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
                        users_sorted = np.argsort(beta_vector[:, best_ap])
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

# betas = np.array([[0.5, 0.25, 0.3, 0.7, 0.8],
#                   [0.8, 0.11, 0.2, 0.1, 0.9],
#                   [0.6, 0.7, 0.8, 0.2, 0.1],
#                   [0.88, 0.45, 0.3, 0.5, 0.2]])
# numPilots = 2
# clusters = massive_access_clustering(betas, numPilots)
# other_clusters, _ = emil_heuristic_dcc_pilot_assignment(betas, numPilots)
# test = []
