from CellFreeNetwork import CellFreeNetwork
import matplotlib.pyplot as plt
from scipy.io import savemat
import json
import numpy as np
from SettingParams import *

params = mock_params
params['clustering_alg'] = "emil"
# num_aps = 9
# num_users = 4
# coverage_area_len = 100
# generate uniform aps_loc
# aps_per_row = int(np.sqrt(num_aps))
# aps_loc = np.zeros((2, num_aps))
# step = coverage_area_len/(aps_per_row + 1)
# for ap in range(num_aps):
#     col = np.mod(ap, aps_per_row) + 1
#     row = np.floor(ap/aps_per_row) + 1
#     aps_loc[0, ap] = row * step
#     aps_loc[1, ap] = col * step
# aps_loc -= coverage_area_len/2
# users_loc = np.zeros((2, num_users))
# users_loc[:,0] = np.array([20,20])
# users_loc[:,1] = np.array([20,-20])
# users_loc[:,2] = np.array([-20,20])
# users_loc[:,3] = np.array([-20,-20])


network = CellFreeNetwork(**params)
network.generate_snapshot()
network.plot_network()

num_users = network.num_users
num_aps = network.num_aps
num_antennas = network.num_antennas
num_trials = 10
num_frames = 50
combining_algs = ["MMSE",]
num_algs = len(combining_algs)
alpha_dl = np.zeros((num_algs, num_trials, num_users))
mu_dl = np.zeros((num_algs, num_trials, num_users))
alpha_ul = np.zeros((num_algs, num_trials, num_users))
mu_ul = np.zeros((num_algs, num_trials, num_users))


for j in range(num_trials):
    collective_channels, collective_channel_estimates, _, _ = \
        network.generate_channel_realizations(num_frames)
    network.downlink_power_vec = network.downlink_power_alloc_fractional_centralized_conservative()
    Q_dl, Q_ul, T_ul, T_dl, C_v = network.compute_clustering_optimization_model_mrc()
    for i, combining_alg in enumerate(combining_algs):
        precoders = network.simulate_downlink_centralized(combining_alg, collective_channels,
                                                          collective_channel_estimates, conservative=True)
        combiners = network.simulate_uplink_centralized(combining_alg, collective_channels, collective_channel_estimates)
        alpha_dl[i, j], mu_dl[i, j] = network.estimate_alpha_mu_dl(collective_channels, precoders, Q_dl, T_dl)
        alpha_ul[i, j], mu_ul[i, j] = network.estimate_alpha_mu_ul(collective_channels, combiners, Q_ul, T_ul, C_v)

B = np.zeros((num_users, num_aps*num_antennas, num_aps*num_antennas), dtype=complex)
Binv = np.zeros((num_users, num_aps*num_antennas, num_aps*num_antennas), dtype=complex)
Bnorm = np.zeros((num_users,))
eyeLN = np.eye(num_aps*num_antennas, dtype=complex)
D_all = network.clusters_block_diag
p = network.uplink_power_vec
rho = network.downlink_power_vec

for user in range(num_users):
    D_k = D_all[user]
    for other_user in range(num_users):
        R_i = network.channel_model.get_block_diag_corr(other_user)
        B[user] += p[other_user] * (D_k @ R_i @ D_k)
    B[user] += eyeLN
    Binv[user] = np.linalg.pinv(B[user])
    Bnorm[user] = rho[user] * np.linalg.norm(Binv[user])

test = []
