from CellFreeNetwork import CellFreeNetwork
import matplotlib.pyplot as plt
import numpy as np
from SettingParams import *

setting = 'mock'
if setting == 'mock':
    params = mock_params
else:
    params = final_params

params['clustering_alg'] = "emil"
num_layouts = 10
num_frames = 50
num_users = params['num_users']
num_aps = params['num_aps']
num_antennas = params['num_antennas']
combining_algs = ["MMSE", "MRC", "PMMSE","PRZF"]
num_algs = len(combining_algs)
network = CellFreeNetwork(**params)
alpha_dl = np.zeros((num_algs, num_layouts, num_users))
mu_dl = np.zeros((num_algs, num_layouts, num_users))
alpha_ul = np.zeros((num_algs, num_layouts, num_users))
mu_ul = np.zeros((num_algs, num_layouts, num_users))

average_SE_ul = np.zeros((num_algs, num_layouts * num_users))
average_SE_dl = np.zeros((num_algs, num_layouts * num_users))

average_SE_ul_heuristic = np.zeros((num_algs, num_layouts * num_users))
average_SE_dl_heuristic = np.zeros((num_algs, num_layouts * num_users))

for layout in range(num_layouts):
    print(f"Layout: {layout + 1} out of {num_layouts}")
    network.generate_snapshot()
    collective_channels, collective_channel_estimates, _, _ = \
        network.generate_channel_realizations(num_frames)
    network.downlink_power_vec = network.downlink_power_alloc_fractional_centralized_conservative()
    Q_dl, Q_ul, T_ul, T_dl, C_v = network.compute_clustering_optimization_model_mrc()
    for i, combining_alg in enumerate(combining_algs):
        precoders = network.simulate_downlink_centralized(combining_alg, collective_channels,
                                                          collective_channel_estimates, conservative=True)
        combiners = network.simulate_uplink_centralized(combining_alg, collective_channels, collective_channel_estimates)
        alpha_dl[i, layout] , mu_dl[i, layout] = \
            network.estimate_alpha_mu_dl(collective_channels, precoders, Q_dl, T_dl)
        alpha_ul[i, layout] , mu_ul[i, layout] = \
            network.estimate_alpha_mu_ul(collective_channels, combiners, Q_ul, T_ul, C_v)

    collective_channels, collective_channel_estimates, channels, channel_estimates = \
        network.generate_channel_realizations(num_frames)

    for i, combining_alg in enumerate(combining_algs):
        precoders = network.simulate_downlink_centralized(combining_alg, collective_channels,
                                                          collective_channel_estimates, conservative=True)
        average_SE_dl[i, layout * num_users: (layout + 1) * num_users] = \
            network.compute_downlink_SE_centralized(collective_channels, precoders)
        average_SE_dl_heuristic[i, layout * num_users: (layout + 1) * num_users] = \
            network.compute_alpha_mu_SE_dl(alpha_dl[i, layout], mu_dl[i, layout], Q_dl, T_dl)

        combiners = network.simulate_uplink_centralized(combining_alg, collective_channels, collective_channel_estimates)
        average_SE_ul[i, layout * num_users: (layout + 1) * num_users] = \
            network.compute_uplink_SE_centralized_uatf(collective_channels, combiners)
        average_SE_ul_heuristic[i, layout * num_users: (layout + 1) * num_users] = \
            network.compute_alpha_mu_SE_ul(alpha_ul[i, layout], mu_ul[i, layout], Q_ul, T_ul, C_v)


colors = ['r', 'b', 'g', 'm']
average_SE_ul = np.sort(average_SE_ul, axis=1)
average_SE_ul_heuristic = np.sort(average_SE_ul_heuristic, axis=1)
ax = plt.subplot(111)
for i, algorithm in enumerate(combining_algs):
    ax.plot(average_SE_ul[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{algorithm.upper()}", color=colors[i],
            linewidth=2, linestyle='dashed')
    ax.plot(average_SE_ul_heuristic[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{algorithm.upper()} (Approx)", color=colors[i],
            linewidth=2, linestyle='solid', alpha=0.6)
ax.set_xlabel(xlabel="Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.show()

average_SE_dl = np.sort(average_SE_dl, axis=1)
average_SE_dl_heuristic = np.sort(average_SE_dl_heuristic, axis=1)
ax = plt.subplot(111)
for i, algorithm in enumerate(combining_algs):
    ax.plot(average_SE_dl[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{algorithm.upper()}", color=colors[i],
            linewidth=2, linestyle='dashed')
    ax.plot(average_SE_dl_heuristic[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{algorithm.upper()} (Approx)", color=colors[i],
            linewidth=2, linestyle='solid', alpha=0.6)
ax.set_xlabel(xlabel="Downlink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.show()
test = []
