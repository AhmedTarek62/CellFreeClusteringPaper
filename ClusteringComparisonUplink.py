
"""
This script simulates the network for a specific number of coherence frames to calculate the quantities needed to
compute the optimal clustering.
Afterwards, the network is simulated using the same layout using the optimal clustering and other clustering algorithms.
"""

from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import emil_heuristic_dcc_pilot_assignment
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Helpers import reshape_combiners
import numpy as np
import json
import os

setting = 'mock'
cases = [['centralized', 'MMSE'], ['centralized', 'MRC'], ['centralized', 'PMMSE'], ['centralized', 'PRZF']]


layout_dir_path = f'../GraphGenerationLayouts/'
layout_dir_files = os.listdir(layout_dir_path)
solutions_path = f'../MatlabSolutions/'
solutions_dir_files = os.listdir(solutions_path)
ix_opt_cluster = solutions_dir_files.index(f'sol_clustering.mat')
clusters = loadmat(solutions_path + solutions_dir_files[ix_opt_cluster])['x']
ix_pilot_allocs = solutions_dir_files.index(f'sol_pilots.mat')
pilot_allocs = loadmat(solutions_path + solutions_dir_files[ix_pilot_allocs])['a'].astype('uint8')
layout_files = [file for file in layout_dir_files if file.split('.')[1] == 'mat' and file.startswith(setting)]
params_file = [file for file in layout_dir_files if file.split('.')[1] == 'json'][0]
params = json.load(open(layout_dir_path + params_file))


num_pilots = params['pilot_len']
num_users = params['num_users']
network_opt = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
num_layouts = len(layout_files)
num_frames = 50
average_SE_opt = np.zeros((len(cases), num_layouts * num_users))
average_SE_emil = np.zeros((len(cases), num_layouts * num_users))


for layout, file in enumerate(layout_files):
    print(f"Layout: {layout + 1} out of {num_layouts}")
    layout_data_dict = loadmat(layout_dir_path + file)
    path_loss_shadowing = layout_data_dict['path_loss_shadowing']
    dlt_keys = list(layout_data_dict.keys())[0:5]
    dlt_keys.extend(list(layout_data_dict.keys())[12:])
    for key in dlt_keys:
        del layout_data_dict[key]


    clusters_emil, pilot_alloc_emil = emil_heuristic_dcc_pilot_assignment(path_loss_shadowing, num_pilots)
    network_emil.set_clusters(clusters_emil)
    network_emil.set_pilot_alloc(pilot_alloc_emil)
    network_emil.set_snapshot(**layout_data_dict)
    collective_channels_emil, collective_channel_estimates_emil, channels_emil, channel_estimates_emil = \
        network_emil.generate_channel_realizations(num_frames)

    for i, (processing, alg) in enumerate(cases):
        clusters_opt = clusters[num_layouts * i + layout]

        if len(pilot_allocs.shape)==3:
            pilot_alloc_opt = pilot_allocs[layout]
        else:
            pilot_alloc_opt = pilot_allocs[num_layouts * i + layout]
        pilot_alloc_opt = np.where(pilot_alloc_opt == 1)[1]

        network_opt.set_pilot_alloc(pilot_alloc_opt)
        network_opt.set_clusters(clusters_opt)
        network_opt.set_snapshot(**layout_data_dict)
        collective_channels_opt, collective_channel_estimates_opt, channels_opt, channel_estimates_opt = \
            network_opt.generate_channel_realizations(num_frames)

        combiners_opt = network_opt.simulate_uplink_centralized(alg, collective_channels_opt,
                                                        collective_channel_estimates_opt)
        # combiners_opt = reshape_combiners(network_opt.simulate_uplink_distributed(alg, channels_opt, channel_estimates_opt)[0])
        average_SE_layout_i = network_opt.compute_uplink_SE_centralized(collective_channel_estimates_opt, combiners_opt)
        # average_SE_layout_i = network_opt.compute_uplink_SE_centralized_uatf(collective_channels_opt, combiners_opt)
        average_SE_opt[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        combiners_emil = network_emil.simulate_uplink_centralized(alg, collective_channels_emil,
                                                        collective_channel_estimates_emil)
        # combiners_emil = reshape_combiners(network_emil.simulate_uplink_distributed(alg, channels_emil, channel_estimates_emil)[0])
        average_SE_layout_i = network_emil.compute_uplink_SE_centralized(collective_channel_estimates_emil, combiners_emil)
        # average_SE_layout_i = network_emil.compute_uplink_SE_centralized_uatf(collective_channels_emil, combiners_emil)
        average_SE_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i


average_SE_emil = np.sort(average_SE_emil, axis=1)
average_SE_opt = np.sort(average_SE_opt, axis=1)
gain_SE = np.round(np.mean(average_SE_opt - average_SE_emil, axis=1),4)
mmse_index = cases.index(['centralized','MMSE'])
pmmse_index = cases.index(['centralized','PMMSE'])
mrc_index = cases.index(['centralized','MRC'])
przf_index = cases.index(['centralized','PRZF'])
gain_SE_mmse = gain_SE[mmse_index]
gain_SE_pmmse = gain_SE[pmmse_index]
gain_SE_mrc = gain_SE[mrc_index]
gain_SE_przf = gain_SE[przf_index]


colors = ['r', 'b', 'g', 'm']
ax = plt.subplot(111)
for i in range(len(cases)):
    ax.plot(average_SE_emil[i,:], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='dashed',
            label=cases[i][1]+ ' (Emil)', linewidth=2)
    ax.plot(average_SE_opt[i,:], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='solid',
            label=cases[i][1]+ ' (Opt)', linewidth=2)



ax.set_xlabel(xlabel=f"Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.title(f"Gain MMSE: {gain_SE_mmse} - Gain PMMSE: {gain_SE_pmmse}\n "
          f"Gain PRZF: {gain_SE_przf} - Gain MRC: {gain_SE_mrc}")
plt.show()
test = []
