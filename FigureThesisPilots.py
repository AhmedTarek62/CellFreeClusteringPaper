from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import emil_heuristic_dcc_pilot_assignment
from scipy.io import loadmat
import matplotlib.pyplot as plt
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
layout_files = [file for file in layout_dir_files if file.split('.')[1] == 'mat' and file.startswith(setting)]
params_file = [file for file in layout_dir_files if file.split('.')[1] == 'json'][0]
params = json.load(open(layout_dir_path + params_file))

num_pilots = params['pilot_len']
num_users = params['num_users']
network_opt = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
num_layouts = len(layout_files)
num_frames = 50
average_SE_ul_opt = np.zeros((len(cases), num_layouts * num_users))
average_SE_ul_emil = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_opt = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_emil = np.zeros((len(cases), num_layouts * num_users))


for layout in range(num_layouts):
    file = f"{setting}_layout_data_{layout}.mat"
    print(f"Layout: {layout + 1} out of {num_layouts} - file: {file}")
    layout_data_dict = loadmat(layout_dir_path + file)
    path_loss_shadowing = layout_data_dict['path_loss_shadowing']
    dlt_keys = list(layout_data_dict.keys())[0:5]
    dlt_keys.extend(list(layout_data_dict.keys())[12:])
    for key in dlt_keys:
        del layout_data_dict[key]

    clusters_emil, pilot_alloc_emil = emil_heuristic_dcc_pilot_assignment(path_loss_shadowing, num_pilots)

    for i, (processing, alg) in enumerate(cases):
        clusters_opt = clusters[num_layouts * i + layout]

        network_opt.set_pilot_alloc(pilot_alloc_emil)
        network_opt.set_clusters(clusters_opt)
        network_opt.set_snapshot(**layout_data_dict)

        network_emil.set_pilot_alloc(pilot_alloc_emil)
        network_emil.set_clusters(clusters_emil)
        network_emil.set_snapshot(**layout_data_dict)

        collective_channels_opt, collective_channel_estimates_opt, _, _ = \
            network_opt.generate_channel_realizations(num_frames)

        collective_channels_emil, collective_channel_estimates_emil, _, _ = \
            network_emil.generate_channel_realizations(num_frames)

        combiners_opt = network_opt.simulate_uplink_centralized(alg, collective_channels_opt,
                                                        collective_channel_estimates_opt)
        precoders_opt = network_opt.simulate_downlink_centralized(alg, collective_channels_opt,
                                                        collective_channel_estimates_opt)
        combiners_emil = network_emil.simulate_uplink_centralized(alg, collective_channels_emil,
                                                        collective_channel_estimates_emil)
        precoders_emil = network_emil.simulate_downlink_centralized(alg, collective_channels_emil,
                                                        collective_channel_estimates_emil)

        average_SE_layout_i = network_opt.compute_uplink_SE_centralized(collective_channel_estimates_opt, combiners_opt)
        average_SE_ul_opt[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_opt.compute_downlink_SE_centralized(collective_channels_opt, precoders_opt)
        average_SE_dl_opt[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_uplink_SE_centralized(collective_channel_estimates_emil, combiners_emil)
        average_SE_ul_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_downlink_SE_centralized(collective_channels_emil, precoders_emil)
        average_SE_dl_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i


average_SE_ul_emil = np.sort(average_SE_ul_emil, axis=1)
average_SE_ul_opt = np.sort(average_SE_ul_opt, axis=1)

gain_SE = np.round(np.mean(average_SE_ul_opt - average_SE_ul_emil, axis=1),4)
mmse_index = cases.index(['centralized','MMSE'])
pmmse_index = cases.index(['centralized','PMMSE'])
mrc_index = cases.index(['centralized','MRC'])
przf_index = cases.index(['centralized','PRZF'])
gains_ul = [gain_SE[mmse_index], gain_SE[pmmse_index], gain_SE[mrc_index], gain_SE[przf_index]]

colors = ['r', 'b', 'g', 'm']
fig_ul = plt.figure()
ax_ul = plt.subplot(111)
for i in range(len(cases)):
    ax_ul.plot(average_SE_ul_emil[i, :], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='dashed',
               label=cases[i][1]+ ' (Baseline)', linewidth=2)
    ax_ul.plot(average_SE_ul_opt[i, :], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='solid',
               label=cases[i][1]+ ' (Optimized)', linewidth=2)

ax_ul.set_xlabel(xlabel=f"Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax_ul.set_ylabel(ylabel="CDF", fontsize=16)
ax_ul.legend()
plt.show()

average_SE_dl_emil = np.sort(average_SE_dl_emil, axis=1)
average_SE_dl_opt = np.sort(average_SE_dl_opt, axis=1)

gain_SE = np.round(np.mean(average_SE_dl_opt - average_SE_dl_emil, axis=1),4)
mmse_index = cases.index(['centralized','MMSE'])
pmmse_index = cases.index(['centralized','PMMSE'])
mrc_index = cases.index(['centralized','MRC'])
przf_index = cases.index(['centralized','PRZF'])
gains_dl = [gain_SE[mmse_index], gain_SE[pmmse_index], gain_SE[mrc_index], gain_SE[przf_index]]

colors = ['r', 'b', 'g', 'm']
fig_dl = plt.figure()
ax_dl = plt.subplot(111)
for i in range(len(cases)):
    ax_dl.plot(average_SE_dl_emil[i, :], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='dashed',
               label=cases[i][1]+ ' (Baseline)', linewidth=2)
    ax_dl.plot(average_SE_dl_opt[i, :], np.linspace(0, 1, num_layouts * num_users), color=colors[i], linestyle='solid',
               label=cases[i][1]+ ' (Optimized)', linewidth=2)

ax_dl.set_xlabel(xlabel=f"Downlink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax_dl.set_ylabel(ylabel="CDF", fontsize=16)
ax_dl.legend()
plt.show()
plt.savefig(f"../Figures/CDF_ul_optimal_vs_baseline_clustering", dpi=800)

