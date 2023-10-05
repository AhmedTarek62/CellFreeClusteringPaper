from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import *
from scipy.io import loadmat
from Metrics import *
import matplotlib.pyplot as plt
import numpy as np
import json
import os

setting = 'mock'
cases = [['centralized', 'MMSE'], ['centralized', 'PMMSE'], ['centralized', 'PRZF'], ['centralized', 'MR']]
alg_indices = [0, 1, 1, 2]
layout_dir_path = f'GraphGenerationLayouts/'
layout_dir_files = os.listdir(layout_dir_path)
solutions_path = f'MatlabSolutions/'
solutions_dir_files = os.listdir(solutions_path)

ix_pilot_allocs_opt = solutions_dir_files.index(f'sol_pilots.mat')
pilots_all_opt = loadmat(solutions_path + solutions_dir_files[ix_pilot_allocs_opt])['a'].astype('uint8')

ix_opt_cluster = solutions_dir_files.index(f'sol_clustering.mat')
clusters_all_opt = loadmat(solutions_path + solutions_dir_files[ix_opt_cluster])['x']
layout_files = [file for file in layout_dir_files if file.split('.')[1] == 'mat' and file.startswith(setting)]
params_file = [file for file in layout_dir_files if file.split('.')[1] == 'json'][0]
params = json.load(open(layout_dir_path + params_file))

num_pilots = params['pilot_len']
num_users = params['num_users']
network_opt = CellFreeNetwork(**params)
network_chen = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
num_layouts = len(layout_files)
num_frames = 50
average_SE_ul_opt = np.zeros((len(cases), num_layouts * num_users))
average_SE_ul_chen = np.zeros((len(cases), num_layouts * num_users))
average_SE_ul_emil = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_opt = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_chen = np.zeros((len(cases), num_layouts * num_users))
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

    clusters_emil, pilots_emil = emil_heuristic_dcc_pilot_assignment(path_loss_shadowing, num_pilots)
    clusters_chen = massive_access_clustering(path_loss_shadowing, num_pilots)

    for i, (processing, alg) in enumerate(cases):
        ix_alg = alg_indices[i]
        clusters_opt = clusters_all_opt[num_layouts * ix_alg + layout]
        if pilots_all_opt.shape[0] == num_layouts:
            pilots_opt = pilots_all_opt[layout]
        else:
            pilots_opt = pilots_all_opt[num_layouts * ix_alg + layout]
        pilots_opt = np.where(pilots_opt == 1)[1]

        network_opt.set_pilot_alloc(pilots_opt)
        network_opt.set_clusters(clusters_opt)
        network_opt.set_snapshot(**layout_data_dict)

        network_chen.set_pilot_alloc(pilots_emil)
        network_chen.set_clusters(clusters_chen)
        network_chen.set_snapshot(**layout_data_dict)

        network_emil.set_pilot_alloc(pilots_emil)
        network_emil.set_clusters(clusters_emil)
        network_emil.set_snapshot(**layout_data_dict)

        channels_opt, channel_estimates_opt, _, _ = network_opt.generate_channel_realizations(num_frames)

        channels_chen, channel_estimates_chen, _, _ = network_chen.generate_channel_realizations(num_frames)

        channels_emil, channel_estimates_emil, _, _ = network_emil.generate_channel_realizations(num_frames)

        combiners_opt = network_opt.simulate_uplink_centralized(alg, channels_opt, channel_estimates_opt)
        precoders_opt = network_opt.simulate_downlink_centralized(alg, channels_opt, channel_estimates_opt)

        combiners_chen = network_chen.simulate_uplink_centralized(alg, channels_chen, channel_estimates_chen)
        precoders_chen = network_chen.simulate_downlink_centralized(alg, channels_chen, channel_estimates_chen)

        combiners_emil = network_emil.simulate_uplink_centralized(alg, channels_emil, channel_estimates_emil)
        precoders_emil = network_emil.simulate_downlink_centralized(alg, channels_emil, channel_estimates_emil)

        average_SE_layout_i = network_opt.compute_uplink_SE_centralized(channel_estimates_opt, combiners_opt)
        average_SE_ul_opt[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_opt.compute_downlink_SE_centralized(channels_opt, precoders_opt)
        average_SE_dl_opt[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_chen.compute_uplink_SE_centralized(channel_estimates_chen, combiners_chen)
        average_SE_ul_chen[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_chen.compute_downlink_SE_centralized(channels_chen, precoders_chen)
        average_SE_dl_chen[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_uplink_SE_centralized(channel_estimates_emil, combiners_emil)
        average_SE_ul_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_downlink_SE_centralized(channels_emil, precoders_emil)
        average_SE_dl_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i


average_SE_ul_emil = np.sort(average_SE_ul_emil, axis=1)
average_SE_ul_chen = np.sort(average_SE_ul_chen, axis=1)
average_SE_ul_opt = np.sort(average_SE_ul_opt, axis=1)

average_SE_dl_emil = np.sort(average_SE_dl_emil, axis=1)
average_SE_dl_chen = np.sort(average_SE_dl_chen, axis=1)
average_SE_dl_opt = np.sort(average_SE_dl_opt, axis=1)

# colors = ['r', 'b', 'g', 'm']
# fig_sum = plt.figure()
# ax_sum = plt.subplot(111)
# for i in range(len(cases)):
#     ax_sum.plot(average_SE_dl_emil[i, :] + average_SE_ul_emil[i, :], np.linspace(0, 1, num_layouts * num_users),
#                 color=colors[i], linestyle='dashed', label=cases[i][1] + ' (Baseline)', linewidth=2)
#     ax_sum.plot(average_SE_dl_opt[i, :] + average_SE_ul_opt[i, :], np.linspace(0, 1, num_layouts * num_users),
#                 color=colors[i], linestyle='solid', label=cases[i][1] + ' (Optimized)', linewidth=3)
#
# ax_sum.set_xlabel(xlabel=f"Sum Spectral Efficiency (bits/s/Hz)", fontsize=16)
# ax_sum.set_ylabel(ylabel="CDF", fontsize=16)
# ax_sum.legend(fontsize=11)
# ax_sum.tick_params(axis='both', labelsize=11)
# plt.savefig(f"Figures/CDF_optimal_vs_emil", dpi=800)
# plt.show()

algorithms = ["MMSE", "PMMSE", "PRZF", "MR"]
index_95 = np.argmin(np.abs(np.linspace(0, 1, num_layouts * num_users) - 0.05))
sum_SE_opt = average_SE_ul_opt + average_SE_dl_opt
sum_SE_chen = average_SE_ul_chen + average_SE_dl_chen
sum_SE_emil = average_SE_ul_emil + average_SE_dl_emil
X_axis = np.arange(len(algorithms))

fig_bar = plt.figure(figsize=(7.2, 4.8))
ax_bar = plt.subplot(111)
# ax_bar.grid(axis='y')
pps_one = ax_bar.bar(X_axis - 0.2, sum_SE_opt[:, index_95], 0.15, label='Optimized', color='r')
pps_two = ax_bar.bar(X_axis, sum_SE_chen[:, index_95], 0.15, label='GCA', color='m')
pps_three = ax_bar.bar(X_axis + 0.2, sum_SE_emil[:, index_95], 0.15, label='JPC', color='b')

ax_bar.set_xticks(X_axis, algorithms)
ax_bar.set_xlabel("Combining/precoding algorithm", fontsize=16)
ax_bar.set_ylabel("95% likely sum SE (bits/s/Hz)", fontsize=16)
ax_bar.legend(fontsize=11)
for p in pps_one:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height),ha='center')
for p in pps_two:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height),ha='center')
for p in pps_three:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height),ha='center')
ax_bar.tick_params(axis='both', labelsize=11)
# plt.savefig(f"Figures/95_SE_Globecom", dpi=800)
plt.show()


