from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
import json
import os

setting = 'journal_standard'
cases = [['centralized', 'MMSE'], ['centralized', 'PMMSE'], ['centralized', 'PRZF'], ['centralized', 'MR']]
alg_indices = [0, 1, 1, 2]
layout_dir_path = f'StoredLayouts/{setting}/'
layout_dir_files = os.listdir(layout_dir_path)
solutions_path = f'MatlabSolutions/{setting}/'
solutions_dir_files = os.listdir(solutions_path)

# load optimized pilot solutions
ix_pilots = solutions_dir_files.index(f'sol_pilots.mat')
all_pilots = loadmat(solutions_path + solutions_dir_files[ix_pilots])['a'].astype('uint8')

# load optimized clusters
ix_clusters = solutions_dir_files.index(f'sol_clustering.mat')
all_clusters = loadmat(solutions_path + solutions_dir_files[ix_clusters])['x'].astype('uint8')

# load simulation setting files
layout_files = [file for file in layout_dir_files if file.split('.')[1] == 'mat' and file.startswith(setting)]
params_file = [file for file in layout_dir_files if file.split('.')[1] == 'json'][0]
params = json.load(open(layout_dir_path + params_file))

# local vars
num_pilots = params['pilot_len']
num_users = params['num_users']
num_cases = len(cases)
num_layouts = len(layout_files)
num_frames = 50

# simulation vars
network = CellFreeNetwork(**params)
network_rand = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
SE_ul = np.zeros((num_cases, num_layouts, num_users))
SE_ul_rand = np.zeros((num_cases, num_layouts, num_users))
SE_ul_emil = np.zeros((num_cases, num_layouts, num_users))
SE_dl = np.zeros((num_cases, num_layouts, num_users))
SE_dl_rand = np.zeros((num_cases, num_layouts, num_users))
SE_dl_emil = np.zeros((num_cases, num_layouts, num_users))

# main simulation loop
for layout in range(num_layouts):
    file = f"{setting}_layout_data_{layout}.mat"
    print(f"Layout: {layout + 1} out of {num_layouts} - file: {file}")
    layout_data_dict = loadmat(layout_dir_path + file)
    path_loss_shadowing = layout_data_dict['path_loss_shadowing']

    # delete extra dict keys
    dlt_keys = list(layout_data_dict.keys())[0:5]
    dlt_keys.extend(list(layout_data_dict.keys())[12:])
    for key in dlt_keys:
        del layout_data_dict[key]

    _, pilots_emil = emil_heuristic_dcc_pilot_assignment(path_loss_shadowing, num_pilots)
    pilots_rand = np.random.randint(0, num_pilots, (num_users,))

    for i, (processing, alg) in enumerate(cases):
        # get optimal clusters and pilots
        ix_alg = alg_indices[i]
        clusters = all_clusters[num_layouts * ix_alg + layout]
        pilots = all_pilots[num_layouts * ix_alg + layout]

        # set snapshots for baseline networks
        network_emil.set_clusters(clusters)
        network_emil.set_pilot_alloc(pilots_emil)
        network_emil.set_snapshot(**layout_data_dict)

        network_rand.set_clusters(clusters)
        network_rand.set_pilot_alloc(pilots_rand)
        network_rand.set_snapshot(**layout_data_dict)

        # set snapshot for optimal network
        network.set_clusters(clusters)
        network.set_pilot_alloc(pilots)
        network.set_snapshot(**layout_data_dict)

        channels, channel_estimates, _, _ = network.generate_channel_realizations(num_frames)

        channels_emil, channel_estimates_emil, _, _ = network_emil.generate_channel_realizations(num_frames)

        channels_rand, channel_estimates_rand, _, _ = network_rand.generate_channel_realizations(num_frames)

        # compute combiners and precoders
        combiners = network.simulate_uplink_centralized(alg, channels, channel_estimates)
        precoders = network.simulate_downlink_centralized(alg, channels, channel_estimates)

        combiners_emil = network_emil.simulate_uplink_centralized(alg, channels_emil, channel_estimates_emil)
        precoders_emil = network_emil.simulate_downlink_centralized(alg, channels_emil, channel_estimates_emil)

        combiners_rand = network_rand.simulate_uplink_centralized(alg, channels_rand, channel_estimates_rand)
        precoders_rand = network_rand.simulate_downlink_centralized(alg, channels_rand, channel_estimates_rand)

        # compute SEs
        SE_ul[i, layout] = network.compute_uplink_SE_centralized(channel_estimates, combiners)
        SE_dl[i, layout] = network.compute_downlink_SE_centralized(channels, precoders)

        SE_ul_emil[i, layout] = network_emil.compute_uplink_SE_centralized(channel_estimates_emil, combiners_emil)
        SE_dl_emil[i, layout] = network_emil.compute_downlink_SE_centralized(channels_emil, precoders_emil)

        SE_ul_rand[i, layout] = network_rand.compute_uplink_SE_centralized(channel_estimates_rand, combiners_rand)
        SE_dl_rand[i, layout] = network_rand.compute_downlink_SE_centralized(channels_rand, precoders_rand)


# calculate sum SE CDFs
sum_SE_CDF = (SE_dl + SE_ul).reshape((num_cases, num_layouts * num_users))
sum_SE_CDF.sort(axis=1)

sum_SE_CDF_rand = (SE_dl_rand + SE_ul_rand).reshape((num_cases, num_layouts * num_users))
sum_SE_CDF_rand.sort(axis=1)

sum_SE_CDF_emil = (SE_dl_emil + SE_ul_emil).reshape((num_cases, num_layouts * num_users))
sum_SE_CDF_emil.sort(axis=1)


# plot sum SE CDF
colors = ['r', 'b', 'g', 'm']
fig_sum = plt.figure()
ax = plt.subplot(111)
y_axis = np.linspace(0, 1, num_layouts * num_users)
for i in range(num_cases):
    ax.plot(sum_SE_CDF[i, :], y_axis, color=colors[i], linestyle='solid', label=cases[i][1] + ' (Optimized)', linewidth=3)
    ax.plot(sum_SE_CDF_emil[i, :], y_axis, color=colors[i], linestyle='dotted', label=cases[i][1] + ' (JPC)', linewidth=2)
    ax.plot(sum_SE_CDF_rand[i, :], y_axis, color=colors[i], linestyle='dashed', label=cases[i][1] + ' (random)', linewidth=2)

ax.set_xlabel(xlabel=f"Sum Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend(fontsize=11)
ax.tick_params(axis='both', labelsize=11)
plt.savefig(f"Figures/{setting}_CDF_Pilots", dpi=800)
plt.show()

algorithms = ["MMSE", "PMMSE", "PRZF", "MR"]
index_95 = np.argmin(np.abs(y_axis - 0.05))
x_axis = np.arange(len(algorithms))


# plot average min user SE bar chart
min_SE = np.mean(np.min(SE_ul + SE_dl, axis=2), axis=1)
min_SE_emil = np.mean(np.min(SE_ul_emil + SE_dl_emil, axis=2), axis=1)
min_SE_rand = np.mean(np.min(SE_ul_rand + SE_dl_rand, axis=2), axis=1)

fig = plt.figure()
ax = plt.subplot(111)
# ax_bar.grid(axis='y')
bar_size = 0.25
pps_one = ax.bar(x_axis - (bar_size + 0.02), min_SE, bar_size, label='Optimized', color='r')
pps_two = ax.bar(x_axis, min_SE_emil, bar_size, label='JCP', color='b')
pps_three = ax.bar(x_axis + (bar_size + 0.02), min_SE_rand, bar_size, label='Random', color='g')
ax.set_xticks(x_axis, algorithms)
ax.set_xlabel("Combining/precoding algorithm", fontsize=16)
ax.set_ylabel("min user SE (bits/s/Hz)", fontsize=16)
ax.legend(fontsize=11)
for p in pps_one:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
for p in pps_two:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
for p in pps_three:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
ax.tick_params(axis='both', labelsize=11)
plt.savefig(f"Figures/{setting}_CDF_min_piltos", dpi=800)
plt.show()


# plot 95% likely sum SE bar chart
fig = plt.figure()
ax = plt.subplot(111)
# ax_bar.grid(axis='y')
bar_size = 0.25
pps_one = ax.bar(x_axis - (bar_size + 0.02), sum_SE_CDF[:, index_95], bar_size, label='Optimized', color='r')
pps_three = ax.bar(x_axis, sum_SE_CDF_emil[:, index_95], bar_size, label='JCP', color='b')
pps_two = ax.bar(x_axis + (bar_size + 0.02), sum_SE_CDF_rand[:, index_95], bar_size, label='Random', color='g')

ax.set_xticks(x_axis, algorithms)
ax.set_xlabel("Combining/precoding algorithm", fontsize=16)
ax.set_ylabel("95% likely sum SE (bits/s/Hz)", fontsize=16)
ax.legend(fontsize=11)
for p in pps_one:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
for p in pps_three:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')

for p in pps_two:
    height = round(p.get_height(), 2)
    ax.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
ax.tick_params(axis='both', labelsize=11)
plt.savefig(f"Figures/{setting}_CDF_95_Pilots", dpi=800)
plt.show()
