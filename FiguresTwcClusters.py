from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import *
from scipy.io import loadmat
from Metrics import *
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

ix_pilot_allocs_opt = solutions_dir_files.index(f'sol_pilots.mat')
pilots_all_opt = loadmat(solutions_path + solutions_dir_files[ix_pilot_allocs_opt])['a'].astype('uint8')

ix_opt_cluster = solutions_dir_files.index(f'sol_clustering.mat')
clusters_all_opt = loadmat(solutions_path + solutions_dir_files[ix_opt_cluster])['x']
layout_files = [file for file in layout_dir_files if file.split('.')[1] == 'mat' and file.startswith(setting)]
params_file = [file for file in layout_dir_files if file.split('.')[1] == 'json'][0]
params = json.load(open(layout_dir_path + params_file))

num_pilots = params['pilot_len']
num_users = params['num_users']
num_aps = params['num_aps']
network_opt = CellFreeNetwork(**params)
network_chen = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
num_layouts = len(layout_files)

metric_opt = np.zeros((len(cases), num_layouts,))
metric_chen = np.zeros((len(cases), num_layouts,))
metric_emil = np.zeros((len(cases), num_layouts,))

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
        # metric_opt[i, layout] = compute_greedy_index(path_loss_shadowing, clusters_opt, num_pilots)
        # metric_chen[i, layout] = compute_greedy_index(path_loss_shadowing, clusters_chen, num_pilots)
        # metric_emil[i, layout] = compute_greedy_index(path_loss_shadowing, clusters_emil, num_pilots)
        # metric_opt[i, layout] = np.mean(np.sum(clusters_opt, 1))
        # metric_chen[i, layout] = np.mean(np.sum(clusters_chen, 1))
        # metric_emil[i, layout] = np.mean(np.sum(clusters_emil, 1))


algorithms = ["MMSE", "PMMSE", "PRZF", "MR"]
X_axis = np.arange(len(algorithms))

fig_bar = plt.figure(figsize=(7.2, 4.8))
ax_bar = plt.subplot(111)
# ax_bar.grid(axis='y')
pps_one = ax_bar.bar(X_axis - 0.2, np.mean(metric_opt, 1), 0.15, label='Optimized', color='r')
pps_two = ax_bar.bar(X_axis, np.mean(metric_chen, 1), 0.15, label='GCA', color='m')
pps_three = ax_bar.bar(X_axis + 0.2, np.mean(metric_emil, 1), 0.15, label='JPC', color='b')

ax_bar.set_xticks(X_axis, algorithms)
ax_bar.set_xlabel("Combining/precoding algorithm", fontsize=16)
ax_bar.set_ylabel("Greedy Index", fontsize=16)
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
test = []

