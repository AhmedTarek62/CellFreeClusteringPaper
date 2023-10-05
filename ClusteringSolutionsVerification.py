from CellFreeNetwork import CellFreeNetwork
from Helpers import *
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

SE_ul_opt = np.zeros((len(cases), num_layouts))
SE_ul_emil = np.zeros((len(cases), num_layouts))


for layout in range(num_layouts):
    file = f"{setting}_layout_data_{layout}.mat"
    # print(f"Layout: {layout + 1} out of {num_layouts} - file: {file}")
    layout_data_dict = loadmat(layout_dir_path + file)
    path_loss_shadowing = layout_data_dict['path_loss_shadowing']
    clusters_emil = layout_data_dict['clusters_emil'].astype('float')
    pilot_alloc_emil = layout_data_dict['pilot_alloc_emil'].reshape((-1,))
    dlt_keys = list(layout_data_dict.keys())[0:5]
    dlt_keys.extend(list(layout_data_dict.keys())[12:])
    for key in dlt_keys:
        del layout_data_dict[key]

    # network_emil.set_clusters(clusters_emil)
    # network_emil.set_pilot_alloc(pilot_alloc_emil)
    # network_emil.set_snapshot(**layout_data_dict)

    for i, (processing, alg) in enumerate(cases):
        alg_file = f"{alg.upper()}_layout_{layout}.mat"
        sinr_dict = loadmat(layout_dir_path + alg_file)
        clusters_opt = clusters[num_layouts * i + layout]
        # pilots_opt = pilot_allocs[num_layouts * i + layout]
        # network_opt.set_pilot_alloc(pilots_opt)
        # network_opt.set_clusters(clusters_opt)
        # network_opt.set_snapshot(**layout_data_dict)
        Q_ul, T_ul, c_v = sinr_dict['Q_ul'], sinr_dict['T_ul'], sinr_dict['C_v']
        Q_dl, T_dl = sinr_dict['Q_dl'], sinr_dict['T_dl']
        SE_ul_opt[i, layout] = calculate_SE_ul(clusters_opt, T_ul, Q_ul, c_v)
        SE_ul_emil[i, layout] = calculate_SE_ul(clusters_emil, T_ul, Q_ul, c_v)
test = []

