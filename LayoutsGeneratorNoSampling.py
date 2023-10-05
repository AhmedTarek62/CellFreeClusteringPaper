from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import emil_heuristic_dcc_pilot_assignment
from scipy.io import savemat
import json
from SettingParams import *

setting = 'mock'
if setting == 'mock':
    params = mock_params
else:
    params = final_params

num_layouts = 10
combining_algs = ["MMSE", "PMMSE", "PRZF", "MRC"]
network = CellFreeNetwork(**params)
json.dump(params, open(f"../GraphGenerationLayouts/" + setting + "_setting_params.json", 'w'))

for layout in range(num_layouts):
    network.generate_snapshot()
    clusters_emil, pilot_alloc_emil = emil_heuristic_dcc_pilot_assignment(network.channel_model.path_loss_shadowing,
                                                                          network.pilot_len)
    Q_dl, Q_ul, T_ul, T_dl, C_v = network.compute_clustering_optimization_model_mrc()
    layouts_data_dict = {
        'clusters_emil': clusters_emil,
        'pilot_alloc_emil': pilot_alloc_emil,
        'aps_loc': network.aps_loc,
        'users_loc': network.users_loc,
        'user_user_distance': network.user_user_distance,
        'user_ap_distance': network.user_ap_distance,
        'path_loss_shadowing': network.channel_model.path_loss_shadowing,
        'user_ap_corr': network.channel_model.user_ap_corr,
        'user_ap_corr_chol': network.channel_model.user_ap_corr_chol,
        'Q_dl': Q_dl,
        'Q_ul': Q_ul,
        'T_dl': T_dl,
        'T_ul': T_ul,
        'C_v': C_v
    }
    savemat(f"../GraphGenerationLayouts/{setting}_layout_data_{layout}.mat", layouts_data_dict)

