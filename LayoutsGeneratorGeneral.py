from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import *
from scipy.io import savemat
import json
from SettingParams import *

params = journal_params_massive
setting = 'journal_massive'

print(params)
num_layouts = 40
num_frames = 20
network = CellFreeNetwork(**params)
json.dump(params, open(f"GraphGenerationLayouts/" + setting + "_setting_params.json", 'w'))

for layout in range(num_layouts):
    print(f"Layout {layout + 1} out of {num_layouts}")
    network.generate_snapshot()
    clusters_emil, pilot_alloc_emil = emil_heuristic_dcc_pilot_assignment(network.channel_model.path_loss_shadowing,
                                                                          network.pilot_len)
    clusters_chen = massive_access_clustering(network.channel_model.path_loss_shadowing, network.pilot_len)
    collective_channels, collective_channel_estimates, channels, channel_estimates = \
        network.generate_channel_realizations(num_frames)

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
        'collective_channels': collective_channels,
        'clusters_chen': clusters_chen
    }
    savemat(f"GraphGenerationLayouts/{setting}_layout_data_{layout}.mat", layouts_data_dict)
