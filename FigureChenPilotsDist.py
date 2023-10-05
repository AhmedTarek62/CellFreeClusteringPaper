from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import *
from PilotAssignmentsAlgorithms import IB_KM, user_group
from Helpers import calculate_ce
from scipy.io import loadmat
import matplotlib.pyplot as plt
from SettingParams import mock_params
import numpy as np
import json
import os

cases = [['distributed', 'MMSE'], ['distributed', 'PMMSE'], ['distributed', 'PRZF'], ['distributed', 'MR']]
params = mock_params
params['clustering_alg'] = 'emil'
params['num_users'] = 12
params['num_aps'] = 30
params['pilot_len'] = 3
params['coverage_area_len'] = 250
num_pilots = params['pilot_len']
num_users = params['num_users']

network_km = CellFreeNetwork(**params)
network_ug = CellFreeNetwork(**params)
network_emil = CellFreeNetwork(**params)
num_frames = 50
num_layouts = 40
average_SE_ul_km = np.zeros((len(cases), num_layouts * num_users))
average_SE_ul_ug = np.zeros((len(cases), num_layouts * num_users))
average_SE_ul_emil = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_km = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_ug = np.zeros((len(cases), num_layouts * num_users))
average_SE_dl_emil = np.zeros((len(cases), num_layouts * num_users))

layout = 0
while layout < num_layouts:
    print(f"Layout: {layout + 1} out of {num_layouts}")
    network_km.generate_snapshot()
    clusters = massive_access_clustering(network_km.channel_model.path_loss_shadowing, network_km.pilot_len)
    _, pilots_emil = emil_heuristic_dcc_pilot_assignment(network_km.channel_model.path_loss_shadowing, network_km.pilot_len)
    pilots_ug = user_group(network_km.channel_model.path_loss_shadowing, network_km.pilot_len, clusters)
    pilots_km = IB_KM(network_km)

    if len(set(pilots_ug)) != network_km.pilot_len:
        print("UG failed! Repeat.")
        continue

    network_km.set_clusters(clusters)
    network_km.set_pilot_alloc(pilots_km)
    network_ug.set_clusters(clusters)
    network_ug.set_pilot_alloc(pilots_ug)
    network_emil.set_clusters(clusters)
    network_emil.set_pilot_alloc(pilots_emil)

    network_km.set_snapshot(
        network_km.aps_loc,
        network_km.users_loc,
        network_km.user_user_distance,
        network_km.user_ap_distance,
        network_km.channel_model.path_loss_shadowing,
        network_km.channel_model.user_ap_corr,
        network_km.channel_model.user_ap_corr_chol
    )

    network_ug.set_snapshot(
        network_km.aps_loc,
        network_km.users_loc,
        network_km.user_user_distance,
        network_km.user_ap_distance,
        network_km.channel_model.path_loss_shadowing,
        network_km.channel_model.user_ap_corr,
        network_km.channel_model.user_ap_corr_chol
    )

    network_emil.set_snapshot(
        network_km.aps_loc,
        network_km.users_loc,
        network_km.user_user_distance,
        network_km.user_ap_distance,
        network_km.channel_model.path_loss_shadowing,
        network_km.channel_model.user_ap_corr,
        network_km.channel_model.user_ap_corr_chol
    )

    print(f"UG: {np.round(calculate_ce(network_ug),2)}\n"
          f"KM: {np.round(calculate_ce(network_km), 2)}\n"
          f"Emil: {np.round(calculate_ce(network_emil), 2)}")

    _, _, channels_km, channel_estimates_km = network_km.generate_channel_realizations(num_frames)

    _, _, channels_ug, channel_estimates_ug = network_ug.generate_channel_realizations(num_frames)

    _, _, channels_emil, channel_estimates_emil = network_emil.generate_channel_realizations(num_frames)

    for i, (processing, alg) in enumerate(cases):

        combiners_km, _ = network_km.simulate_uplink_distributed(alg, channels_km, channel_estimates_km)
        precoders_km, _ = network_km.simulate_downlink_distributed(alg, channels_km, channel_estimates_km)

        combiners_ug, _ = network_ug.simulate_uplink_distributed(alg, channels_ug, channel_estimates_ug)
        precoders_ug, _ = network_ug.simulate_downlink_distributed(alg, channels_ug, channel_estimates_ug)

        combiners_emil, _ = network_emil.simulate_uplink_distributed(alg, channels_emil, channel_estimates_emil)
        precoders_emil, _ = network_emil.simulate_downlink_distributed(alg, channels_emil, channel_estimates_emil)

        average_SE_layout_i = network_km.compute_uplink_SE_distributed(channel_estimates_km, combiners_km)
        average_SE_ul_km[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_km.compute_downlink_SE_distributed(channels_km, precoders_km)
        average_SE_dl_km[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_ug.compute_uplink_SE_distributed(channel_estimates_ug, combiners_ug)
        average_SE_ul_ug[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_ug.compute_downlink_SE_distributed(channels_ug, precoders_ug)
        average_SE_dl_ug[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_uplink_SE_distributed(channel_estimates_emil, combiners_emil)
        average_SE_ul_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        average_SE_layout_i = network_emil.compute_downlink_SE_distributed(channels_emil, precoders_emil)
        average_SE_dl_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

    layout += 1


average_SE_ul_emil = np.sort(average_SE_ul_emil, axis=1)
average_SE_ul_km = np.sort(average_SE_ul_km, axis=1)
average_SE_ul_ug = np.sort(average_SE_ul_ug, axis=1)

average_SE_dl_emil = np.sort(average_SE_dl_emil, axis=1)
average_SE_dl_km = np.sort(average_SE_dl_km, axis=1)
average_SE_dl_ug = np.sort(average_SE_dl_ug, axis=1)

# colors = ['r', 'b', 'g', 'm']
# fig_sum = plt.figure()
# ax_sum = plt.subplot(111)
# for i in range(len(cases)):
#     ax_sum.plot(average_SE_dl_emil[i, :] + average_SE_ul_emil[i, :], np.linspace(0, 1, num_layouts * num_users),
#                 color=colors[i], linestyle='dashed', label=cases[i][1] + ' (Chen-Emil)', linewidth=2)
#     ax_sum.plot(average_SE_dl_km[i, :] + average_SE_ul_km[i, :], np.linspace(0, 1, num_layouts * num_users),
#                 color=colors[i], linestyle='solid', label=cases[i][1] + ' (Chen-KM)', linewidth=3)
#
#
# ax_sum.set_xlabel(xlabel=f"Sum Spectral Efficiency (bits/s/Hz)", fontsize=16)
# ax_sum.set_ylabel(ylabel="CDF", fontsize=16)
# ax_sum.legend(fontsize=11)
# ax_sum.tick_params(axis='both', labelsize=11)
# plt.show()

algorithms = ["MMSE", "PMMSE", "PRZF", "MR"]
index_95 = np.argmin(np.abs(np.linspace(0, 1, num_layouts * num_users) - 0.05))
sum_SE_km = average_SE_ul_km + average_SE_dl_km
sum_SE_ug = average_SE_ul_ug + average_SE_dl_ug
sum_SE_emil = average_SE_ul_emil + average_SE_dl_emil
X_axis = np.arange(len(algorithms))

fig_bar = plt.figure(figsize=(7.2, 4.8))
ax_bar = plt.subplot(111)
# ax_bar.grid(axis='y')
pps_one = ax_bar.bar(X_axis - 0.2, sum_SE_ug[:, index_95], 0.15, label='Chen-UG', color='r')
pps_two = ax_bar.bar(X_axis, sum_SE_km[:, index_95], 0.15, label='Chen-KM', color='b')
pps_three = ax_bar.bar(X_axis + 0.2, sum_SE_emil[:, index_95], 0.15, label='Chen-Emil', color='g')


ax_bar.set_xticks(X_axis, algorithms)
ax_bar.set_xlabel("Combining/precoding algorithm", fontsize=16)
ax_bar.set_ylabel("95% likely sum SE (bits/s/Hz)", fontsize=16)
ax_bar.legend(fontsize=11)
for p in pps_one:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
for p in pps_two:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
for p in pps_three:
    height = round(p.get_height(), 2)
    ax_bar.text(x=p.get_x() + p.get_width() / 2, y=height+.10, s="{}".format(height), ha='center')
ax_bar.tick_params(axis='both', labelsize=11)
plt.show()
test = []
