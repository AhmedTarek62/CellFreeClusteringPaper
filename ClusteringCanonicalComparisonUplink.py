from CellFreeNetwork import CellFreeNetwork
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from SettingParams import *


network = CellFreeNetwork(**mock_params)
mock_params['clustering_alg'] = 'emil'
network_emil = CellFreeNetwork(**mock_params)
num_layouts = 40
num_frames = 50
num_users = network.num_users
processing_cases = ['centralized']
combining_algs = ['mmse', 'pmmse', 'przf', 'mrc']
# processing_cases = ['distributed']
# combining_algs = ['mmse', 'pmmse', 'mrc']
cases = list(product(processing_cases, combining_algs))
average_SE = np.zeros((len(cases), num_layouts * num_users))
average_SE_emil = np.zeros((len(cases), num_layouts * num_users))

for layout in range(num_layouts):
    print(f"Layout: {layout + 1} out of {num_layouts}")
    network.generate_snapshot()
    collective_channels, collective_channel_estimates, _, _ = \
        network.generate_channel_realizations(num_frames)

    network_emil.generate_snapshot()
    collective_channels_emil, collective_channel_estimates_emil, _, _ = \
        network_emil.generate_channel_realizations(num_frames)
    for i, (processing, combining_alg) in enumerate(cases):
        combiners = network.simulate_uplink_centralized(combining_alg, collective_channels,
                                                        collective_channel_estimates)
        average_SE_layout_i = network.compute_uplink_SE_centralized(collective_channel_estimates, combiners)
        average_SE[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

        combiners = network_emil.simulate_uplink_centralized(combining_alg, collective_channels_emil,
                                                             collective_channel_estimates_emil)
        average_SE_layout_i = network_emil.compute_uplink_SE_centralized(collective_channel_estimates_emil, combiners)
        average_SE_emil[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i

colors = ['r', 'b', 'g', 'm']

average_SE = np.sort(average_SE, axis=1)
average_SE_emil = np.sort(average_SE_emil, axis=1)
ax = plt.subplot(111)
for i, case in enumerate(cases):
    ax.plot(average_SE[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{case[1].upper()} (All)", color=colors[i],
            linewidth=3, linestyle='dashed')
    ax.plot(average_SE_emil[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{case[1].upper()} (DCC)", color=colors[i],
            linewidth=3, linestyle='solid')

ax.set_xlabel(xlabel="Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.show()
test = []
