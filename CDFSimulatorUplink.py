from CellFreeNetwork import CellFreeNetwork
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from SettingParams import *


mock_params['clustering_alg'] = 'emil'
network = CellFreeNetwork(**mock_params)
num_layouts = 10
num_frames = 50
num_users = network.num_users
processing_cases = ['centralized']
combining_algs = ['mmse', 'pmmse', 'przf', 'mrc']
# processing_cases = ['distributed']
# combining_algs = ['mmse', 'pmmse', 'mrc']
cases = list(product(processing_cases, combining_algs))
average_SE = np.zeros((len(cases), num_layouts * num_users))

for layout in range(num_layouts):
    print(f"Layout: {layout + 1} out of {num_layouts}")
    network.generate_snapshot()
    collective_channels, collective_channel_estimates, channels, channel_estimates = \
        network.generate_channel_realizations(num_frames)
    for i, (processing, combining_alg) in enumerate(cases):
        if processing =='centralized':
            combiners = network.simulate_uplink_centralized(combining_alg, collective_channels,
                                                            collective_channel_estimates)
            # average_SE_layout_i = network.compute_uplink_SE_centralized_uatf(collective_channels, combiners)
            average_SE_layout_i = network.compute_uplink_SE_centralized(collective_channel_estimates, combiners)
            average_SE[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i
        elif processing=='distributed':
            combiners, ap_weights = network.simulate_uplink_distributed(combining_alg, channels, channel_estimates)
            average_SE_layout_i = network.compute_uplink_SE_distributed(channels, combiners, ap_weights)
            average_SE[i, layout * num_users: (layout + 1) * num_users] = average_SE_layout_i


colors = ['r', 'b', 'g', 'm']
average_SE = np.sort(average_SE, axis=1)
ax = plt.subplot(111)
for i, case in enumerate(cases):
    ax.plot(average_SE[i,:], np.linspace(0, 1, num_layouts * num_users), label=f"{case[1].upper()}", color=colors[i],
            linewidth=3, linestyle='dashed')
ax.set_xlabel(xlabel="Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.show()
