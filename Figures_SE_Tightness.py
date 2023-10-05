from CellFreeNetwork import CellFreeNetwork
import matplotlib.pyplot as plt
from SettingParams import *


setting = 'massive'
if setting == 'standard':
    params = journal_params
else:
    params = journal_params_massive

params['clustering_alg'] = 'emil'
network = CellFreeNetwork(**params)
num_layouts = 40
num_frames = 50
num_users = params['num_users']
num_aps = params['num_aps']
algs = ['MMSE', 'PMMSE', 'PRZF', 'MR']
num_algs = len(algs)

alpha_dl = np.zeros((num_algs, num_layouts, num_users))
mu_dl = np.zeros((num_algs, num_layouts, num_users))
alpha_ul = np.zeros((num_algs, num_layouts, num_users))
mu_ul = np.zeros((num_algs, num_layouts, num_users))

SE_ul = np.zeros((num_algs, num_layouts * num_users))
SE_dl = np.zeros((num_algs, num_layouts * num_users))

SE_ul_hat = np.zeros((num_algs, num_layouts * num_users))
SE_dl_hat = np.zeros((num_algs, num_layouts * num_users))

for layout in range(num_layouts):
    print(f"Layout: {layout + 1} out of {num_layouts}")
    network.generate_snapshot()
    channels, channel_estimates, _, _ = network.generate_channel_realizations(num_frames)

    all_algs_Q_dl = []
    all_algs_Q_ul = []
    all_algs_T_dl = []
    all_algs_T_ul = []
    all_algs_C_v = []

    for i, alg in enumerate(algs):
        precoders = network.simulate_downlink_centralized(alg, channels, channel_estimates)
        combiners = network.simulate_uplink_centralized(alg, channels, channel_estimates)
        Q_dl, Q_ul, T_ul, T_dl, C_v = network.compute_clustering_optimization_model_mrc()
        all_algs_T_dl.append(T_dl)
        all_algs_T_ul.append(T_ul)
        all_algs_Q_dl.append(Q_dl)
        all_algs_Q_ul.append(Q_ul)
        all_algs_C_v.append(C_v)
        alpha_dl[i, layout], mu_dl[i, layout] = (
            network.estimate_alpha_mu_dl(channels, precoders, all_algs_Q_dl[i], all_algs_T_dl[i]))
        alpha_ul[i, layout], mu_ul[i, layout] = (
            network.estimate_alpha_mu_ul(channels, combiners, all_algs_Q_ul[i], all_algs_T_ul[i], all_algs_C_v[i]))

    channels, channel_estimates, _, _ = network.generate_channel_realizations(num_frames)

    for i, alg in enumerate(algs):
        precoders = network.simulate_downlink_centralized(alg, channels, channel_estimates)
        SE_dl[i, layout * num_users: (layout + 1) * num_users] =\
            network.compute_downlink_SE_centralized(channels, precoders)
        SE_dl_hat[i, layout * num_users: (layout + 1) * num_users] = \
            network.compute_alpha_mu_SE_dl(alpha_dl[i, layout], mu_dl[i, layout], all_algs_Q_dl[i], all_algs_T_dl[i])

        combiners = network.simulate_uplink_centralized(alg, channels, channel_estimates)
        SE_ul[i, layout * num_users: (layout + 1) * num_users] =\
            network.compute_uplink_SE_centralized_uatf(channels, combiners)
        SE_ul_hat[i, layout * num_users: (layout + 1) * num_users] = network.compute_alpha_mu_SE_ul(
            alpha_ul[i, layout], mu_ul[i, layout], all_algs_Q_ul[i], all_algs_T_ul[i], all_algs_C_v[i])


colors = ['r', 'b', 'g', 'm']
SE_ul = np.sort(SE_ul, axis=1)
SE_ul_hat = np.sort(SE_ul_hat, axis=1)
ax = plt.subplot(111)
for i, alg in enumerate(algs):
    ax.plot(SE_ul[i, :], np.linspace(0, 1, num_layouts * num_users), label=f"{alg.upper()}", color=colors[i],
            linewidth=3, linestyle='dashed')
    ax.plot(SE_ul_hat[i, :], np.linspace(0, 1, num_layouts * num_users), label=f"{alg.upper()} (Approx)",
            color=colors[i], linewidth=3, linestyle='solid', alpha=0.6)
ax.set_xlabel(xlabel="Uplink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.savefig(f"Figures/{setting}_CDF_ul_tightness", dpi=800)
plt.show()

SE_dl = np.sort(SE_dl, axis=1)
SE_dl_hat = np.sort(SE_dl_hat, axis=1)
ax = plt.subplot(111)
for i, alg in enumerate(algs):
    ax.plot(SE_dl[i, :], np.linspace(0, 1, num_layouts * num_users), label=f"{alg.upper()}", color=colors[i],
            linewidth=3, linestyle='dashed')
    ax.plot(SE_dl_hat[i, :], np.linspace(0, 1, num_layouts * num_users), label=f"{alg.upper()} (Approx)",
            color=colors[i], linewidth=3, linestyle='solid', alpha=0.6)
ax.set_xlabel(xlabel="Downlink Spectral Efficiency (bits/s/Hz)", fontsize=16)
ax.set_ylabel(ylabel="CDF", fontsize=16)
ax.legend()
plt.savefig(f"Figures/{setting}_CDF_dl_tightness", dpi=800)
plt.show()
test = []
