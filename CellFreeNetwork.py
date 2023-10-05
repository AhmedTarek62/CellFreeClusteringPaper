"""
This scripts implements a cell-free network class.
"""
from ProbDist import Uniform2D, Normal2D
from ChannelModels import CorrelatedRayleigh
from numpy.random import uniform, normal
from numpy.linalg import pinv, cholesky
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, zeros, ones, exp, sqrt, eye, conj, pi, dot, where, real, sum, mean, tile, min, diag, repeat, \
    triu, transpose, trace
from numpy.matlib import repmat
from scipy.linalg import block_diag
from ClusteringAlgorithms import emil_heuristic_dcc_pilot_assignment


class CellFreeNetwork:
    def __init__(
            self,
            num_aps: int,
            num_users: int,
            num_antennas: int,
            ap_dist: str,
            users_dist: str,
            coverage_area_len: int,
            channel_model: str,
            block_len: int,
            clustering_alg: str,
            pilot_len: int,
            pilot_alloc_alg: str,
            pilot_power_control_alg: str,
            uplink_power_control_alg: str,
            user_max_power: float,
            downlink_power_alloc_alg: str,
            ap_max_power: float,
            uplink_noise_power: float,
            downlink_noise_power: float
    ):
        """
        Cell-Free Network in a square coverage area
        """
        self.num_aps = num_aps
        self.num_users = num_users
        self.num_antennas = num_antennas
        self.coverage_area_len = coverage_area_len
        self.pilot_len = pilot_len
        self.block_len = block_len
        self.clustering_alg = clustering_alg
        self.pilot_alloc_alg = pilot_alloc_alg
        self.pilot_power_control_alg = pilot_power_control_alg
        self.uplink_power_control_alg = uplink_power_control_alg
        self.downlink_power_alloc_alg = downlink_power_alloc_alg
        self.sigma_uplink = sqrt((10 ** (uplink_noise_power / 10)) / 1000)
        self.sigma_downlink = sqrt((10 ** (downlink_noise_power / 10)) / 1000)
        self.user_max_power = user_max_power
        self.ap_max_power = ap_max_power
        self.pilots = self.generate_pilots()

        if channel_model == "Correlated Rayleigh":
            self.channel_model = CorrelatedRayleigh()

        if ap_dist == "Uniform":
            self.ap_dist = Uniform2D(coverage_area_len, coverage_area_len)
        else:
            raise NotImplementedError

        if users_dist == "Uniform":
            self.users_dist = Uniform2D(coverage_area_len, coverage_area_len)
        elif users_dist == "Normal":
            self.users_dist = Normal2D(coverage_area_len, coverage_area_len, 0, 65)
        else:
            raise NotImplementedError

        self.aps_loc = None
        self.users_loc = None
        self.user_ap_distance = None
        self.user_user_distance = None
        self.pilot_alloc_vec = None
        self.pilots = None
        self.pilot_power_vec = None
        self.uplink_power_vec = None
        self.downlink_power_vec = None

        # corresponds to canonical cell-free (no clustering)
        self.clusters = ones((num_users, num_aps))
        self.clusters_diag = zeros((num_users, num_aps, num_antennas, num_antennas))
        self.clusters_block_diag = zeros((num_users, num_antennas * num_aps, num_antennas * num_aps))
        for user in range(num_users):
            for AP in range(num_aps):
                self.clusters_diag[user, AP] = eye(num_antennas)
            self.clusters_block_diag[user] = block_diag(*(self.clusters_diag[user, ap] for ap in range(num_aps)))

    def generate_aps_placement(self):
        aps_loc = array([self.ap_dist.sample() for _ in range(self.num_aps)]).T
        return aps_loc

    def generate_users_placement(self):
        users_loc = array([self.users_dist.sample() for _ in range(self.num_users)]).T
        return users_loc

    def compute_wrap_around_locs(self, locs):
        coverage_area_len = self.coverage_area_len
        wrap_around_locs = array([
            [0,0],
            [coverage_area_len, 0],
            [0, coverage_area_len],
            [coverage_area_len, coverage_area_len],
            [coverage_area_len, -coverage_area_len],
            [-coverage_area_len, coverage_area_len],
            [-coverage_area_len, -coverage_area_len],
            [-coverage_area_len, 0],
            [0, -coverage_area_len]
        ]).T.reshape((2, 1, 9))
        locs_mat = tile(locs.reshape((locs.shape[0], locs.shape[1], 1)),(1, 1, 9))
        return locs_mat + wrap_around_locs

    def compute_users_aps_wrap_around_distances(self):
        num_users = self.num_users
        users_loc = self.users_loc
        aps_loc = self.aps_loc
        num_aps = self.num_aps
        user_ap_distance = zeros((num_users, num_aps))
        aps_wrapped_loc = self.compute_wrap_around_locs(aps_loc)
        for user in range(num_users):
            user_loc = tile(users_loc[:,user].reshape((2, 1, 1)), (1, num_aps, 9))
            user_ap_distance[user] = sqrt(10**2 + min(sum((aps_wrapped_loc - user_loc)**2, axis=0), axis=1))
        return user_ap_distance

    def compute_user_user_wrap_around_distance(self):
        users_loc = self.users_loc
        num_users = self.num_users
        users_wrapped_loc = self.compute_wrap_around_locs(users_loc)
        user_user_distance = zeros((num_users, num_users))
        for user in range(num_users):
            user_loc = tile(users_loc[:,user].reshape((2, 1, 1)), (1, num_users, 9))
            user_user_distance[user, :] = sqrt(min(sum((users_wrapped_loc - user_loc)**2, axis=0), axis=1))
        return user_user_distance


    def generate_snapshot(self):
        self.aps_loc = self.generate_aps_placement()
        self.users_loc = self.generate_users_placement()
        self.user_user_distance = self.compute_user_user_wrap_around_distance()
        self.user_ap_distance = self.compute_users_aps_wrap_around_distances()
        self.channel_model.compute_large_scale_info(self.num_antennas, self.user_ap_distance, self.user_user_distance)
        if self.clustering_alg == "emil":
            cluster_mat, pilot_alloc_vec = emil_heuristic_dcc_pilot_assignment(self.channel_model.path_loss_shadowing,
                                                                               self.pilot_len)
            self.set_clusters(cluster_mat)
            self.set_pilot_alloc(pilot_alloc_vec)
        else:
            _, pilot_alloc_vec = emil_heuristic_dcc_pilot_assignment(self.channel_model.path_loss_shadowing,
                                                                               self.pilot_len)
            self.set_clusters(ones((self.num_users, self.num_aps)))
            self.set_pilot_alloc(pilot_alloc_vec)
        self.pilot_power_vec = self.uplink_power_control()
        self.uplink_power_vec = self.uplink_power_control()
        self.channel_model.compute_error_sig_corr(self.pilot_alloc_vec, self.pilot_power_vec, self.pilot_len)

    def set_snapshot(self,
                     aps_loc: np.ndarray,
                     users_loc: np.ndarray,
                     user_user_distance: np.ndarray = None,
                     user_ap_distance: np.ndarray = None,
                     path_loss_shadowing: np.ndarray = None,
                     user_ap_corr: np.ndarray = None,
                     user_ap_corr_chol: np.ndarray = None,
                     ):
        num_users = users_loc.shape[1]
        num_aps = aps_loc.shape[1]
        num_antennas = self.num_antennas
        self.aps_loc = aps_loc
        self.users_loc = users_loc

        if user_user_distance is None:
            self.user_user_distance = self.compute_user_user_wrap_around_distance()
        else:
            self.user_user_distance = user_user_distance
        if user_ap_distance is None:
            self.user_ap_distance = self.compute_users_aps_wrap_around_distances()
        else:
            self.user_ap_distance = user_ap_distance

        if path_loss_shadowing is None:
            self.channel_model.compute_large_scale_info(num_antennas, self.user_ap_distance, self.user_user_distance)
            return

        if user_ap_corr_chol is None and user_ap_corr is not None:
            user_ap_corr_chol = [[cholesky(user_ap_corr[user, AP]) for AP in range(num_aps)]
                                  for user in range(num_users)]

        self.channel_model.set_large_scale_info(
            self.num_antennas,
            self.user_ap_distance,
            self.user_user_distance,
            path_loss_shadowing,
            user_ap_corr,
            user_ap_corr_chol
        )
        self.pilot_power_vec = self.uplink_power_control()
        self.uplink_power_vec = self.uplink_power_control()
        self.channel_model.compute_error_sig_corr(self.pilot_alloc_vec, self.pilot_power_vec, self.pilot_len)

    def plot_network(self):
        aps_loc = self.aps_loc
        users_loc = self.users_loc
        ax = plt.subplot(111)
        ax.scatter(aps_loc[0, :], aps_loc[1, :], c='r', marker='*', label='APs')
        ax.scatter(users_loc[0, :], users_loc[1, :], c='b', label='UEs')
        ax.legend(loc=(0.32, 1.02), ncol=2)
        for ap in range(self.num_aps):
            ax.annotate(f"AP {ap}", (aps_loc[0, ap], aps_loc[1, ap]))
        for user in range(self.num_users):
            ax.annotate(f"UE {user}", (users_loc[0, user], users_loc[1, user]))
        ax.set_xlim(-self.coverage_area_len/2, self.coverage_area_len/2)
        ax.set_ylim(-self.coverage_area_len/2, self.coverage_area_len/2)
        plt.show()


    def set_clusters(self,
                     clusters: np.ndarray):
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_diag = zeros((num_users, num_aps, num_antennas, num_antennas))
        clusters_block_diag = zeros((num_users, num_antennas * num_aps, num_antennas * num_aps))
        for user in range(num_users):
            for ap in range(num_aps):
                if clusters[user, ap]:
                    clusters_diag[user, ap] = eye(num_antennas)
            clusters_block_diag[user, :, :] = block_diag(*(clusters_diag[user, AP] for AP in range(num_aps)))
        self.clusters_diag = clusters_diag
        self.clusters_block_diag = clusters_block_diag
        self.clusters = clusters

    def set_pilot_alloc(self, pilot_alloc_vec: np.ndarray):
        self.pilot_alloc_vec = pilot_alloc_vec

    def allocate_pilots(self, pilot_alloc_alg: str):
        # generate pilots
        pilot_len = self.pilot_len
        pilots = np.ones((pilot_len, pilot_len), dtype=complex)
        omega_tau_pilots = exp(-1j * 2 * pi / pilot_len)
        for i in range(1, pilot_len):
            for j in range(1, pilot_len):
                pilots[i, j] = omega_tau_pilots ** (i * j)

        # allocate pilots
        num_users = self.num_users
        pilot_alloc = -ones((num_users,))
        if pilot_alloc_alg == 'random':
            pilot_alloc = np.round(uniform(0, pilot_len - 1, (num_users,)))
        self.pilot_alloc_vec = pilot_alloc
        self.pilots = pilots

    def generate_pilots(self):
        # generate pilots
        pilot_len = self.pilot_len
        pilots = np.ones((pilot_len, pilot_len), dtype=complex)
        omega_tau_pilots = exp(-1j * 2 * pi / pilot_len)
        for i in range(1, pilot_len):
            for j in range(1, pilot_len):
                pilots[i, j] = omega_tau_pilots ** (i * j)
        return pilots

    def reshape_combiners(self,
                          combiners: np.ndarray):
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        reshaped_combiners = zeros((num_users, num_aps, num_antennas), dtype=complex)
        for user in range(num_users):
            for ap in range(num_aps):
                reshaped_combiners[user, ap] = combiners[user, ap * num_antennas: (ap + 1) * num_antennas]
        return reshaped_combiners

    def uplink_power_control(self):
        return self.user_max_power * ones((self.num_users,))

    def downlink_power_alloc_fractional_centralized_conservative(self):
        beta = 10 ** (self.channel_model.path_loss_shadowing/10)
        clusters = self.clusters
        rho_max = self.ap_max_power
        num_users = self.num_users
        v = -1
        power_vec = np.zeros((num_users,))
        for user in range(num_users):
            user_cluster = where(clusters[user,:]==1)[0]
            numerator = sum(beta[user, user_cluster]) ** v
            denominator = 0
            for ap in user_cluster:
                candidate = 0
                ap_cluster = where(clusters[:,ap]==1)[0]
                for other_user in ap_cluster:
                    other_user_cluster = where(clusters[other_user,:]==1)[0]
                    candidate += sum(beta[other_user, other_user_cluster]) ** v
                if candidate > denominator:
                    denominator = candidate
            power_vec[user] = rho_max * numerator/denominator
        self.downlink_power_vec = power_vec
        return power_vec

    def downlink_power_alloc_fractional_centralized(self, precoders):
        p_max = self.ap_max_power
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        power_vec = zeros((num_users,))
        beta_vec = 10 ** (self.channel_model.path_loss_shadowing / 10)

        # calculate omega
        omega = -ones((num_users,))
        for user in range(num_users):
            for ap in range(num_aps):
                w_kl_all_frames = precoders[:, user, ap * num_antennas:(ap+1) * num_antennas]
                w_kl_norm = real(mean(sum(conj(w_kl_all_frames) * w_kl_all_frames, axis=1)))
                if w_kl_norm > omega[user]:
                    omega[user] = w_kl_norm

        for user in range(num_users):
            if omega[user] == 0:
                power_vec[user] = 0
                continue
            user_cluster = where(clusters[user, :] == 1)[0]
            numerator = (1 / sqrt(omega[user])) / (sqrt(sum([beta_vec[user, ap] for ap in user_cluster])))
            denominator = -np.inf
            for ap in user_cluster:
                ap_cluster = where(clusters[:, ap] == 1)[0]
                candidate_denominator = 0
                for other_user in ap_cluster:
                    candidate_denominator += \
                        sqrt(omega[other_user]) / \
                        sqrt(sum([beta_vec[other_user, other_ap] for other_ap in range(num_aps)
                                if clusters[other_user, other_ap]]))
                if candidate_denominator > denominator:
                    denominator = candidate_denominator
            power_vec[user] = p_max * numerator / denominator

        self.downlink_power_vec = power_vec
        return power_vec

    def downlink_power_alloc_fractional_distributed(self):
        p_max = self.ap_max_power
        num_users = self.num_users
        num_aps = self.num_aps
        beta_vec = 10 ** (self.channel_model.path_loss_shadowing / 10)
        clusters = self.clusters
        power_vec = zeros((num_users, num_aps))
        for user in range(num_users):
            for ap in range(num_aps):
                if clusters[user, ap]:
                    power_vec[user, ap] = \
                        p_max * sqrt(beta_vec[user, ap]) \
                        / np.sum([sqrt(beta_vec[other_user, ap]) for other_user in range(num_users)
                                  if clusters[other_user, ap]])
        return power_vec

    def generate_channel_realizations(self, num_frames):
        # local object variables
        num_aps = self.num_aps
        num_users = self.num_users
        num_antennas = self.num_antennas


        # simulate one layout for a designated number of frames
        channels = self.channel_model.sample(num_frames)
        channel_estimates = channels - self.channel_model.generate_channel_error(num_frames)
        collective_channels = channels.reshape((num_frames, num_users, num_aps*num_antennas))
        collective_channel_estimates = channel_estimates.reshape((num_frames, num_users, num_aps*num_antennas))

        return collective_channels, collective_channel_estimates, channels, channel_estimates

    def simulate_downlink_centralized(self,
                                      combining_alg: str,
                                      channels: np.ndarray,
                                      channel_estimates: np.ndarray,
                                      combiners_only: bool = False,
                                      conservative: bool = False):
        # local object variables
        combining_alg = combining_alg.lower()
        num_aps = self.num_aps
        num_users = self.num_users
        num_antennas = self.num_antennas
        num_frames = channels.shape[0]

        if combining_alg == 'mmse':
            precoders = self.compute_combiners_centralized_mmse(channel_estimates, normalize=True)
        elif combining_alg == 'mr':
            precoders = self.compute_combiners_centralized_mrc(channel_estimates, normalize=True)
        elif combining_alg == 'pmmse':
            precoders = self.compute_combiners_centralized_pmmse(channel_estimates, normalize=True)
        elif combining_alg == 'przf':
            precoders = self.compute_combiners_centralized_przf(channel_estimates, normalize=True)
        else:
            raise NotImplementedError

        if combiners_only:
            return precoders

        if conservative:
            downlink_power_vec = self.downlink_power_alloc_fractional_centralized_conservative().reshape((1, num_users, 1))
            downlink_power_mat = np.tile(downlink_power_vec, (num_frames, 1, num_antennas * num_aps))
        else:
            downlink_power_vec = self.downlink_power_alloc_fractional_centralized(precoders).reshape((1, num_users, 1))
            downlink_power_mat = np.tile(downlink_power_vec, (num_frames, 1, num_antennas*num_aps))

        return sqrt(downlink_power_mat) * precoders

    def simulate_downlink_distributed(self,
                                      combining_alg: str,
                                      channels: np.ndarray,
                                      channel_estimates: np.ndarray,
                                      combiners_only: bool = False):
        # local object variables
        num_aps = self.num_aps
        num_users = self.num_users
        num_antennas = self.num_antennas
        num_frames = channels.shape[0]
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)
        downlink_power_vec = self.downlink_power_alloc_fractional_distributed().reshape((1, num_users, num_aps, 1))
        downlink_power_mat = np.tile(downlink_power_vec, (num_frames, 1, 1, num_antennas))

        if combining_alg.lower() == 'mmse':
            combiners =self.compute_combiners_distributed_mmse(channel_estimates, normalize=True)
        elif combining_alg.lower() == 'pmmse':
            combiners = self.compute_combiners_distributed_pmmse(channel_estimates, normalize=True)
        elif combining_alg.lower() == 'mr':
            combiners = self.compute_combiners_distributed_mrc(channel_estimates, normalize=True)
        elif combining_alg.lower() == 'przf':
            combiners = self.compute_combiners_distributed_przf(channel_estimates, normalize=True)
        if combiners_only:
            return combiners

        return sqrt(downlink_power_mat) * combiners, self.compute_ap_weights()

    def compute_downlink_SE_centralized(self, channels, precoders, alpha=False, sinr=False):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interference_norm = zeros((num_users, num_users), dtype=complex)

        for user in range(num_users):
            serving_aps = where(clusters[user]==1)[0]
            if not serving_aps.size:
                continue
            serving_aps_extended = \
                repeat(serving_aps, num_antennas) * num_antennas \
                + repmat(array([0, 1, 2, 3]), num_aps, 1).reshape((-1,))[:serving_aps.shape[0]*num_antennas]
            for frame in range(num_frames):
                w_k = precoders[frame, user].reshape((-1, 1))
                h_k = channels[frame, user].reshape((-1, 1))
                sig[user] += (conj(h_k[serving_aps_extended]).T @ w_k[serving_aps_extended])[0, 0]
                W = precoders[frame].T
                interference = (conj(h_k).T @ W).reshape((-1,))
                interference_norm[user] += conj(interference) * interference
        sig /= num_frames
        interference_norm /= num_frames
        interference_norm = sum(interference_norm, axis=1)
        sig_norm = conj(sig) * sig
        if alpha:
            return real(sig_norm), real(interference_norm - sig_norm)

        average_sinr = real(sig_norm / (interference_norm - sig_norm + 1))

        if sinr:
            return average_sinr

        return prelog * np.log2(1 + average_sinr)

    def compute_downlink_SE_centralized_iterative(self, channels, combiners):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        clusters_block_diag = self.clusters_block_diag
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interference_norm = zeros((num_users, num_users), dtype=complex)

        for frame in range(num_frames):
            for user in range(num_users):
                w_k = combiners[frame, user].reshape((-1, 1))
                D_k = clusters_block_diag[user]
                h_k = channels[frame, user].reshape((-1, 1))
                sig[user] += (conj(h_k).T @ D_k @ w_k)[0, 0]
                for other_user in range(num_users):
                    w_i = combiners[frame, other_user].reshape((-1, 1))
                    D_i = clusters_block_diag[other_user]
                    q_i = (conj(h_k).T @ D_i @ w_i)[0, 0]
                    interference_norm[user, other_user] += conj(q_i) * q_i
        sig /= num_frames
        interference_norm /= num_frames
        interference_norm = np.sum(interference_norm, axis=1)
        sig_norm = conj(sig) * sig
        average_sinr = real(sig_norm / (interference_norm - sig_norm + 1))
        return prelog * np.log2(1 + average_sinr)

    def compute_downlink_SE_distributed(self, channels: np.ndarray, combiners: np.ndarray):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        clusters_diag = self.clusters_diag
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        sig = zeros((num_users, num_aps), dtype=complex)
        interference_norm = zeros((num_users,), dtype=complex)

        for frame in range(num_frames):
            for user in range(num_users):
                for ap in range(num_aps):
                    h_kl = channels[frame, user, ap].reshape((-1, 1))
                    D_kl = clusters_diag[user, ap]
                    w_kl = combiners[frame, user, ap].reshape((-1, 1))
                    sig[user, ap] += (conj(h_kl).T @ D_kl @ w_kl)[0, 0]

                for other_user in range(num_users):
                    interference = 0
                    for other_ap in range(num_aps):
                        h_kl = channels[frame, user, other_ap].reshape((-1, 1))
                        D_il = clusters_diag[other_user, other_ap]
                        w_il = combiners[frame, other_user, other_ap].reshape((-1, 1))
                        interference += (conj(h_kl).T @ D_il @ w_il)[0, 0]
                    interference_norm[user] += conj(interference) * interference

        sig /= num_frames
        interference_norm /= num_frames

        sig_norm = conj(sum(sig, axis=1)) * sum(sig, axis=1)
        average_sinr = real(sig_norm / (interference_norm - sig_norm + 1))

        return prelog * np.log2(1 + average_sinr)

    def simulate_uplink_centralized(self,
                                    combining_alg: str,
                                    channels: np.ndarray,
                                    channel_estimates: np.ndarray):
        """
        This function simulates centralized uplink transmission for a number of frames using a specific combining
        algorithm
        :param combining_alg: string specifying combining algorithm of choice
        :param channels
        :param channel_estimates
        :return: combiners
        """
        # local object variables
        combining_alg = combining_alg.lower()

        if combining_alg == 'mmse':
            combiners = self.compute_combiners_centralized_mmse(channel_estimates)
        elif combining_alg == 'mr':
            combiners = self.compute_combiners_centralized_mrc(channel_estimates)
        elif combining_alg == 'pmmse':
            combiners = self.compute_combiners_centralized_pmmse(channel_estimates)
        elif combining_alg == 'przf':
            combiners = self.compute_combiners_centralized_przf(channel_estimates)
        else:
            raise NotImplementedError
        return combiners

    def simulate_uplink_distributed(self,
                                    combining_alg: str,
                                    channels: np.ndarray,
                                    channel_estimates: np.ndarray):
        # local object variables
        num_aps = self.num_aps
        num_users = self.num_users
        num_antennas = self.num_antennas
        num_frames = channels.shape[0]
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)

        if combining_alg.lower() == 'mmse':
            combiners = self.compute_combiners_distributed_mmse(channel_estimates)
        elif combining_alg.lower() == 'pmmse':
            combiners = self.compute_combiners_distributed_pmmse(channel_estimates)
        elif combining_alg.lower() == 'mr':
            combiners = self.compute_combiners_distributed_mrc(channel_estimates)
        elif combining_alg.lower() == 'przf':
            combiners = self.compute_combiners_distributed_przf(channel_estimates)
        return combiners, self.compute_ap_weights()

    def compute_combiners_centralized_mrc(self, channel_estimates: np.ndarray, normalize=False):
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_block_diag = self.clusters_block_diag
        combiners = zeros((num_frames, num_users, num_aps*num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(self.num_users):
                D_k = clusters_block_diag[user]
                h_hat_k = channel_estimates[frame, user]
                combiners[frame, user] = (D_k @ h_hat_k).reshape((-1,))
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1, -1, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps*num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_mrc(self, channel_estimates: np.ndarray, normalize=False):
        clusters_diag = self.clusters_diag
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        num_frames = channel_estimates.shape[0]
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(self.num_users):
                for ap in range(self.num_aps):
                    D_kl = clusters_diag[user, ap]
                    h_hat_kl = channel_estimates[frame, user, ap]
                    combiners[frame, user, ap] = (D_kl @ h_hat_kl).reshape((-1,))
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler==0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_mmse(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_block_diag = self.clusters_block_diag
        power_vec = self.uplink_power_vec
        power_mat = diag(power_vec)
        C_all = sum(tile(power_vec.reshape((-1,1,1)),(1, num_aps*num_antennas, num_aps*num_antennas))
                    * self.channel_model.block_user_error_corr, axis=0)
        eyeLN = eye(num_antennas * num_aps, dtype=complex)
        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)

        for user in range(num_users):
            D_k = clusters_block_diag[user]
            p_k = power_vec[user]
            for frame in range(num_frames):
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                H_hat = channel_estimates[frame].T
                inv_mat = D_k @ ((H_hat @ power_mat @ conj(H_hat).T) + C_all) @ D_k + eyeLN
                combiners[frame, user] = (p_k * pinv(inv_mat) @ D_k @ h_hat_k).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1,-1, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps*num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_pmmse(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_block_diag = self.clusters_block_diag
        power_vec = self.uplink_power_vec
        eyeLN = eye(num_antennas * num_aps, dtype=complex)
        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)

        for user in range(num_users):
            s_k = array([other_user for other_user in range(num_users)
                         if sum(clusters[user] * clusters[other_user] >= 1)])
            if not s_k.size:
                continue
            D_k = clusters_block_diag[user]
            p_k = power_vec[user]
            power_mat = diag(power_vec[s_k])
            C_partial = sum(tile(power_vec[s_k].reshape((-1, 1, 1)), (1, num_aps * num_antennas, num_aps * num_antennas))
                        * self.channel_model.block_user_error_corr[s_k], axis=0)
            for frame in range(num_frames):
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                H_hat = channel_estimates[frame, s_k].T
                inv_mat = D_k @ ((H_hat @ power_mat @ conj(H_hat).T) + C_partial) @ D_k + eyeLN
                combiners[frame, user] = (p_k * pinv(inv_mat) @ D_k @ h_hat_k).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1, -1, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps * num_antennas))
            combiners = combiners / sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_przf(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_block_diag = self.clusters_block_diag
        power_vec = self.uplink_power_vec
        eyeLN = eye(num_antennas * num_aps, dtype=complex)
        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)

        for user in range(num_users):
            s_k = array([other_user for other_user in range(num_users)
                         if sum(clusters[user] * clusters[other_user] >= 1)])
            if not s_k.size:
                continue
            D_k = clusters_block_diag[user]
            p_k = power_vec[user]
            power_mat = diag(power_vec[s_k])
            for frame in range(num_frames):
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                H_hat = channel_estimates[frame, s_k].T
                inv_mat = D_k @ (H_hat @ power_mat @ conj(H_hat).T) @ D_k + eyeLN
                combiners[frame, user] = (p_k * pinv(inv_mat) @ D_k @ h_hat_k).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1, -1, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps * num_antennas))
            combiners = combiners / sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_mmse(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_diag = self.clusters_diag
        user_ap_error_corr = self.channel_model.user_ap_error_corr
        power_vec = self.uplink_power_vec
        eyeN = eye(num_antennas, dtype=complex)
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)

        for user in range(num_users):
            p_k = power_vec[user]
            for ap in range(num_aps):
                D_kl = clusters_diag[user, ap]
                C_ap_l = sum(tile(power_vec.reshape((-1,1,1)), (1, num_antennas, num_antennas))
                             * user_ap_error_corr[:, ap], axis=0)
                for frame in range(num_frames):
                    h_hat_kl = channel_estimates[frame, user, ap].reshape((-1, 1))
                    H_hat = channel_estimates[frame, :, ap].T
                    inv_mat = H_hat @ diag(power_vec) @ conj(H_hat).T + C_ap_l + eyeN
                    combiners[frame, user, ap] = p_k * (pinv(inv_mat) @ D_kl @ h_hat_kl).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler==0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_pmmse(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_diag = self.clusters_diag
        user_ap_error_corr = self.channel_model.user_ap_error_corr
        power_vec = self.uplink_power_vec
        eyeN = eye(num_antennas, dtype=complex)
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)

        for user in range(num_users):
            p_k = power_vec[user]
            for ap in range(num_aps):
                s_l = where(clusters[:, ap] == 1)[0]
                D_kl = clusters_diag[user, ap]
                C_ap_l = sum(tile(power_vec[s_l].reshape((-1, 1, 1)), (1, num_antennas, num_antennas))
                             * user_ap_error_corr[s_l, ap], axis=0)
                for frame in range(num_frames):
                    h_hat_kl = channel_estimates[frame, user, ap].reshape((-1, 1))
                    H_hat = channel_estimates[frame, s_l, ap].T
                    inv_mat = H_hat @ diag(power_vec[s_l]) @ conj(H_hat).T + C_ap_l + eyeN
                    combiners[frame, user, ap] = p_k * (pinv(inv_mat) @ D_kl @ h_hat_kl).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners / sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_przf(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_diag = self.clusters_diag
        power_vec = self.uplink_power_vec
        eyeN = eye(num_antennas, dtype=complex)
        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)

        for user in range(num_users):
            p_k = power_vec[user]
            for ap in range(num_aps):
                s_l = where(clusters[:, ap] == 1)[0]
                D_kl = clusters_diag[user, ap]
                for frame in range(num_frames):
                    h_hat_kl = channel_estimates[frame, user, ap].reshape((-1, 1))
                    H_hat = channel_estimates[frame, s_l, ap].T
                    inv_mat = H_hat @ diag(power_vec[s_l]) @ conj(H_hat).T + eyeN
                    combiners[frame, user, ap] = p_k * (pinv(inv_mat) @ D_kl @ h_hat_kl).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler == 0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners / sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_mmse_iterative(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_block_diag = self.clusters_block_diag
        uplink_power_vec = self.uplink_power_vec

        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(num_users):
                D_k = clusters_block_diag[user]
                p_k = uplink_power_vec[user]
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                intermediate_mat = eye(num_antennas * num_aps, dtype=complex)
                for other_user in range(num_users):
                    h_hat_i = channel_estimates[frame, other_user].reshape((-1, 1))
                    C_i = self.channel_model.get_block_diag_error_corr(other_user)
                    p_i = uplink_power_vec[other_user]
                    intermediate_mat += p_i * (D_k @ (h_hat_i @ conj(h_hat_i).T + C_i) @ D_k)
                combiners[frame, user] = (p_k * (pinv(intermediate_mat) @ D_k @ h_hat_k)).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1,-1, 1)))
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps*num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_pmmse_iterative(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_block_diag = self.clusters_block_diag
        uplink_power_vec = self.uplink_power_vec

        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(num_users):
                user_cluster = clusters[user]
                D_k = clusters_block_diag[user]
                p_k = uplink_power_vec[user]
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                intermediate_mat = eye(num_antennas * num_aps, dtype=complex)
                for other_user in range(num_users):
                    other_user_cluster = clusters[other_user]
                    if sum(other_user_cluster * user_cluster) == 0:
                        continue
                    h_hat_i = channel_estimates[frame, other_user].reshape((-1, 1))
                    C_i = self.channel_model.get_block_diag_error_corr(other_user)
                    p_i = uplink_power_vec[other_user]
                    intermediate_mat += p_i * (D_k @ (h_hat_i @ conj(h_hat_i).T + C_i) @ D_k)
                combiners[frame, user] = (p_k * (pinv(intermediate_mat) @ D_k @ h_hat_k)).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1,-1, 1)))
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps*num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_centralized_przf_iterative(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_block_diag = self.clusters_block_diag
        uplink_power_vec = self.uplink_power_vec

        # go through frames and compute combiners
        combiners = zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(num_users):
                user_cluster = clusters[user]
                D_k = clusters_block_diag[user]
                p_k = uplink_power_vec[user]
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                intermediate_mat = eye(num_antennas * num_aps, dtype=complex)
                for other_user in range(num_users):
                    other_user_cluster = clusters[other_user]
                    if sum(other_user_cluster * user_cluster) == 0:
                        continue
                    h_hat_i = channel_estimates[frame, other_user].reshape((-1, 1))
                    p_i = uplink_power_vec[other_user]
                    intermediate_mat += p_i * (D_k @ h_hat_i @ conj(h_hat_i).T @ D_k)
                combiners[frame, user] = (p_k * (pinv(intermediate_mat) @ D_k @ h_hat_k)).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=2), axis=0).reshape((1,-1, 1)))
            scaler_mat = np.tile(scaler, (num_frames, 1, num_aps*num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_mmse_iterative(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters_diag = self.clusters_diag
        uplink_power_vec = self.uplink_power_vec

        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(num_users):
                p_k = uplink_power_vec[user]
                for ap in range(num_aps):
                    D_kl = clusters_diag[user, ap]
                    h_hat_kl = channel_estimates[frame, user, ap].reshape((-1, 1))
                    intermediate_mat = eye(num_antennas, dtype=complex)
                    for other_user in range(num_users):
                        p_i = uplink_power_vec[other_user]
                        h_hat_il = channel_estimates[frame, other_user, ap, :].reshape((-1, 1))
                        C_il = self.channel_model.user_ap_error_corr[other_user, ap]
                        intermediate_mat += p_i * (h_hat_il @ conj(h_hat_il).T + C_il)
                    combiners[frame, user, ap] = (p_k * (pinv(intermediate_mat) @ D_kl @ h_hat_kl)).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler==0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return combiners

    def compute_combiners_distributed_pmmse_iterative(self, channel_estimates: np.ndarray, normalize=False):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        clusters = self.clusters
        clusters_diag = self.clusters_diag
        uplink_power_vec = self.uplink_power_vec

        combiners = zeros((num_frames, num_users, num_aps, num_antennas), dtype=complex)
        for frame in range(num_frames):
            for user in range(num_users):
                p_k = uplink_power_vec[user]
                for ap in range(num_aps):
                    D_kl = clusters_diag[user, ap]
                    h_hat_kl = channel_estimates[frame, user, ap].reshape((-1, 1))
                    intermediate_mat = eye(num_antennas, dtype=complex)
                    for other_user in where(clusters[:, ap]==1)[0]:
                        p_i = uplink_power_vec[other_user]
                        h_hat_il = channel_estimates[frame, other_user, ap, :].reshape((-1, 1))
                        C_il = self.channel_model.user_ap_error_corr[other_user, ap]
                        intermediate_mat += p_i * (h_hat_il @ conj(h_hat_il).T + C_il)
                    combiners[frame, user, ap] = (p_k * (pinv(intermediate_mat) @ D_kl @ h_hat_kl)).reshape((-1,))
        # normalize if asked
        if normalize:
            scaler = real(mean(sum(conj(combiners) * combiners, axis=3), axis=0).reshape((1, num_users, num_aps, 1)))
            scaler[scaler==0] = 1
            scaler_mat = np.tile(scaler, (num_frames, 1, 1, num_antennas))
            combiners = combiners/sqrt(scaler_mat)
        return

    def compute_ap_weights(self):
        # LSFD computation - no weighting for now
        num_users = self.num_users
        num_aps = self.num_aps
        clusters = self.clusters
        ap_weights = zeros((num_users, num_aps))
        for user in range(num_users):
            for ap in range(num_aps):
                if clusters[user, ap]:
                    ap_weights[user, ap] = 1
        return ap_weights

    def compute_uplink_SE_centralized(self, channel_estimates: np.ndarray, combiners: np.ndarray):
        # local definitions
        num_frames = channel_estimates.shape[0]
        clusters = self.clusters
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        power_vec = self.uplink_power_vec
        block_user_error_corr = self.channel_model.block_user_error_corr
        prelog = 1 - self.pilot_len / self.block_len

        # go through all frames (realizations) and compute the expectations
        SE = zeros((num_users,))
        for user in range(num_users):
            serving_aps = where(clusters[user] == 1)[0]
            if not serving_aps.size:
                continue
            serving_aps_extended = \
                repeat(serving_aps, num_antennas) * num_antennas \
                + repmat(array([0, 1, 2, 3]), num_aps, 1).reshape((-1,))[:serving_aps.shape[0]*num_antennas]
            p_k = power_vec[user]
            Z_k = sum(tile(
                power_vec.reshape((-1,1,1)),(1, serving_aps.shape[0]*num_antennas, serving_aps.shape[0]*num_antennas))
                      * block_user_error_corr[:, serving_aps_extended, :][:, :, serving_aps_extended], axis=0)

            for frame in range(num_frames):
                v_k = combiners[frame, user, serving_aps_extended].reshape((-1, 1))
                h_hat_k = channel_estimates[frame, user, serving_aps_extended].reshape((-1, 1))
                H_hat = channel_estimates[frame, :, serving_aps_extended]
                H_hat[:, user] = 0

                sig = (conj(v_k).T @ h_hat_k)[0, 0]
                sig_norm = p_k * conj(sig) * sig
                interference = (conj(v_k).T @ (H_hat @ diag(sqrt(power_vec)))).reshape(-1,)
                interference_norm = (conj(v_k).T @ Z_k @ v_k + conj(v_k).T @ v_k)[0, 0] +\
                                    sum(conj(interference) * interference)

                SINR_user_frame = real(sig_norm / interference_norm)
                SE[user] += np.log2(1 + SINR_user_frame)

        return prelog * SE / num_frames

    def compute_uplink_SE_centralized_iterative(self, channel_estimates: np.ndarray, combiners: np.ndarray):
        # local definitions
        num_frames = channel_estimates.shape[0]
        num_users = self.num_users
        uplink_power_vec = self.uplink_power_vec
        clusters_block_diag = self.clusters_block_diag
        prelog = 1 - self.pilot_len / self.block_len

        # go through all frames (realizations) and compute the expectations
        SE = zeros((num_users,))
        for user in range(num_users):

            p_k = uplink_power_vec[user]
            D_k = clusters_block_diag[user]
            Z_k = p_k * (D_k @ self.channel_model.get_block_diag_error_corr(user) @ D_k)
            for other_user in range(num_users):
                if other_user != user:
                    p_i = uplink_power_vec[other_user]
                    Z_k += p_i * (D_k @ self.channel_model.get_block_diag_error_corr(other_user) @ D_k)

            for frame in range(num_frames):

                v_k = combiners[frame, user].reshape((-1, 1))
                h_hat_k = channel_estimates[frame, user].reshape((-1, 1))
                sig = (sqrt(p_k) * conj(v_k).T @ D_k @ h_hat_k)[0, 0]
                sig_norm = conj(sig) * sig
                interference_norm = ((conj(v_k).T @ Z_k @ v_k) + conj(D_k @ v_k).T @ (D_k @ v_k))[0, 0]

                for other_user in range(num_users):
                    if user != other_user:
                        p_i = uplink_power_vec[other_user]
                        h_hat_i = channel_estimates[frame, other_user].reshape((-1, 1))
                        interference = sqrt(p_i) * (conj(v_k).T @ D_k @ h_hat_i)[0, 0]
                        interference_norm += conj(interference) * interference
                SINR_user_frame = real(sig_norm/interference_norm)
                SE[user] += np.log2(1+SINR_user_frame)

        return prelog * SE/num_frames

    def compute_uplink_SE_centralized_uatf(self, channels: np.ndarray, combiners: np.ndarray, alpha=False):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        power_vec = self.uplink_power_vec
        clusters = self.clusters
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interference_norm = zeros((num_users, num_users), dtype=complex)
        noise_norm = zeros((num_users,), dtype=complex)

        # go through all frames (realizations) and compute the expectations

        for user in range(num_users):
            if np.sum(clusters[user]) == 0:
                noise_norm[user] = 1
                continue
            p_k = power_vec[user]
            serving_aps = where(clusters[user] == 1)[0]
            serving_aps_extended = \
                repeat(serving_aps, num_antennas) * num_antennas \
                + repmat(array([0, 1, 2, 3]), num_aps, 1).reshape((-1,))[:serving_aps.shape[0] * num_antennas]
            for frame in range(num_frames):
                h_k = channels[frame, user, serving_aps_extended].reshape((-1, 1))
                v_k = combiners[frame, user, serving_aps_extended].reshape((-1, 1))
                sig[user] += sqrt(p_k) * (conj(v_k).T @ h_k)[0, 0]
                noise_norm[user] += (conj(v_k).T @ v_k)[0, 0]

                H = channels[frame, :, serving_aps_extended]
                interference = (conj(v_k).T @ H @ diag(sqrt(power_vec))).reshape((-1,))
                interference_norm[user] += conj(interference) * interference

        sig /= num_frames
        interference_norm /= num_frames
        noise_norm /= num_frames
        sig_norm = sig * conj(sig)
        interference_norm = sum(interference_norm, axis=1)
        if alpha:
            return sig_norm, interference_norm
        # compute the spectral efficiency
        average_sinr = real(sig_norm/(interference_norm - sig_norm + noise_norm))

        return prelog * np.log2(1 + average_sinr)

    def compute_uplink_SE_centralized_uatf_iterative(self, channels: np.ndarray, combiners: np.ndarray):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        uplink_power_vec = self.uplink_power_vec
        clusters_block_diag = self.clusters_block_diag
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interference_norm = zeros((num_users,), dtype=complex)
        noise_norm = zeros((num_users,), dtype=complex)

        # go through all frames (realizations) and compute the expectations
        for frame in range(num_frames):
            for user in range(num_users):
                p_k = uplink_power_vec[user]
                h_k = channels[frame, user].reshape((-1, 1))
                v_k = combiners[frame, user].reshape((-1, 1))
                D_k = clusters_block_diag[user]
                sig[user] += sqrt(p_k) * (conj(v_k).T @ D_k @ h_k)[0, 0]
                v_k_eff = D_k @ v_k
                noise_norm[user] += (conj(v_k_eff).T @ v_k_eff)[0, 0]
                for other_user in range(num_users):
                    p_i = uplink_power_vec[other_user]
                    h_i = channels[frame, other_user].reshape((-1, 1))
                    q_i = (conj(v_k).T @ D_k @ h_i)[0, 0]
                    interference_norm[user] += p_i * conj(q_i) * q_i
        sig /= num_frames
        interference_norm /= num_frames
        noise_norm /= num_frames
        sig_norm = sig * conj(sig)
        # compute the spectral efficiency
        average_sinr = real(sig_norm/(interference_norm - sig_norm + noise_norm))

        return prelog * np.log2(1 + average_sinr)

    def compute_uplink_SE_distributed(self, channels: np.ndarray, combiners: np.ndarray, ap_weights=None):
        if ap_weights is None:
            ap_weights = self.compute_ap_weights()
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps
        uplink_power_vec = self.uplink_power_vec
        clusters_diag = self.clusters_diag
        prelog = 1 - self.pilot_len/self.block_len

        # SINR accumulation variables
        g = zeros((num_users, num_users, num_aps), dtype=complex)
        g_outer = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        F = zeros((num_users, num_aps, num_aps), dtype=complex)
        # go through all frames (realizations) and compute the expectations
        for frame in range(num_frames):
            for user in range(num_users):
                for ap in range(num_aps):
                    v_kl = combiners[frame, user, ap].reshape((-1, 1))
                    D_kl = clusters_diag[user, ap]
                    v_kl_eff = D_kl @ v_kl
                    F[user, ap, ap] += (conj(v_kl_eff).T @ v_kl_eff)[0, 0]
                    for other_user in range(num_users):
                        h_il = channels[frame, other_user, ap].reshape((-1, 1))
                        g[user, other_user, ap] += (conj(v_kl).T @ D_kl @ h_il)[0, 0]

                for other_user in range(num_users):
                    g_ki = zeros((num_aps, 1), dtype=complex)
                    for ap in range(num_aps):
                        v_kl = combiners[frame, user, ap].reshape((-1, 1))
                        D_kl = clusters_diag[user, ap]
                        h_il = channels[frame, other_user, ap].reshape((-1, 1))
                        g_ki[ap] = (conj(v_kl).T @ D_kl @ h_il)[0, 0]
                    g_outer[user, other_user] += g_ki @ conj(g_ki).T
        F /= num_frames
        g /= num_frames
        g_outer /= num_frames

        # compute the SE
        average_SE = zeros((num_users,))
        for user in range(num_users):
            p_k = uplink_power_vec[user]
            a_k = ap_weights[user].reshape((-1, 1))
            g_kk = g[user, user].reshape((-1, 1))
            sig = (sqrt(p_k) * conj(a_k).T @ g_kk)[0, 0]
            sig_norm = conj(sig) * sig
            interference_mat = F[user] - p_k * g_kk @ conj(g_kk).T
            for other_user in range(num_users):
                p_i = uplink_power_vec[other_user]
                interference_mat += p_i * g_outer[user, other_user]
            sinr = real(sig_norm/(conj(a_k).T @ interference_mat @ a_k)[0, 0])
            average_SE[user] = prelog * np.log2(1 + sinr)

        return average_SE

    def compute_clustering_optimization_model(self,
                                              channels: np.ndarray,
                                              collective_channels: np.ndarray,
                                              collective_channel_estimates: np.ndarray,
                                              combining_alg: str):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps

        # optimization matrices
        C_vh = zeros((num_users, num_aps), dtype=complex)
        C_wh = zeros((num_users, num_aps), dtype=complex)
        C_vh_squared = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        C_wh_squared = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        C_v = zeros((num_users, num_aps), dtype=complex)

        precoders = self.simulate_downlink_centralized(combining_alg, collective_channels, collective_channel_estimates,
                                                       combiners_only=True)
        combiners = self.simulate_uplink_centralized(combining_alg, collective_channels, collective_channel_estimates)
        # precoders = self.simulate_downlink_centralized(combining_alg, collective_channels, collective_channels,
        #                                                combiners_only=True)
        # combiners = self.simulate_uplink_centralized(combining_alg, collective_channels, collective_channels)

        for frame in range(num_frames):
            frame_channels = channels[frame]
            frame_combiners = self.reshape_combiners(combiners[frame])
            frame_precoders = self.reshape_combiners(precoders[frame])
            for user in range(num_users):

                # compute 1st order combiner/channel & precoder/channel correlation
                for ap in range(num_aps):
                    v_kn = frame_combiners[user, ap].reshape((-1, 1))
                    w_kn = frame_precoders[user, ap].reshape((-1, 1))
                    h_kn = frame_channels[user, ap].reshape((-1, 1))

                    C_vh[user, ap] += (conj(v_kn).T @ h_kn)[0, 0]
                    C_wh[user, ap] += (conj(h_kn).T @ w_kn)[0, 0]

                    C_v[user, ap] += (conj(v_kn).T @ v_kn)[0, 0]

                # compute 2nd order combiner/channel & precoder/channel correlation
                for other_user in range(num_users):
                    for ap in range(num_aps):
                        v_kn = frame_combiners[user, ap].reshape((-1, 1))
                        h_in = frame_channels[other_user, ap].reshape((-1, 1))

                        h_kn = frame_channels[user, ap].reshape((-1, 1))
                        w_in = frame_precoders[other_user, ap].reshape((-1, 1))
                        for other_ap in range(num_aps):
                            v_km = frame_combiners[user, other_ap].reshape((-1, 1))
                            h_im = frame_channels[other_user, other_ap].reshape((-1, 1))

                            h_km = frame_channels[user, other_ap].reshape((-1, 1))
                            w_im = frame_precoders[other_user, other_ap].reshape((-1, 1))
                            if ap <= other_ap:
                                C_vh_squared[user, other_user, ap, other_ap] += \
                                    (conj(conj(v_kn).T @ h_in) * (conj(v_km).T @ h_im))[0,0]
                                C_wh_squared[user, other_user, ap, other_ap] += \
                                    (conj(conj(h_kn).T @ w_in) * (conj(h_km).T @ w_im))[0,0]

                # compute 1st order combiner/combiner correlation


        # compute the expectations
        C_v /= num_frames
        C_vh /= num_frames
        C_wh /= num_frames
        C_vh_squared /= num_frames
        C_wh_squared /= num_frames

        # compute the lower triangle of the correlation matrices
        C_vh_squared = (C_vh_squared + transpose(conj(triu(C_vh_squared, 1)), (0, 1, 3, 2)))
        C_wh_squared = (C_wh_squared + transpose(conj(triu(C_wh_squared, 1)), (0, 1, 3, 2)))

        # form the optimization matrices
        T_ul = zeros((num_users, num_aps, num_aps), dtype=complex)
        T_dl = zeros((num_users, num_aps, num_aps), dtype=complex)
        Q_ul = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        Q_dl = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)

        for user in range(num_users):
            for ap in range(num_aps):
                for other_ap in range(num_aps):
                    T_ul[user, ap, other_ap] = (conj(C_vh[user, ap]) * C_vh[user, other_ap] +
                                                C_vh[user, ap] * conj(C_vh[user, other_ap]))/2
                    T_dl[user, ap, other_ap] = (conj(C_wh[user, ap]) * C_wh[user, other_ap] +
                                                C_wh[user, ap] * conj(C_wh[user, other_ap]))/2
                    for other_user in range(num_users):
                        Q_ul[user, other_user, ap, other_ap] = \
                            (C_vh_squared[user, other_user, ap, other_ap] + C_vh_squared[user, other_user, other_ap, ap])/2
                        Q_dl[user, other_user, ap, other_ap] = \
                            (C_wh_squared[user, other_user, ap, other_ap] + C_wh_squared[user, other_user, other_ap, ap])/2

        return Q_dl, Q_ul, T_ul, T_dl, C_v


    def compute_clustering_optimization_model_distributed(self,
                                              channels: np.ndarray,
                                              channel_estimates: np.ndarray,
                                              combining_alg: str):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        num_aps = self.num_aps

        # optimization matrices
        C_vh = zeros((num_users, num_aps), dtype=complex)
        C_wh = zeros((num_users, num_aps), dtype=complex)
        C_vh_squared = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        C_wh_squared = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        C_v = zeros((num_users, num_aps), dtype=complex)

        precoders = self.simulate_downlink_distributed(combining_alg, channels, channel_estimates, combiners_only=True)
        combiners, _ = self.simulate_uplink_distributed(combining_alg, channels, channel_estimates)

        for frame in range(num_frames):
            frame_channels = channels[frame]
            frame_combiners = combiners[frame]
            frame_precoders = precoders[frame]
            for user in range(num_users):

                # compute 1st order combiner/channel & precoder/channel correlation
                for ap in range(num_aps):
                    v_kn = frame_combiners[user, ap].reshape((-1, 1))
                    w_kn = frame_precoders[user, ap].reshape((-1, 1))
                    h_kn = frame_channels[user, ap].reshape((-1, 1))

                    C_vh[user, ap] += (conj(v_kn).T @ h_kn)[0, 0]
                    C_wh[user, ap] += (conj(h_kn).T @ w_kn)[0, 0]

                    C_v[user, ap] += (conj(v_kn).T @ v_kn)[0, 0]

                # compute 2nd order combiner/channel & precoder/channel correlation
                for other_user in range(num_users):
                    for ap in range(num_aps):
                        v_kn = frame_combiners[user, ap].reshape((-1, 1))
                        h_in = frame_channels[other_user, ap].reshape((-1, 1))

                        h_kn = frame_channels[user, ap].reshape((-1, 1))
                        w_in = frame_precoders[other_user, ap].reshape((-1, 1))
                        for other_ap in range(num_aps):
                            v_km = frame_combiners[user, other_ap].reshape((-1, 1))
                            h_im = frame_channels[other_user, other_ap].reshape((-1, 1))

                            h_km = frame_channels[user, other_ap].reshape((-1, 1))
                            w_im = frame_precoders[other_user, other_ap].reshape((-1, 1))
                            if ap <= other_ap:
                                C_vh_squared[user, other_user, ap, other_ap] += \
                                    (conj(conj(v_kn).T @ h_in) * (conj(v_km).T @ h_im))[0,0]
                                C_wh_squared[user, other_user, ap, other_ap] += \
                                    (conj(conj(h_kn).T @ w_in) * (conj(h_km).T @ w_im))[0,0]

                # compute 1st order combiner/combiner correlation


        # compute the expectations
        C_v /= num_frames
        C_vh /= num_frames
        C_wh /= num_frames
        C_vh_squared /= num_frames
        C_wh_squared /= num_frames

        # compute the lower triangle of the correlation matrices
        C_vh_squared = (C_vh_squared + transpose(conj(triu(C_vh_squared, 1)), (0, 1, 3, 2)))
        C_wh_squared = (C_wh_squared + transpose(conj(triu(C_wh_squared, 1)), (0, 1, 3, 2)))

        # form the optimization matrices
        T_ul = zeros((num_users, num_aps, num_aps), dtype=complex)
        T_dl = zeros((num_users, num_aps, num_aps), dtype=complex)
        Q_ul = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        Q_dl = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)

        for user in range(num_users):
            for ap in range(num_aps):
                for other_ap in range(num_aps):
                    T_ul[user, ap, other_ap] = (conj(C_vh[user, ap]) * C_vh[user, other_ap] +
                                                C_vh[user, ap] * conj(C_vh[user, other_ap]))/2
                    T_dl[user, ap, other_ap] = (conj(C_wh[user, ap]) * C_wh[user, other_ap] +
                                                C_wh[user, ap] * conj(C_wh[user, other_ap]))/2
                    for other_user in range(num_users):
                        Q_ul[user, other_user, ap, other_ap] = \
                            (C_vh_squared[user, other_user, ap, other_ap] + C_vh_squared[user, other_user, other_ap, ap])/2
                        Q_dl[user, other_user, ap, other_ap] = \
                            (C_wh_squared[user, other_user, ap, other_ap] + C_wh_squared[user, other_user, other_ap, ap])/2

        return Q_dl, Q_ul, T_ul, T_dl, C_v

    def compute_clustering_optimization_model_mrc(self):
        # local object variables
        num_aps = self.num_aps
        num_users = self.num_users
        R = self.channel_model.user_ap_corr
        R_squared = self.channel_model.compute_fourth_moment()
        rho = self.downlink_power_vec
        # optimization matrices
        T_ul = zeros((num_users, num_aps, num_aps), dtype=complex)
        T_dl = zeros((num_users, num_aps, num_aps), dtype=complex)
        Q_ul = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        Q_dl = zeros((num_users, num_users, num_aps, num_aps), dtype=complex)
        C_v = zeros((num_users, num_aps), dtype=complex)


        for user in range(num_users):
            trace_R_k = trace(self.channel_model.get_block_diag_corr(user))
            rho_k = rho[user]
            for ap in range(num_aps):
                R_kn = R[user, ap]
                C_v[user, ap] = trace(R_kn)
                for other_ap in range(num_aps):
                    R_km = R[user, other_ap]
                    T_ul[user, ap, other_ap] = trace(R_kn) * trace(R_km)
                    T_dl[user, ap, other_ap] = rho_k * trace(R_kn) * trace(R_km) / trace_R_k
                    for other_user in range(num_users):
                        rho_i = rho[other_user]
                        R_in = R[other_user, ap]
                        trace_R_i = trace(self.channel_model.get_block_diag_corr(other_user))
                        if user != other_user and ap == other_ap:
                            Q_ul[user, other_user, ap, other_ap] = trace(R_kn @ R_in)
                            Q_dl[user, other_user, ap, other_ap] = sqrt(rho_k * rho_i) * trace(R_kn @ R_in)/trace_R_i
                        elif user == other_user:
                            if ap == other_ap:
                                Q_ul[user, other_user, ap, other_ap] = trace(R_squared[user, ap])
                                Q_dl[user, other_user, ap, other_ap] = rho_k * trace(R_squared[user, ap])/trace_R_k
                            else:
                                Q_ul[user, other_user, ap, other_ap] = trace(R_kn) * trace(R_km)
                                Q_dl[user, other_user, ap, other_ap] = rho_k * trace(R_kn) * trace(R_km)/trace_R_k

        return Q_dl, Q_ul, T_ul, T_dl, C_v

    def compute_SINR_ul_mrc_components(self, Q_ul, T_ul, C_v):
        # local definitions
        p = self.uplink_power_vec
        num_users = self.num_users
        X = self.clusters

        # storing variables
        sig = zeros((num_users,))
        interf_noise = zeros((num_users,))

        for user in range(num_users):
            p_k = p[user]
            x_k = X[user, :].reshape((-1, 1))
            if np.sum(x_k) == 0:
                continue
            sig[user] = real(p_k * (x_k.T @ T_ul[user] @ x_k))[0, 0]
            for other_user in range(num_users):
                p_i = p[other_user]
                interf_noise[user] += real(p_i * (x_k.T @ Q_ul[user, other_user] @ x_k))[0, 0]
            interf_noise[user] -= sig[user]
            interf_noise[user] += real(x_k.T @ C_v[user].reshape((-1, 1)))[0, 0]

        return sig, interf_noise

    def compute_SINR_dl_mrc_components(self, Q_dl, T_dl):
        # local definitions
        num_users = self.num_users
        X = self.clusters

        # storing variables
        sig = zeros((num_users,))
        interf_noise = ones((num_users,))

        for user in range(num_users):
            x_k = X[user, :].reshape((-1, 1))
            if np.sum(x_k) == 0:
                continue
            sig[user] = real(x_k.T @ T_dl[user] @ x_k)[0, 0]
            for other_user in range(num_users):
                x_i = X[other_user, :].reshape((-1, 1))
                interf_noise[user] += real(x_i.T @ Q_dl[user, other_user] @ x_i)[0, 0]
            interf_noise[user] -= sig[user]

        return sig, interf_noise

    def estimate_alpha_mu_dl(self, channels, precoders, Q_dl, T_dl):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        clusters_block_diag = self.clusters_block_diag
        clusters = self.clusters

        # accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interf_norm = zeros((num_users, num_users))

        for user in range(num_users):
            if np.sum(clusters[user]) == 0:
                continue
            for frame in range(num_frames):
                # compute sig
                w_k = precoders[frame, user].reshape((-1, 1))
                h_k = channels[frame, user].reshape((-1, 1))
                D_k = clusters_block_diag[user]
                sig[user] += (conj(h_k).T @ D_k @ w_k)[0, 0]
                # compute interference
                for other_user in range(num_users):
                    w_i = precoders[frame, other_user].reshape((-1, 1))
                    D_i = clusters_block_diag[other_user]
                    q = (conj(h_k).T @ D_i @ w_i)[0, 0]
                    interf_norm[user, other_user] += real(conj(q) * q)

        sig /= num_frames
        interf_norm /= num_frames
        interf_norm = sum(interf_norm, axis=1)
        sig_norm = real(conj(sig) * sig)
        sig_norm_ref, interf_noise_norm_nerf = self.compute_SINR_dl_mrc_components(Q_dl, T_dl)
        sig_norm_ref[sig_norm_ref == 0] = 1

        return real(sig_norm / sig_norm_ref), real((interf_norm - sig_norm + 1) / interf_noise_norm_nerf)

    def estimate_alpha_mu_ul(self, channels, combiners, Q_ul, T_ul, C_v):
        # local definitions
        num_frames = channels.shape[0]
        num_users = self.num_users
        clusters_block_diag = self.clusters_block_diag
        p = self.uplink_power_vec
        clusters = self.clusters

        # accumulation variables
        sig = zeros((num_users,), dtype=complex)
        interf_norm = zeros((num_users, num_users))
        noise_norm = zeros((num_users,))

        for user in range(num_users):
            if np.sum(clusters[user]) == 0:
                continue
            for frame in range(num_frames):
                # compute sig
                p_k = p[user]
                v_k = combiners[frame, user].reshape((-1, 1))
                h_k = channels[frame, user].reshape((-1, 1))
                D_k = clusters_block_diag[user]
                sig[user] += sqrt(p_k) * (conj(v_k).T @ D_k @ h_k)[0, 0]
                noise_norm[user] += real(conj(D_k @ v_k).T @ (D_k @ v_k))[0, 0]

                # compute interference
                for other_user in range(num_users):
                    p_i = p[other_user]
                    h_i = channels[frame, other_user].reshape((-1, 1))
                    q = sqrt(p_i) * (conj(v_k).T @ D_k @ h_i)[0, 0]
                    interf_norm[user, other_user] += real(conj(q) * q)

        sig /= num_frames
        interf_norm /= num_frames
        noise_norm /= num_frames
        interf_norm = sum(interf_norm, axis=1)
        sig_norm = conj(sig) * sig

        sig_norm_ref, interf_noise_norm_ref = self.compute_SINR_ul_mrc_components(Q_ul, T_ul, C_v)
        sig_norm_ref[sig_norm_ref == 0] = 1
        interf_noise_norm_ref[interf_noise_norm_ref == 0] = 1

        return real(sig_norm / sig_norm_ref), \
            real((interf_norm - sig_norm + noise_norm) / interf_noise_norm_ref)

    def compute_alpha_mu_SE_dl(self, alpha, mu, Q_dl, T_dl):
        # local definitions
        prelog = 1 - self.pilot_len/self.block_len
        sig_norm, interf_noise_norm = self.compute_SINR_dl_mrc_components(Q_dl, T_dl)
        mu[mu == 0] = 1
        alpha[mu == 0] = 0
        interf_noise_norm[interf_noise_norm == 0] = 1
        sig_norm[interf_noise_norm == 0] = 0
        SINR = (alpha * sig_norm) / (mu * interf_noise_norm)
        return prelog * np.log2(1 + SINR)

    def compute_alpha_mu_SE_ul(self, alpha, mu, Q_ul, T_ul, C_v):
        # local definitions
        prelog = 1 - self.pilot_len / self.block_len
        sig_norm, interf_noise_norm = self.compute_SINR_ul_mrc_components(Q_ul, T_ul, C_v)
        mu[mu == 0] = 1
        alpha[mu == 0] = 0
        interf_noise_norm[interf_noise_norm == 0] = 1
        sig_norm[interf_noise_norm == 0] = 0
        SINR = (alpha * sig_norm) / (mu * interf_noise_norm)
        return prelog * np.log2(1 + SINR)



