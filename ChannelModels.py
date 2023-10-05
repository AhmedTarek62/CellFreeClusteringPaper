import numpy as np
from numpy import log10, dot, array, zeros, sqrt, cos, sin, pi, exp, matmul, conj, where, eye
from numpy.linalg import pinv
from scipy.integrate import dblquad
from scipy.linalg import sqrtm
from numpy.random import normal
from numpy.linalg import cholesky
from scipy.linalg import block_diag
from Helpers import *


class CorrelatedRayleigh:
    def __init__(self):
        self.user_ap_distance = None
        self.user_user_distance = None
        self.num_antennas = None
        self.num_users = None
        self.num_aps = None
        self.path_loss_shadowing = None
        self.user_ap_corr = None
        self.user_ap_corr_chol = None
        self.user_ap_sig_corr = None
        self.user_ap_error_corr = None
        self.block_user_error_corr = None

    def compute_large_scale_info(self, num_antennas: int, user_ap_distance: np.ndarray, user_user_distance: np.ndarray):
        self.user_ap_distance = user_ap_distance
        self.user_user_distance = user_user_distance
        self.num_antennas = num_antennas
        self.num_users = user_user_distance.shape[0]
        self.num_aps = user_ap_distance.shape[1]
        self.path_loss_shadowing = self.compute_path_loss_shadowing()
        self.user_ap_corr = self.generate_joint_gaussian_corr()
        self.user_ap_corr_chol = [[cholesky(self.user_ap_corr[user, AP]) for AP in range(self.num_aps)]
                                  for user in range(self.num_users)]


    def set_large_scale_info(self,
                             num_antennas: int,
                             user_ap_distance: np.ndarray,
                             user_user_distance: np.ndarray,
                             path_loss_shadowing: np.ndarray,
                             user_ap_corr: np.ndarray,
                             user_ap_corr_chol: np.ndarray,
                             ):
        self.user_ap_distance = user_ap_distance
        self.user_user_distance = user_user_distance
        self.num_antennas = num_antennas
        self.num_users = user_user_distance.shape[0]
        self.num_aps = user_ap_distance.shape[1]
        self.path_loss_shadowing = path_loss_shadowing
        self.user_ap_corr = user_ap_corr
        self.user_ap_corr_chol = user_ap_corr_chol

    def compute_path_loss_shadowing(self):
        user_user_distance = self.user_user_distance
        num_users = self.num_users
        B = 20e6
        noise_figure = 7
        noise_variance_dBm = -174 + 10 * log10(B) + noise_figure
        path_loss_shadowing = -30.5 + -36.7 * log10(self.user_ap_distance) - noise_variance_dBm

        cov_shadowing = zeros((num_users, num_users))
        for user in range(num_users):
            cov_shadowing[user, :] = 16 * (2**(-user_user_distance[user, :]/9))
        c = cholesky(cov_shadowing)
        for ap in range(self.num_aps):
            path_loss_shadowing[:, ap] += dot(c, normal(loc=0, scale=1, size=(num_users,)))
        return path_loss_shadowing

    def compute_fourth_moment(self):
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        fourth_moment = zeros((num_users, num_aps, num_antennas**2, num_antennas**2), dtype=complex)
        I = np.eye(num_antennas**2)
        K = comm_mat(num_antennas, num_antennas)
        for user in range(num_users):
            for ap in range(num_aps):
                R_kl = self.user_ap_corr[user, ap]
                fourth_moment[user , ap] = (I + K) @ np.kron(R_kl, R_kl)
        return fourth_moment

    def generate_joint_gaussian_corr(self):
        def f_real(phi, theta, m, n, sigma_phi, sigma_theta):
            return cos(pi * (m - n) * sin(phi) * cos(theta))\
                   * 1 / (2 * pi * sigma_phi * sigma_theta) \
                * exp(-(phi - 30 / 180 * pi) ** 2 / (2 * sigma_phi ** 2)) \
                * exp(-(theta + 15 / 180 * pi) ** 2 / (2 * sigma_theta ** 2))

        def f_imag(phi, theta, m, n, sigma_phi, sigma_theta):
            return sin(pi * (m - n) * sin(phi) * cos(theta))\
                   * 1 / (2 * pi * sigma_phi * sigma_theta)\
                   * exp(-(phi - 30 / 180 * pi)**2 / (2 * sigma_phi ** 2))\
                   * exp(-(theta + 15 / 180 * pi)**2 / (2 * sigma_theta ** 2))

        num_antennas = self.num_antennas
        num_users = self.num_users
        num_aps = self.num_aps
        sigma_phi = 15/180*pi
        sigma_theta = 15/180*pi
        R = zeros((num_users, num_aps, num_antennas, num_antennas), dtype=complex)
        for i in range(num_antennas):
            for j in range(num_antennas):
                if j >= i:
                    R_ij = dblquad(f_real, -20*sigma_phi, 20*sigma_phi, -20*sigma_theta, 20*sigma_theta,
                                   args=(i, j, sigma_phi, sigma_theta))[0] \
                           + 1j * dblquad(f_imag, -20*sigma_phi, 20*sigma_phi, -20*sigma_theta, 20*sigma_theta,
                                   args=(i, j, sigma_phi, sigma_theta))[0]
                    R[:, :, i, j] = R_ij
                    R[:, :, j, i] = conj(R_ij)
        betas = 10 ** (self.path_loss_shadowing/10)
        betas_mat = np.tile(betas.reshape((num_users, num_aps, 1, 1)), (1, 1, num_antennas, num_antennas))
        return betas_mat * R

    def sample(self, num_frames):
        num_users = self.num_users
        num_aps = self.num_aps
        num_antennas = self.num_antennas
        user_ap_corr_chol = self.user_ap_corr_chol
        H = sqrt(1/2) * (normal(0, 1, (num_frames, num_users, num_aps, num_antennas)) +
                                     1j * normal(0, 1, (num_frames, num_users, num_aps, num_antennas)))

        for user in range(self.num_users):
            for ap in range(self.num_aps):
                Rsqrt = user_ap_corr_chol[user][ap]
                for frame in range(num_frames):
                    H[frame, user, ap] = dot(Rsqrt, H[frame, user, ap])
        return H

    @staticmethod
    def compute_collective_channel_repr(H: np.ndarray):
        num_users, num_aps, num_antennas = H.shape
        reshaped_H = zeros((num_users, num_aps * num_antennas), dtype=complex)
        for user in range(num_users):
            for ap in range(num_aps):
                reshaped_H[user, ap * num_antennas:(ap + 1) * num_antennas] = H[user, ap]
        return reshaped_H

    def compute_error_sig_corr(self, pilot_alloc: np.ndarray, pilot_power_vec: np.ndarray,
                               pilot_len: int):

        num_antennas = self.num_antennas
        num_users = self.num_users
        num_aps = self.num_aps
        user_ap_corr = self.user_ap_corr
        tau_p = pilot_len
        user_ap_sig_corr = zeros((num_users, num_aps, num_antennas, num_antennas), dtype=complex)
        user_ap_error_corr = zeros((num_users, num_aps, num_antennas, num_antennas), dtype=complex)
        for ap in range(num_aps):
            for user in range(num_users):
                R_kl = user_ap_corr[user, ap]
                eta_k = pilot_power_vec[user]
                pilot_sharing_users = where(pilot_alloc == pilot_alloc[user])[0]
                for other_user in pilot_sharing_users:
                    eta_i = pilot_power_vec[other_user]
                    R_il = user_ap_corr[other_user, ap]
                    user_ap_sig_corr[user, ap] += eta_i * tau_p * R_il
                user_ap_sig_corr[user, ap] += eye(num_antennas)
                sig_corr_inv = pinv(user_ap_sig_corr[user, ap])
                user_ap_error_corr[user, ap] = R_kl - eta_k * tau_p * (R_kl @ sig_corr_inv @ R_kl)
        self.user_ap_sig_corr = user_ap_sig_corr
        self.user_ap_error_corr = user_ap_error_corr
        self.block_user_error_corr = array([block_diag(*(user_ap_error_corr[user, ap] for ap in range(num_aps)))
                                            for user in range(num_users)])

    def estimate_channel(self, cluster_mat: np.ndarray, received_pilots: np.ndarray, pilots: np.ndarray,
                         pilot_power_vec: np.ndarray, pilot_alloc: np.ndarray):
        num_antennas = self.num_antennas
        num_users = self.num_users
        num_aps = self.num_aps
        user_ap_corr = self.user_ap_corr
        user_ap_sig_corr = self.user_ap_sig_corr
        tau_p = pilots.shape[0]

        channel_estimate = zeros((num_users, num_aps, num_antennas), dtype=complex)
        for ap in range(num_aps):
            proj_received_pilots = matmul(received_pilots[ap], conj(pilots) / sqrt(tau_p))
            for user in range(num_users):
                if cluster_mat[user, ap]:
                    user_proj = proj_received_pilots[:, int(pilot_alloc[user])].reshape((-1, 1))
                    R_kl = user_ap_corr[user, ap]
                    eta_k = pilot_power_vec[user]
                    sig_corr_inv = pinv(user_ap_sig_corr[user, ap])
                    channel_estimate[user, ap] \
                        = sqrt(eta_k * tau_p) * (R_kl @ sig_corr_inv @ user_proj).reshape((-1,))
        return channel_estimate

    def generate_channel_estimate(self, num_frames):
        num_antennas = self.num_antennas
        num_users = self.num_users
        num_aps = self.num_aps
        user_ap_corr = self.user_ap_corr
        user_ap_error_corr = self.user_ap_error_corr
        channel_estimate = sqrt(1/2) * (normal(0, 1, (num_frames, num_users, num_aps, num_antennas)) +
                                     1j * normal(0, 1, (num_frames, num_users, num_aps, num_antennas)))
        for user in range(num_users):
            for ap in range(num_aps):
                R_kl_hat_sqrt = sqrtm(user_ap_corr[user, ap] - user_ap_error_corr[user, ap])
                for frame in range(num_frames):
                    channel_estimate[frame, user, ap] = dot(R_kl_hat_sqrt, channel_estimate[frame, user, ap])
        return channel_estimate

    def generate_channel_error(self, num_frames):
        num_antennas = self.num_antennas
        num_users = self.num_users
        num_aps = self.num_aps
        user_ap_error_corr = self.user_ap_error_corr
        channel_error = sqrt(1/2) * (normal(0, 1, (num_frames, num_users, num_aps, num_antennas)) +
                                     1j * normal(0, 1, (num_frames, num_users, num_aps, num_antennas)))
        for user in range(num_users):
            for ap in range(num_aps):
                C_kl_sqrt = sqrtm(user_ap_error_corr[user, ap])
                for frame in range(num_frames):
                    channel_error[frame, user, ap] = dot(C_kl_sqrt, channel_error[frame, user, ap])
        return channel_error

    def get_error_corr(self, user: int, AP: int):
        return self.user_ap_error_corr[user, AP]

    def get_corr(self, user: int, AP: int):
        return self.user_ap_corr[user, AP]

    def get_block_diag_error_corr(self, user: int):
        return self.block_user_error_corr[user]

    def get_block_diag_corr(self, user: int):
        return block_diag(*(self.user_ap_corr[user, AP] for AP in range(self.num_aps)))
