import numpy as np
import CellFreeNetwork

def comm_mat(m,n):
    # determine permutation applied by K
    w = np.arange(m*n).reshape((m,n),order='F').T.ravel(order='F')

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(m*n)[w,:]

def calculate_SE_ul(X, T_ul, Q_ul, c_v):
    num_users, num_aps = X.shape
    SE_ul = 0
    for user in range(num_users):
        x_k = X[user,:].reshape((-1, 1))
        numerator_ul = (x_k.T @ T_ul[user] @ x_k)[0,0]
        denominator_ul = -numerator_ul
        for other_user in range(num_users):
            denominator_ul += (x_k.T @ Q_ul[user, other_user] @ x_k)[0,0]
        sinr_ul = np.real(numerator_ul/(denominator_ul + (x_k.T @ c_v[user].reshape((-1,1)))[0, 0]))
        SE_ul += np.log2(1 + sinr_ul)
    return SE_ul

def calculate_SE_dl(X, T_dl, Q_dl):
    num_users, num_aps = X.shape
    SE_dl = 0
    for user in range(num_users):
        x_k = X[user,:].reshape((-1, 1))
        numerator_dl = (x_k.T @ T_dl[user] @ x_k)[0,0]
        denominator_dl = -numerator_dl
        for other_user in range(num_users):
            x_i = X[other_user, :].reshape((-1, 1))
            denominator_dl += (x_i.T @ Q_dl[user, other_user] @ x_i)[0,0]
        SE_dl += np.log2(1 + numerator_dl/(denominator_dl + 1))
    return SE_dl

def calculate_objective_fun(X, Q, T, F):
    num_users, num_aps, _ = F.shape
    sigma_downlink = np.sqrt(10 ** (-94 / 10) / 1000)
    f_val_uplink = 0
    f_val_downlink = 0
    for user in range(num_users):
        x_k = X[user,:].reshape((-1, 1))

        # compute downlink SINR
        numerator_dl =  x_k.T @ Q[user, user] @ x_k
        denominator_dl = sigma_downlink ** 2
        for other_user in range(num_users):
            if user != other_user:
                x_i = X[other_user,:].reshape((-1, 1))
                denominator_dl += x_i.T @ Q[user, other_user] @ x_i
        sinr_dl = np.real(numerator_dl/denominator_dl)[0,0]

        # compute uplink SINR
        numerator_ul =  x_k.T @ T[user, user] @ x_k
        denominator_ul = F[user]
        for other_user in range(num_users):
            if user != other_user:
                denominator_ul += T[user, other_user]
        denominator_ul = x_k.T @ denominator_ul @ x_k
        sinr_ul = np.real(numerator_ul/denominator_ul)[0,0]

        f_val_uplink += np.log2(1 + sinr_ul)
        f_val_downlink += np.log2(1 + sinr_dl)
    return f_val_uplink, f_val_downlink


def reshape_combiners(combiners: np.ndarray):
    num_frames, num_users, num_aps, num_antennas = combiners.shape
    reshaped_combiners = np.zeros((num_frames, num_users, num_aps * num_antennas), dtype=complex)
    for frame in range(num_frames):
        for user in range(num_users):
            for ap in range(num_aps):
                reshaped_combiners[frame, user, ap * num_antennas: (ap + 1) * num_antennas] = combiners[frame, user, ap]
    return reshaped_combiners


def calculate_ce(network: CellFreeNetwork, clustering = False):
    num_users = network.num_users
    num_aps = network.num_aps
    ce = 0
    for user in range(num_users):
        for ap in range(num_aps):
            if clustering:
                if network.clusters_all_opt[user, ap] == 1:
                    ce += np.trace(network.channel_model.get_error_corr(user, ap))
            else:
                ce += np.trace(network.channel_model.get_error_corr(user, ap))
    return np.real(ce)
