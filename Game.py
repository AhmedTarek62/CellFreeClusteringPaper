from CellFreeNetwork import CellFreeNetwork
import numpy as np
import random

class ClusteringGame:
    def __init__(self, num_users: int, num_aps: int, users_per_ap: int, network: CellFreeNetwork, alg: str):
        self.num_users = num_users
        self.num_aps = num_aps
        self.users_per_ap = users_per_ap
        self.num_actions = num_users * num_aps
        self.network = network
        self.all_actions = [i for i in range(self.num_actions)]
        self.alg = alg
        self.betas = None

    def generate_episode(self):
        self.network.generate_snapshot()
        betas = 10 ** (self.network.channel_model.path_loss_shadowing/10)
        self.betas = 0.1 + (0.99 - 0.1)*(betas - np.min(betas))/(np.max(betas) - np.min(betas))

    def get_init_state(self):
        """
        state description: (user0_cluster, user1_cluster, ..., userK_cluster)
        :return:
        """
        return tuple(0 for _ in range(self.num_aps * self.num_users))

    def sample_action(self):
        return random.choice(self.all_actions)

    def result(self, state, action):
        mutable_state = list(state)
        mutable_state[action] = 1
        return tuple(mutable_state)

    def get_clusters_from_state(self, state):
        clusters = np.zeros((self.num_users, self.num_aps))
        for user in range(self.num_users):
            for ap in range(self.num_aps):
                clusters[user, ap] = state[user * self.num_aps + ap]
        return clusters

    def reward(self, state):
        if sum(state) == 0:
            return 0

        self.network.set_clusters(self.get_clusters_from_state(state))
        num_frames = 50
        collective_channels, _, _, _ = self.network.generate_channel_realizations(num_frames)
        combiners = self.network.simulate_uplink_centralized(self.alg, collective_channels, collective_channels)
        precoders = self.network.simulate_downlink_centralized(self.alg, collective_channels, collective_channels)
        reward = np.sum(
            np.nan_to_num(self.network.compute_uplink_SE_centralized(collective_channels, combiners)) +
            np.nan_to_num(self.network.compute_downlink_SE_centralized(collective_channels, precoders))
        )

        return reward

    def terminal_test(self, state):
        clusters = self.get_clusters_from_state(state)
        return all(np.sum(clusters[:, ap]) == self.users_per_ap for ap in range(self.num_aps))

    def string_representation(self, state):
        return ''.join(state[i][j] for i in range(self.num_aps) for j in range(self.num_users))
