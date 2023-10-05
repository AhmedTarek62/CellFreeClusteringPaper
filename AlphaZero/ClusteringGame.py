from SimulationScripts.CellFreeNetwork import CellFreeNetwork
from SimulationScripts.SettingParams import mock_params
import numpy as np


params = mock_params

class ClusteringGame:
    def __init__(self, num_users: int, num_aps: int, users_per_ap: int, network: CellFreeNetwork, alg: str):
        self.num_users = num_users
        self.num_aps = num_aps
        self.users_per_ap = users_per_ap
        self.num_actions = num_users * num_aps
        self.network = network
        self.alg = alg

    def get_init_board(self):
        return tuple(tuple(0 for _ in range(self.num_users)) for _ in range(self.num_aps))

    def actions(self, state):
        action_list = []
        for ap, ap_cluster in enumerate(state):
            if sum(ap_cluster) < self.users_per_ap:
                action_list.extend([(ap, user) for user in range(self.num_users) if state[ap][user] == 0])

    def result(self, state, action):
        mutable_state = list(state)
        mutable_ap_alloc = list(state[action[0]])
        mutable_ap_alloc[action[1]] = 1
        mutable_state[action[0]] = mutable_ap_alloc
        return tuple(mutable_state)

    def _reward(self, state):
        self.network.set_clusters(np.array(state).T)

        num_frames = 50
        collective_channels, _, _, _ = self.network.generate_channel_realizations(num_frames)
        combiners = self.network.simulate_uplink_centralized(self.alg, collective_channels, collective_channels)
        precoders = self.network.simulate_downlink_centralized(self.alg, collective_channels, collective_channels)
        reward = \
            (self.network.compute_uplink_SE_centralized(collective_channels, combiners) +
             self.network.compute_downlink_SE_centralized(collective_channels, precoders)
             )/self.num_actions # average sum SE per UE
        return reward
    def terminal_test(self, state):
        if all(sum(ap_alloc) == self.users_per_ap for ap_alloc in state):
            return self._reward(state)
        else:
            return 0

    def string_representation(self, state):
        return ''.join(state[i][j] for i in range(self.num_aps) for j in range(self.num_users))

