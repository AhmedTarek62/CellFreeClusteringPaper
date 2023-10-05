from torch import nn
from Game import ClusteringGame
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, game: ClusteringGame):
        super(DQN, self).__init__()
        input_size = (game.num_aps * game.num_users) * 2
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, game.num_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
