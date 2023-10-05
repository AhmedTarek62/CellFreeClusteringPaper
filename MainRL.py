from Game import ClusteringGame
import math
import torch
import torch.optim as optim
from DQN import DQN
import matplotlib
import matplotlib.pyplot as plt
from SettingParams import mock_params
from CellFreeNetwork import CellFreeNetwork
from ClusteringAlgorithms import massive_access_clustering
from ReplayMemory import *
from torch import nn
import numpy as np
from itertools import count

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize clustering game
params = {'num_aps': 10,
          'num_users': 7,
          'num_antennas': 4,
          'ap_dist': 'Uniform',
          'users_dist': 'Uniform',
          'coverage_area_len': 200,
          'channel_model': 'Correlated Rayleigh',
          'block_len': 200,
          'pilot_len': 5,
          'pilot_alloc_alg': 'random',
          'pilot_power_control_alg': 'max',
          'uplink_power_control_alg': 'max',
          'downlink_power_alloc_alg': 'fractional',
          'user_max_power': 100,
          'ap_max_power': 200,
          'uplink_noise_power': -94,
          'downlink_noise_power': -94,
          'clustering_alg': 'canonical'
          }

num_users = params['num_users']
num_aps = params['num_aps']
num_actions = num_users * num_aps
users_per_ap = params['pilot_len']
alg = 'MMSE'
game = ClusteringGame(num_users, num_aps, users_per_ap, CellFreeNetwork(**params), alg)

# training parameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-1
steps_done = 0

policy_net = DQN(game).to(device)
target_net = DQN(game).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


# helper functions
def select_action(game: ClusteringGame, state_t):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state_t).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[game.sample_action()]], device=device, dtype=torch.int64)


def create_network_state(game: ClusteringGame, state):
    state_processed = [game.betas[i][j] for i in range(game.num_users) for j in range(game.num_aps)]
    state_processed.extend([state[i] for i in range(game.num_actions)])
    return state_processed


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    next_states = torch.cat([s for s in batch.next_state])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    with torch.no_grad():
        next_state_values = target_net(next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


episode_rewards = []


def plot_rewards(show_result=False):
    plt.figure(1)
    rewards_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards_t.numpy())
    # Take 100 episode averages and plot them too
    if len(rewards_t) >= 100:
        means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


num_episodes = 600
control_episodes = 10
i_control_episodes = 0
max_steps = game.num_aps * (game.users_per_ap + 1)

for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    print(f"Episode: {i_episode}")
    game.generate_episode()
    if i_episode % 10 == 0 and i_control_episodes < control_episodes:
        clusters = massive_access_clustering(game.network.channel_model.path_loss_shadowing, game.network.pilot_len)
        control_action_seq = np.where(clusters.reshape((-1,)) == 1)[0]
        i_control_episodes += 1
        control_episode = True
        print(f"Control episode: {i_control_episodes}")
    else:
        control_episode = False

    state = game.get_init_state()
    state_t = torch.tensor(create_network_state(game, state), dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        print(f"step {t + 1} out of {max_steps}")
        if control_episode:
            action = torch.tensor([[control_action_seq[t]]], device=device, dtype=torch.int64)
        else:
            action = select_action(game, state_t)
        new_state = game.result(state, action)
        reward = game.reward(new_state) - game.reward(state)
        terminated = game.terminal_test(new_state)
        if terminated:
            reward += 10

        reward_t = torch.tensor([reward], device=device)
        next_state_t = torch.tensor(create_network_state(game, new_state), dtype=torch.float32,
                                    device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state_t, action, next_state_t, reward_t)

        # Move to the next state
        state = new_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if t == max_steps or terminated:
            episode_rewards.append(reward)
            plot_rewards()
            break

print('Complete')
plot_rewards(show_result=True)
plt.ioff()
plt.show()
test = []
