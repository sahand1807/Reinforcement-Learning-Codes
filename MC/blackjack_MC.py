import numpy as np
import random
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from tqdm import trange
import gymnasium
from gymnasium.envs.toy_text.blackjack import BlackjackEnv

# -------------------- Set random seeds for reproducibility --------------------
SEED = 10
np.random.seed(SEED)
random.seed(SEED)

# -------------------- Helper Function --------------------
class StateActionSet:
    def __init__(self):
        self.pairs = []
        self.map = set()

    def add_pair(self, pair):
        if pair in self.map:
            return
        self.pairs.append(pair)
        self.map.add(pair)

def get_hand(hard_sum, usable_ace):
    hand = []
    if usable_ace:
        remaining = hard_sum - 1
        while remaining > 0:
            card = min(10, remaining)
            hand.append(card)
            remaining -= card
        hand.append(1)
    else:
        remaining = hard_sum
        while remaining > 0:
            card = min(10, remaining)
            if card == 1:
                for i in range(len(hand) - 1, -1, -1):
                    if hand[i] >= 2:
                        hand[i] -= 1
                        remaining += 1
                        card = 2
                        break
            hand.append(card)
            remaining -= card
    return hand

def update_policy(Q, policy, returns, state_action_pairs, reward):
    for pair in state_action_pairs:
        returns[pair] += 1
        Q[pair] += (reward - Q[pair]) / returns[pair]

        state = pair[0]
        if Q.get((state, True), 0) > Q.get((state, False), 0):
            policy[state] = True
        else:
            policy[state] = False

def generate_episode(Q, policy, returns):
    # Exploring start
    current_value = np.random.randint(11, 22)
    usable_ace = bool(np.random.randint(0, 2))
    if usable_ace:
        hard_sum = current_value - 10
    else:
        hard_sum = current_value
    dealer_card = np.random.randint(1, 11)

    env = BlackjackEnv()
    env.player = get_hand(hard_sum, usable_ace)
    env.dealer = [dealer_card]

    state_action_set = StateActionSet()
    hit = bool(np.random.randint(0, 2))  # hit=True, stick=False
    current_state = (current_value, usable_ace, dealer_card)
    state_action_set.add_pair((current_state, hit))

    if not hit:
        # Stick
        _, reward, _, _, _ = env.step(0)
        update_policy(Q, policy, returns, state_action_set.pairs, reward)
        return

    # Hit
    observation, reward, terminated, _, _ = env.step(1)
    if terminated:
        update_policy(Q, policy, returns, state_action_set.pairs, -1)
        return

    new_value = observation[0]
    new_usable = observation[2]
    new_state = (new_value, new_usable, dealer_card)

    while True:
        hit = policy.get(new_state, False)
        state_action_set.add_pair((new_state, hit))
        if not hit:
            break

        observation, reward, terminated, _, _ = env.step(1)
        if terminated:
            update_policy(Q, policy, returns, state_action_set.pairs, -1)
            return

        new_value = observation[0]
        new_usable = observation[2]
        new_state = (new_value, new_usable, dealer_card)

    # Stick
    _, reward, _, _, _ = env.step(0)
    update_policy(Q, policy, returns, state_action_set.pairs, reward)

# -------------------- 3. Training --------------------
def monte_carlo_es_blackjack(episodes=2000000):
    Q = {}
    policy = {}
    returns = {}

    # Initialize all states
    for player_sum in range(11, 22):
        for usable_ace in [False, True]:
            for dealer_card in range(1, 11):
                state = (player_sum, usable_ace, dealer_card)
                Q[(state, True)] = 0
                Q[(state, False)] = 0
                returns[(state, True)] = 0
                returns[(state, False)] = 0
                policy[state] = False if player_sum >= 20 else True

    # Monte Carlo ES with progress bar
    for _ in trange(episodes, desc="Training Episodes", ncols=100):
        generate_episode(Q, policy, returns)

    return Q, policy

# -------------------- 4. Visualization --------------------
def plot_policy_heatmap(policy):
    usable_ace_matrix = np.zeros((11, 10))
    no_usable_ace_matrix = np.zeros((11, 10))

    for state, hit in policy.items():
        player_sum, usable_ace, dealer_card = state
        row = player_sum - 11
        col = dealer_card - 1
        if usable_ace:
            usable_ace_matrix[row, col] = int(hit)
        else:
            no_usable_ace_matrix[row, col] = int(hit)

    cmap = ListedColormap(['green', 'red'])  # Stick=green, Hit=red

    fig1, axs1 = plt.subplots(1, 2, figsize=(14,6))
    im1 = axs1[0].imshow(usable_ace_matrix, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axs1[0].set_title('Policy with Usable Ace')
    axs1[0].set_xticks(np.arange(10))
    axs1[0].set_xticklabels(['A','2','3','4','5','6','7','8','9','10'])
    axs1[0].set_yticks(np.arange(11))
    axs1[0].set_yticklabels([str(i) for i in range(11,22)])
    fig1.colorbar(im1, ax=axs1[0], ticks=[0,1]).ax.set_yticklabels(['Stick','Hit'])

    im2 = axs1[1].imshow(no_usable_ace_matrix, origin='lower', cmap=cmap, vmin=0, vmax=1, aspect='auto')
    axs1[1].set_title('Policy without Usable Ace')
    axs1[1].set_xticks(np.arange(10))
    axs1[1].set_xticklabels(['A','2','3','4','5','6','7','8','9','10'])
    axs1[1].set_yticks(np.arange(11))
    axs1[1].set_yticklabels([str(i) for i in range(11,22)])
    fig1.colorbar(im2, ax=axs1[1], ticks=[0,1]).ax.set_yticklabels(['Stick','Hit'])

    return fig1

# -------------------- 5. Run Monte Carlo ES --------------------
Q, policy = monte_carlo_es_blackjack(episodes=2000000)

# Save policy for later use
with open("policy.pkl", "wb") as f:
    pickle.dump(policy, f)

# Visualize
fig_policy = plot_policy_heatmap(policy)
plt.show()