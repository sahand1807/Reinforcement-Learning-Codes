import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from tqdm import trange

# -------------------- 1. Environment Simulation --------------------
class BlackjackPlayer:
    def __init__(self, current_sum, usable_ace, dealer_card):
        self.current_sum = current_sum
        self.usable_ace = usable_ace
        self.using_ace = usable_ace
        self.dealer_card = dealer_card

    def add_card(self, card):
        if self.using_ace and self.current_sum + card > 21:
            self.using_ace = False
            self.current_sum += card - 10
        else:
            self.current_sum += card

    def bust(self):
        return self.current_sum > 21

    def get_state(self):
        return (self.current_sum, self.usable_ace, self.dealer_card)

    def should_hit(self, policy):
        return policy[self.get_state()]

class BlackjackDealer:
    def __init__(self, cards):
        self.cards = cards

    def add_card(self, card):
        self.cards.append(card)

    def bust(self):
        return self.get_value() > 21

    def get_value(self):
        total = 0
        ace_count = 0
        for card in self.cards:
            if card == 1:
                ace_count += 1
            else:
                total += card
        while ace_count > 0:
            ace_count -= 1
            total += 11
            if total > 21:
                ace_count += 1
                total -= 11
                total += ace_count
                break
        return total

    def should_hit(self):
        return self.get_value() < 17

# -------------------- 2. Monte Carlo Exploring Starts --------------------
class StateActionSet:
    def __init__(self):
        self.pairs = []
        self.map = set()

    def add_pair(self, pair):
        if pair in self.map:
            return
        self.pairs.append(pair)
        self.map.add(pair)

def draw_card():
    card = np.random.randint(1, 14)
    return 10 if card > 9 else card

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
    player_sum = np.random.randint(11, 22)
    usable_ace = bool(np.random.randint(0, 2))
    dealer_card = np.random.randint(1, 11)

    player = BlackjackPlayer(player_sum, usable_ace, dealer_card)
    dealer = BlackjackDealer([dealer_card])

    state_action_set = StateActionSet()
    action = bool(np.random.randint(0, 2))  # hit=True, stick=False
    state_action_set.add_pair((player.get_state(), action))

    if action:
        player.add_card(draw_card())
        while not player.bust() and player.should_hit(policy):
            state_action_set.add_pair((player.get_state(), True))
            player.add_card(draw_card())

    if player.bust():
        update_policy(Q, policy, returns, state_action_set.pairs, -1)
        return

    state_action_set.add_pair((player.get_state(), False))
    dealer.add_card(draw_card())
    while not dealer.bust() and dealer.should_hit():
        dealer.add_card(draw_card())

    if dealer.bust() or dealer.get_value() < player.current_sum:
        reward = 1
    elif dealer.get_value() > player.current_sum:
        reward = -1
    else:
        reward = 0

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

def plot_value_function(Q):
    player_range = np.arange(12,22)
    dealer_range = np.arange(1,11)
    X, Y = np.meshgrid(dealer_range, player_range)

    Z_usable = np.zeros_like(X, dtype=float)
    Z_no_usable = np.zeros_like(X, dtype=float)
    for i, player in enumerate(player_range):
        for j, dealer in enumerate(dealer_range):
            Z_usable[i,j] = max(Q.get(((player, True, dealer), True),0),
                                 Q.get(((player, True, dealer), False),0))
            Z_no_usable[i,j] = max(Q.get(((player, False, dealer), True),0),
                                    Q.get(((player, False, dealer), False),0))

    fig2, axs2 = plt.subplots(1, 2, figsize=(14,6), subplot_kw={'projection':'3d'})
    axs2[0].plot_surface(X, Y, Z_usable, cmap='viridis')
    axs2[0].set_title('Value Function with Usable Ace')
    axs2[0].set_xlabel('Dealer Showing')
    axs2[0].set_ylabel('Player Sum')
    axs2[0].set_zlabel('Value')

    axs2[1].plot_surface(X, Y, Z_no_usable, cmap='viridis')
    axs2[1].set_title('Value Function without Usable Ace')
    axs2[1].set_xlabel('Dealer Showing')
    axs2[1].set_ylabel('Player Sum')
    axs2[1].set_zlabel('Value')

    return fig2

# -------------------- 5. Run MC --------------------
Q, policy = monte_carlo_es_blackjack(episodes=2000000)
fig_policy = plot_policy_heatmap(policy)
fig_value = plot_value_function(Q)

plt.show()
