"""Evaluate the trained policy for the Blackjack environment and compare with dealer-like policy with reproducible episodes."""
import gymnasium as gym
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the trained policy
with open("policy.pkl", "rb") as f:
    policy = pickle.load(f)

# Function to play one episode given a policy and optional seed
def play_episode(env, policy_func, seed=None):
    observation, _ = env.reset(seed=seed)
    done = False
    while not done:
        current_sum, dealer_card, usable_ace = observation
        action = policy_func(current_sum, dealer_card, usable_ace)
        observation, reward, done, _, _ = env.step(action)
    return reward

# Policy wrapper for trained policy
def agent_policy(current_sum, dealer_card, usable_ace):
    if current_sum < 12:
        return 1  # Hit if sum < 12
    state = (current_sum, usable_ace, dealer_card)
    return 1 if policy.get(state, False) else 0  # 1=hit, 0=stick

# Dealer-like policy: hit until reaching 17
def dealer_like_policy(current_sum, dealer_card, usable_ace):
    return 1 if current_sum < 17 else 0

# Simulation function
def simulate_policy(env, policy_func, seeds):
    wins = losses = draws = 0
    for seed in tqdm(seeds):
        reward = play_episode(env, policy_func, seed=seed)
        if reward > 0:
            wins += 1
        elif reward < 0:
            losses += 1
        else:
            draws += 1
    return wins, losses, draws

# Create environment
env = gym.make('Blackjack-v1', natural=False)

# Number of episodes
num_games = 500000
# Create seeds for reproducibility (Python ints)
seeds = [int(s) for s in range(num_games)]

# Simulate trained agent
agent_results = simulate_policy(env, agent_policy, seeds)
# Simulate dealer-like policy using the same seeds
dealer_results = simulate_policy(env, dealer_like_policy, seeds)

env.close()

# Print results
print("Trained Agent:")
print(f"Player Wins: {agent_results[0]} ({agent_results[0]/num_games*100:.2f}%)")
print(f"Dealer Wins: {agent_results[1]} ({agent_results[1]/num_games*100:.2f}%)")
print(f"Draws: {agent_results[2]} ({agent_results[2]/num_games*100:.2f}%)\n")

print("Dealer-like Policy:")
print(f"Player Wins: {dealer_results[0]} ({dealer_results[0]/num_games*100:.2f}%)")
print(f"Dealer Wins: {dealer_results[1]} ({dealer_results[1]/num_games*100:.2f}%)")
print(f"Draws: {dealer_results[2]} ({dealer_results[2]/num_games*100:.2f}%)")

# Plot side-by-side comparison
categories = ['Player Wins', 'Dealer Wins', 'Draws']
x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, agent_results, width, label='Trained Agent', color='green')
rects2 = ax.bar(x + width/2, dealer_results, width, label='Dealer-like Policy', color='red')

ax.set_ylabel('Number of Games')
ax.set_title(f'Comparison over {num_games} Games')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.show()
