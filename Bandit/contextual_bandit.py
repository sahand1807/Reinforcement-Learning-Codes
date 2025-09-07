"""Contextual Bandit for Pick-and-Place Task with different agents:
    1. Epsilon-Greedy (Greedy for ε=0)
    2. Upper Confidence Bound (UCB)
    3. Gradient Bandit
Simulating a robotic arm handling different object types with different grip 
strengths.
Visualizes:
    - Average % Optimal Action over episodes
    - Reward Table with optimal actions marked
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D

# ================= Environment =================
class PickPlaceEnv:
    """Pick-and-place contextual bandit environment"""
    def __init__(self):
        self.object_types = ["fragile_light", "medium_rigid", "heavy_rigid"]
        self.best_grips = [0, 1, 2]  # low, medium, high

        # Discrete reward table
        self.reward_table = np.array([
            [10, -10, -5],   # fragile_light
            [-10, 5, 0],     # medium_rigid
            [-5, 0, 8]       # heavy_rigid
        ])

    def reset(self):
        """Return a new random object type (state)"""
        self.obj_idx = random.randint(0, 2)
        return self.obj_idx

    def step(self, action):
        """Apply action, return next state, reward, and if optimal"""
        reward = self.reward_table[self.obj_idx, action]
        correct_action = self.best_grips[self.obj_idx]
        optimal = 1 if action == correct_action else 0
        next_state = self.reset()
        return next_state, reward, optimal

    def get_scenario_info(self):
        """Return information about object types, rewards, and optimal actions"""
        return {
            "object_types": self.object_types,
            "reward_table": self.reward_table,
            "best_grips": self.best_grips
        }

# ================= Agents =================
class EpsilonGreedyAgent:
    """Standard Epsilon-Greedy Agent"""
    def __init__(self, n_states, n_actions, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = np.zeros((n_states, n_actions))
        self.counts = np.zeros((n_states, n_actions))

    def select_action(self, state):
        """Select action based on epsilon-greedy strategy"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_values.shape[1])
        return np.argmax(self.q_values[state])

    def update(self, state, action, reward):
        """Update Q-values incrementally"""
        self.counts[state, action] += 1
        alpha = 1 / self.counts[state, action]
        self.q_values[state, action] += alpha * (reward - self.q_values[state, action])

class UCBAgent:
    """Upper Confidence Bound (UCB) Agent"""
    def __init__(self, n_states, n_actions, c=1.0):
        self.c = c
        self.q_values = np.zeros((n_states, n_actions))
        self.counts = np.zeros((n_states, n_actions))
        self.t = 0

    def select_action(self, state):
        """Select action using UCB formula"""
        self.t += 1
        ucb_values = self.q_values[state] + self.c * np.sqrt(
            np.log(self.t + 1) / (self.counts[state] + 1e-5)
        )
        return np.argmax(ucb_values)

    def update(self, state, action, reward):
        """Update Q-values incrementally"""
        self.counts[state, action] += 1
        alpha = 1 / self.counts[state, action]
        self.q_values[state, action] += alpha * (reward - self.q_values[state, action])

class GradientBanditAgent:
    """Gradient Bandit Agent with baseline"""
    def __init__(self, n_states, n_actions, alpha=0.1):
        self.alpha = alpha
        self.n_states = n_states
        self.n_actions = n_actions
        self.reset()

    def reset(self):
        """Reset preferences and baseline reward"""
        self.preferences = np.zeros((self.n_states, self.n_actions))
        self.avg_reward = np.zeros(self.n_states)
        self.t = np.zeros(self.n_states)

    def select_action(self, state):
        """Select action using softmax over preferences"""
        exp_prefs = np.exp(self.preferences[state] - np.max(self.preferences[state]))
        self.probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.n_actions, p=self.probs)

    def update(self, state, action, reward):
        """Update preferences with incremental baseline"""
        self.t[state] += 1
        self.avg_reward[state] += (reward - self.avg_reward[state]) / self.t[state]
        for a in range(self.n_actions):
            if a == action:
                self.preferences[state][a] += self.alpha * (reward - self.avg_reward[state]) * (1 - self.probs[a])
            else:
                self.preferences[state][a] -= self.alpha * (reward - self.avg_reward[state]) * self.probs[a]

# ================= Runner Function =================
def run_one_episode(env, agent, episodes=500):
    """Run one episode of interaction between agent and environment"""
    optimal_actions = np.zeros(episodes)
    state = env.reset()
    for t in range(episodes):
        action = agent.select_action(state)
        next_state, reward, optimal = env.step(action)
        agent.update(state, action, reward)
        optimal_actions[t] = optimal
        state = next_state
    return optimal_actions

# ================= Main Simulation =================
if __name__ == "__main__":
    episodes = 500
    n_runs = 2000
    env = PickPlaceEnv()
    
    agents = {
        "EpsilonGreedy (ε=0.1)": lambda: EpsilonGreedyAgent(n_states=3, n_actions=3, epsilon=0.1),
        "UCB (c=1.0)": lambda: UCBAgent(n_states=3, n_actions=3, c=1.0),
        "Gradient Bandit (α=0.1)": lambda: GradientBanditAgent(n_states=3, n_actions=3, alpha=0.1)
    }

    avg_optimal = {name: np.zeros(episodes) for name in agents}

    print("Running simulation over", n_runs, "runs...")
    for run_idx in range(n_runs):
        for name, agent_factory in agents.items():
            agent = agent_factory()
            optimal = run_one_episode(env, agent, episodes)
            avg_optimal[name] += optimal

    # Average over runs
    for name in agents:
        avg_optimal[name] = (avg_optimal[name] / n_runs) * 100

    # ================= Plotting =================
    info = env.get_scenario_info()
    reward_table = info["reward_table"]
    object_types = info["object_types"]
    best_grips = info["best_grips"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- Left: % Optimal Action ----
    for name, optimal in avg_optimal.items():
        axes[0].plot(optimal, label=name)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("% Optimal Action")
    axes[0].set_title(f"Pick-and-Place Contextual Bandit: {n_runs} runs average")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Right: Discrete Reward Table ----
    reward_values = np.unique(reward_table)
    colors = ["darkred", "red", "orange", "yellow", "lightgreen", "green"][:len(reward_values)]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(reward_values)+1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    reward_indices = np.zeros_like(reward_table)
    for i, val in enumerate(reward_values):
        reward_indices[reward_table == val] = i

    im = axes[1].imshow(reward_indices, cmap=cmap, norm=norm)

    # Annotate rewards
    for i in range(reward_table.shape[0]):
        for j in range(reward_table.shape[1]):
            axes[1].text(j, i, f"{reward_table[i,j]}", ha="center", va="center", color="black", fontsize=12)

    # Mark optimal action with star (offset)
    for i, best in enumerate(best_grips):
        axes[1].text(best + 0.3, i - 0.3, "★", ha="center", va="center", color="blue", fontsize=16)

    axes[1].set_xticks(range(3))
    axes[1].set_xticklabels(["Grip Low", "Grip Medium", "Grip High"])
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(object_types)
    axes[1].set_title("Pick-and-Place Rewards")
    axes[1].set_xlabel("Actions (Grip Strength)")
    axes[1].set_ylabel("States (Object Types)")
    

    # Colorbar for reward values
    cbar = fig.colorbar(im, ax=axes[1], ticks=range(len(reward_values)))
    cbar.ax.set_yticklabels([str(val) for val in reward_values])
    cbar.set_label("Reward Value")

    # Legend for the star
    star_handle = Line2D([0], [0], marker='*', color='w', label='Optimal Action',
                          markerfacecolor='blue', markersize=15)
    axes[1].legend(handles=[star_handle], loc='upper right', fontsize=12, frameon=True)

    plt.tight_layout()
    plt.show()
