"""k-armed Bandit Agents for Exploration-Exploitation Comparison 
Multiple agent classes:
    1. Epsilon-Greedy (Greedy for ε=0)
    2. Upper Confidence Bound (UCB)
    3. Gradient Bandit
Simultion numbers are based on the implementation from 
Sutton & Barto's Book."""

import numpy as np
import matplotlib.pyplot as plt
from buffalo_gym.envs.buffalo_gym import BuffaloEnv

# ================= Agent classes =================
class EpsilonGreedyAgent:
    """ Standard Epsilon-Greedy Agent """
    def __init__(self, n_actions, epsilon):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.reset()

    def reset(self):
        self.q_estimates = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)

    def select_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return np.argmax(self.q_estimates)

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

class UCBAgent:
    """ Upper Confidence Bound (UCB) Agent """
    def __init__(self, n_actions, c):
        self.n_actions = n_actions
        self.c = c
        self.reset()

    def reset(self):
        self.q_estimates = np.zeros(self.n_actions)
        self.action_counts = np.zeros(self.n_actions)
        self.t = 0

    def select_action(self):
        self.t += 1
        ucb_values = self.q_estimates + self.c * np.sqrt(
            np.log(self.t + 1) / (self.action_counts + 1e-5)
        )
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        alpha = 1.0 / self.action_counts[action]
        self.q_estimates[action] += alpha * (reward - self.q_estimates[action])

class GradientBanditAgent:
    """ Gradient Bandit Agent with Baseline """
    def __init__(self, n_actions, alpha=0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.reset()

    def reset(self):
        self.preferences = np.zeros(self.n_actions)
        self.avg_reward = 0.0
        self.t = 0

    def select_action(self):
        # Softmax probabilities
        exp_prefs = np.exp(self.preferences - np.max(self.preferences))  
        self.probs = exp_prefs / np.sum(exp_prefs)
        return np.random.choice(self.n_actions, p=self.probs)

    def update(self, action, reward):
        self.t += 1
        self.avg_reward += (reward - self.avg_reward) / self.t  # incremental baseline
        for a in range(self.n_actions):
            if a == action:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * self.probs[a]

# ================= Parameters =================
n_arms = 10
n_steps = 1000
n_runs = 2000

agents = {
    "Greedy (ε=0)": lambda: EpsilonGreedyAgent(n_arms, 0.0),
    "ε-Greedy (ε=0.1)": lambda: EpsilonGreedyAgent(n_arms, 0.1),
    "UCB (c=2)": lambda: UCBAgent(n_arms, 2.0),
    "Gradient Bandit (α=0.1)": lambda: GradientBanditAgent(n_arms, alpha=0.1)
}

avg_rewards = {name: np.zeros(n_steps) for name in agents}
avg_optimal = {name: np.zeros(n_steps) for name in agents}

# ================= Main =================
for run in range(n_runs):
    true_q = np.random.normal(0, 1, n_arms)

    # Initialize environment with offsets set to true_q
    env = BuffaloEnv(
        arms=n_arms,
        optimal_arms=1,
        optimal_mean=0.0,       
        optimal_std=1.0,
        min_suboptimal_mean=-1.0,
        max_suboptimal_mean=1.0,
        suboptimal_std=1.0
    )
    env.offsets[0] = true_q
    env.stds = [1.0] * n_arms

    optimal_arm = np.argmax(true_q)

    for name, agent_policy in agents.items():
        agent = agent_policy()
        agent.reset()
        rewards = np.zeros(n_steps)
        optimal_selected = np.zeros(n_steps)

        for t in range(n_steps):
            action = agent.select_action()
            reward = np.random.normal(true_q[action], 1.0)
            agent.update(action, reward)
            rewards[t] = reward
            optimal_selected[t] = 1 if action == optimal_arm else 0

        avg_rewards[name] += rewards
        avg_optimal[name] += optimal_selected

# ================= Average over runs =================
for name in agents:
    avg_rewards[name] /= n_runs
    avg_optimal[name] = (avg_optimal[name] / n_runs) * 100

# ================= Plot =================
plt.figure(figsize=(14,5))

plt.subplot(1,2,1)
for name, avg in avg_rewards.items():
    plt.plot(avg, label=name)
plt.xlabel("Steps")
plt.ylabel("Average reward")
plt.title("Average Reward (10-armed testbed)")
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
for name, avg in avg_optimal.items():
    plt.plot(avg, label=name)
plt.xlabel("Steps")
plt.ylabel("% Optimal action")
plt.title("Percentage of Optimal Action")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
