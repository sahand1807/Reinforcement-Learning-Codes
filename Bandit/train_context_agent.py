import numpy as np
import random
import pickle  # for saving trained agent

# ===== Use your existing PickPlaceEnv =====
class PickPlaceEnv:
    def __init__(self):
        self.object_types = ["fragile_light", "medium_rigid", "heavy_rigid"]
        self.best_grips = [0, 1, 2]
        self.reward_table = np.array([
            [10, -10, -5],
            [-10, 5, 0],
            [-5, 0, 8]
        ])

    def reset(self):
        self.obj_idx = random.randint(0, 2)
        return self.obj_idx

    def step(self, action):
        reward = self.reward_table[self.obj_idx, action]
        correct_action = self.best_grips[self.obj_idx]
        optimal = 1 if action == correct_action else 0
        next_state = self.reset()
        return next_state, reward, optimal

# ===== Simple Q-learning / Epsilon-Greedy Agent =====
class EpsilonGreedyAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1):
        self.epsilon = epsilon
        self.q_values = np.zeros((n_states, n_actions))
        self.counts = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.q_values.shape[1])
        return np.argmax(self.q_values[state])

    def update(self, state, action, reward):
        self.counts[state, action] += 1
        alpha = 1 / self.counts[state, action]
        self.q_values[state, action] += alpha * (reward - self.q_values[state, action])

# ===== Training =====
env = PickPlaceEnv()
agent = EpsilonGreedyAgent(n_states=3, n_actions=3, epsilon=0.1)
episodes = 5000

for _ in range(episodes):
    state = env.reset()
    action = agent.select_action(state)
    next_state, reward, optimal = env.step(action)
    agent.update(state, action, reward)

# Save trained Q-values
with open("trained_q_values.pkl", "wb") as f:
    pickle.dump(agent.q_values, f)

print("Training finished and Q-values saved.")
