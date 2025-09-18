import gymnasium as gym
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt


class Discretizer:
    def __init__(self, observation_space, bins=(6, 12, 6, 12)):
        self.bins = bins
        self.obs_low = observation_space.low.copy()
        self.obs_high = observation_space.high.copy()
        self.obs_high[1] = 5
        self.obs_low[1] = -5
        self.obs_high[3] = 5
        self.obs_low[3] = -5
        self.bin_edges = [
            np.linspace(self.obs_low[i], self.obs_high[i], bins[i]-1)
            for i in range(len(bins))
        ]

    def __call__(self, obs):
        state = 0
        for i, edges in enumerate(self.bin_edges):
            digitized = np.digitize(obs[i], edges)
            state *= self.bins[i]
            state += digitized
        return state


class MonteCarloESAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.returns = [[[] for _ in range(n_actions)] for _ in range(n_states)]
        self.policy = np.zeros(n_states, dtype=int)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return self.policy[state]

    def update_policy(self):
        self.policy = np.argmax(self.Q, axis=1)

    def update(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                self.returns[s][a].append(G)
                self.Q[s, a] = np.mean(self.returns[s][a])
        self.update_policy()


class MonteCarloESoftAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1, gamma=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        self.returns = [[[] for _ in range(n_actions)] for _ in range(n_states)]
        self.policy = np.ones((n_states, n_actions)) / n_actions

    def choose_action(self, state):
        return np.random.choice(self.n_actions, p=self.policy[state])

    def update(self, episode):
        G = 0
        visited = set()
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = self.gamma * G + r
            if (s, a) not in visited:
                visited.add((s, a))
                self.returns[s][a].append(G)
                self.Q[s, a] = np.mean(self.returns[s][a])
                best_a = np.argmax(self.Q[s])
                for a_i in range(self.n_actions):
                    if a_i == best_a:
                        self.policy[s, a_i] = 1 - self.epsilon + self.epsilon / self.n_actions
                    else:
                        self.policy[s, a_i] = self.epsilon / self.n_actions


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    bins = (6, 12, 6, 12)
    disc = Discretizer(env.observation_space, bins)

    n_states = np.prod(bins)
    n_actions = env.action_space.n

    # === choose your agent here ===
    # agent = MonteCarloESAgent(n_states, n_actions, epsilon=0.1, gamma=1.0)
    agent = MonteCarloESoftAgent(n_states, n_actions, epsilon=0.02, gamma=1.0)

    n_episodes = 50000
    rewards_per_episode = []

    for ep in trange(n_episodes, desc="Training"):
        state = disc(env.reset()[0])
        episode = []
        done = False
        total_reward = 0
        while not done:
            action = agent.choose_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = disc(obs)
            episode.append((state, action, reward))
            state = next_state
            total_reward += reward
        rewards_per_episode.append(total_reward)
        agent.update(episode)

    np.save("policy.npy", agent.policy)
    np.save("rewards.npy", rewards_per_episode)
    print("Training complete. Q-table, policy, and rewards saved.")
