import numpy as np
import pickle

class QLearningAgent:
    """Tabular Q-learning agent for discrete environments."""

    def __init__(self, obs_space, action_space, alpha=0.1, gamma=0.99, epsilon=0.2):
        """
        obs_space: MultiDiscrete([num_ball_y, num_goal_y])
        action_space: discrete number of actions
        """
        self.n_states = obs_space.nvec
        self.n_actions = action_space.n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table initialized to zeros
        self.Q = np.zeros((*self.n_states, self.n_actions))

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.Q[tuple(state)])

    def update(self, state, action, reward, next_state, done):
        """Q-learning update"""
        best_next = 0 if done else np.max(self.Q[tuple(next_state)])
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[tuple(state)][action]
        self.Q[tuple(state)][action] += self.alpha * td_error

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.Q = pickle.load(f)
