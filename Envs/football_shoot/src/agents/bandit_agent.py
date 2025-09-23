import numpy as np

class BanditAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1):
        """
        n_states: tuple/list of discrete sizes for each state dimension, e.g., (3, 5)
        n_actions: number of possible actions
        epsilon: exploration probability
        """
        self.n_actions = n_actions
        self.epsilon = epsilon

        # Q-table: expected reward per state-action
        self.Q = np.zeros(n_states + (n_actions,))
        # Counts for incremental update
        self.N = np.zeros(n_states + (n_actions,), dtype=int)

    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return int(self.Q[state].argmax())

    def update(self, state, action, reward):
        """Incremental update of expected reward"""
        self.N[state + (action,)] += 1
        n = self.N[state + (action,)]
        self.Q[state + (action,)] += (reward - self.Q[state + (action,)]) / n
