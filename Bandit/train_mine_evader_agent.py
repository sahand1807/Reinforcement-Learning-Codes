import numpy as np
import pickle
import random

# ================= Parameters =================
n_states = 16  # 4-bit mine configurations
n_actions = 4  # Up, Down, Left, Right
epsilon = 0.1  # exploration
episodes = 50000

# ================= Initialize Q-values =================
q_values = np.zeros((n_states, n_actions))
counts = np.zeros((n_states, n_actions))

# ================= Helper functions =================
def state_to_index(state):
    """Convert 4-bit state to integer index (0-15)"""
    return state[0]*8 + state[1]*4 + state[2]*2 + state[3]

def generate_random_state():
    """Random mine configuration"""
    # randomly place 1-3 mines
    n_mines = random.randint(1,3)
    state = [0,0,0,0]
    positions = random.sample([0,1,2,3], n_mines)
    for pos in positions:
        state[pos] = 1
    return state

def select_action(state_idx):
    """Epsilon-greedy action selection"""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return np.argmax(q_values[state_idx])

def get_reward(state, action):
    """+1 if safe, -1 if mine"""
    return 1 if state[action] == 0 else -1

# ================= Training loop =================
for ep in range(episodes):
    state = generate_random_state()
    state_idx = state_to_index(state)
    action = select_action(state_idx)
    reward = get_reward(state, action)
    
    counts[state_idx, action] += 1
    alpha = 1 / counts[state_idx, action]
    q_values[state_idx, action] += alpha * (reward - q_values[state_idx, action])

# ================= Save trained Q-values =================
with open("trained_mine_values.pkl", "wb") as f:
    pickle.dump(q_values, f)

print("Training completed. Q-values saved to 'trained_mine_values.pkl'.")
