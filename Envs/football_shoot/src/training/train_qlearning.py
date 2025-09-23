import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ------------------ Add project root to sys.path ------------------ #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # parent of src/
src_path = os.path.join(project_root)  # path to src folder
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ------------------ Imports ------------------ #
from envs.football_shoot_env import FootballShootEnv
from agents.qlearning_agent import QLearningAgent

# ------------------ Hyperparameters ------------------ #
n_episodes = 5000
alpha = 0.1
gamma = 0.99
epsilon = 0.2
moving_avg_window = 100

# ------------------ Create folders ------------------ #
os.makedirs("models", exist_ok=True)

# ------------------ Environment & Agent ------------------ #
env = FootballShootEnv(render_mode=None)
agent = QLearningAgent(env.observation_space, env.action_space,
                       alpha=alpha, gamma=gamma, epsilon=epsilon)

episode_rewards = []

# ------------------ Training Loop ------------------ #
for ep in range(n_episodes):
    state, _ = env.reset()
    done = False
    reward_total = 0

    # single-step episode
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    agent.update(state, action, reward, next_state, terminated)
    reward_total += reward
    episode_rewards.append(reward_total)

# ------------------ Save the full agent object ------------------ #
agent_path = os.path.join("models", "qlearning_agent.pkl")
with open(agent_path, "wb") as f:
    import pickle
    pickle.dump(agent, f)

print(f"Training complete. Agent saved at {agent_path}")

# ------------------ Plot learning curve ------------------ #
plt.figure(figsize=(10,5))
plt.plot(episode_rewards, alpha=0.3, label="Episode Reward")
moving_avg = np.convolve(episode_rewards, np.ones(moving_avg_window)/moving_avg_window, mode='valid')
plt.plot(range(moving_avg_window-1, len(episode_rewards)), moving_avg, color='red', label=f"{moving_avg_window}-episode Moving Avg")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Q-learning Training Progress")
plt.legend()
plt.grid(True)
plt.show()  # block until closed
