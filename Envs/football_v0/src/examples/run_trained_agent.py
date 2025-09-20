import sys
import os
import time
import pickle

# ------------------ Add project root to sys.path ------------------ #
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))  # parent of src/
src_path = os.path.join(project_root)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ------------------ Imports ------------------ #
from envs.football_shoot_env import FootballShootEnv
from agents.qlearning_agent import QLearningAgent

# ------------------ Load environment with human render ------------------ #
env = FootballShootEnv(render_mode="human")

# ------------------ Load trained agent ------------------ #
#agent_path = os.path.join("models", "qlearning_agent.pkl")
agent_path = os.path.join("models", "bandit_agent.pkl")
with open(agent_path, "rb") as f:
    agent = pickle.load(f)

# ------------------ Initialize figure ONCE ------------------ #
env.fig = None
env.ax = None
env.render()  # creates the figure and axes

# ------------------ Run multiple episodes on same figure ------------------ #
n_episodes = 5
for ep in range(n_episodes):
    state, _ = env.reset()
    done = False

    # Clear previous trajectory line but keep ball and goal
    env.traj_line.set_data([], [])

    while not done:
        # Greedy action
        action = int(agent.Q[tuple(state)].argmax())
        state, reward, terminated, truncated, info = env.step(action)

        # Update ball and trajectory on same figure
        env.render()
        env.animate_shot(fps=60)

        done = terminated or truncated

    print(f"Episode {ep+1} finished with reward: {reward}")
    time.sleep(1)  # short pause before next episode

input("Press Enter to close environment...")
env.close()
