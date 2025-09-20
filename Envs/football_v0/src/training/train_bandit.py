import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
import matplotlib.pyplot as plt
from envs.football_shoot_env import FootballShootEnv
from agents.bandit_agent import BanditAgent

# --- Environment ---
env = FootballShootEnv(render_mode=None)

# --- Agent ---
n_states = (len(env.ball_y_options), len(env.goal_y_options))
n_actions = len(env.angles)
agent = BanditAgent(n_states, n_actions, epsilon=0.1)

# --- Training ---
n_train_episodes = 5000
train_rewards = []

for ep in range(n_train_episodes):
    state, _ = env.reset()
    action = agent.select_action(tuple(state))
    _, reward, terminated, _, _ = env.step(action)
    agent.update(tuple(state), action, reward)
    train_rewards.append(reward)

print("Training complete!")

# --- Moving average ---
window = 50  # episodes
moving_avg = np.convolve(train_rewards, np.ones(window)/window, mode='valid')

# --- Plot training rewards ---
plt.figure(figsize=(10,4))
plt.plot(train_rewards, color='blue', linewidth=1, alpha=0.4, label='Episode reward')
plt.plot(range(window-1, n_train_episodes), moving_avg, color='red', linewidth=2, label=f'{window}-episode moving avg')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Bandit Training Rewards per Episode")
plt.legend()
plt.grid(True)
plt.show()

# --- Evaluation ---
n_eval_episodes = 500
eval_rewards = []

for ep in range(n_eval_episodes):
    state, _ = env.reset()
    action = int(agent.Q[tuple(state)].argmax())  # greedy
    _, reward, terminated, _, _ = env.step(action)
    eval_rewards.append(reward)

eval_rewards = np.array(eval_rewards)
avg_reward = eval_rewards.mean()
print(f"Average reward over {n_eval_episodes} evaluation episodes: {avg_reward:.2f}")

# --- Save trained agent ---
os.makedirs("models", exist_ok=True)
with open("models/bandit_agent.pkl", "wb") as f:
    pickle.dump(agent, f)
