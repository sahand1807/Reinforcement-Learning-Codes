import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from racetrack_env import RacetrackEnv  # environment file
from tqdm import tqdm  # progress bar
import time

# ------------------- Map -------------------
example_map = [
    "XXXXXXXXXXXXXXXXXX",
    "XXXX............F.",
    "XXXX............F.",
    "XXXX............F.",
    "XXXX............F.",
    "XXXX.......XXXXXXX",
    "XXXX.......XXXXXXX",
    "X..........XXXXXXX",
    "X...........XXXXXX",
    "XXXX........XXXXXX",
    "XXXXXXX.....XXXXXX",
    "XXXXXXX.....XXXXXX",
    "XXXX........XXXXXX",
    "XXXX........XXXXXX",
    "XXXX........XXXXXX",
    "XXXX........XXXXXX",
    "XXXX........XXXXXX",
    "XXXXSSSSSSSSXXXXXX",
    "XXXXXXXXXXXXXXXXXX",
]

# ------------------- Utilities -------------------
def state_to_key(obs):
    return tuple(int(x) for x in obs)

def valid_actions_for_state(env, obs):
    r, c, vx, vy = obs
    valid = []
    for i, (ax, ay) in enumerate(env.action_list):
        new_vx = int(np.clip(vx + ax, 0, env.max_speed - 1))
        new_vy = int(np.clip(vy + ay, 0, env.max_speed - 1))
        if new_vx == 0 and new_vy == 0 and not env.is_start((r, c)):
            continue
        valid.append(i)
    return valid

def epsilon_greedy_action(env, Q, obs, eps):
    s = state_to_key(obs)
    valid_actions = valid_actions_for_state(env, obs)
    if random.random() < eps or s not in Q:
        return random.choice(valid_actions)
    qvals = Q[s]
    return max(valid_actions, key=lambda a: qvals[a])

def generate_episode(env, Q, eps, human_render=False, pause_time=0.2, current_episode=None):
    episode = []
    obs, _ = env.reset()
    for t in range(1000):
        a = epsilon_greedy_action(env, Q, obs, eps)
        next_obs, r, terminated, truncated, _ = env.step(a)
        episode.append((state_to_key(obs), a, r))
        obs = next_obs
        if human_render:
            env.render(pause_time=pause_time, episode=current_episode)
        if terminated or truncated:
            break
    return episode, len(episode)

# ------------------- Monte Carlo Training -------------------
def mc_control_onpolicy(env, num_episodes=5000, eps=0.1, render_every=200, pause_time=0.2):
    Q = dict()
    N = defaultdict(lambda: np.zeros(env.action_space.n, dtype=int))
    episode_lengths = []

    with tqdm(total=num_episodes, desc="Training Racetrack") as pbar:
        for i in range(1, num_episodes + 1):
            human_render = (i % render_every == 0)
            episode, ep_len = generate_episode(env, Q, eps, human_render, pause_time, current_episode=i)
            episode_lengths.append(ep_len)

            # Monte Carlo update
            G = 0
            for t in reversed(range(len(episode))):
                s_t, a_t, r_t = episode[t]
                G = r_t + G
                if s_t not in Q:
                    Q[s_t] = np.zeros(env.action_space.n)
                N[s_t][a_t] += 1
                Q[s_t][a_t] += (G - Q[s_t][a_t]) / N[s_t][a_t]

            # Update progress bar with average over last 50 episodes
            window = 50
            avg_len = np.mean(episode_lengths[-window:]) if len(episode_lengths) >= window else np.mean(episode_lengths)
            pbar.set_postfix({"avg_steps": f"{avg_len:.2f}"})
            pbar.update(1)

    return Q, episode_lengths

# ------------------- Main -------------------
if __name__ == "__main__":
    env = RacetrackEnv(example_map, max_speed=5)
    Q, episode_lengths = mc_control_onpolicy(env, num_episodes=5000, eps=0.1, render_every=1000, pause_time=0.2)

    # Save Q-values
    np.save("q_values.npy", Q)
    print("Training finished and Q-values saved to q_values.npy")

    # ------------------- Plot learning curve -------------------
    plt.ioff()  # turn off interactive mode for blocking
    plt.figure(figsize=(10,5))
    plt.plot(episode_lengths, label='Episode steps')

    # Moving average starting at episode 0
    window = 50
    smoothed = []
    for i in range(len(episode_lengths)):
        if i < window:
            smoothed.append(np.mean(episode_lengths[:i+1]))
        else:
            smoothed.append(np.mean(episode_lengths[i-window+1:i+1]))
    plt.plot(smoothed, label=f'{window}-episode moving average', color='red')

    plt.xlabel("Episode")
    plt.ylabel("Number of steps")
    plt.title("Racetrack Learning Curve")
    plt.legend()
    plt.grid()
    plt.show(block=True)