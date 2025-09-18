import gymnasium as gym 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# --- Discretizer ---
class Discretizer:
    def __init__(self, observation_space, bins=(6, 12, 6, 12)):
        self.bins = bins
        self.obs_low = observation_space.low.copy()
        self.obs_high = observation_space.high.copy()
        self.obs_high[1] = 5
        self.obs_low[1] = -5
        self.obs_high[3] = 5
        self.obs_low[3] = -5
        self.bin_edges = [np.linspace(self.obs_low[i], self.obs_high[i], bins[i]-1) for i in range(len(bins))]

    def __call__(self, obs):
        state = 0
        for i, edges in enumerate(self.bin_edges):
            digitized = np.digitize(obs[i], edges)
            state *= self.bins[i]
            state += digitized
        return state

# --- Main ---
if __name__ == "__main__":
    env = gym.make("CartPole-v1", render_mode='rgb_array')
    disc = Discretizer(env.observation_space)

    # Load policy and rewards
    policy = np.load("policy_esoft_50k_0.02.npy", allow_pickle=True)
    rewards_per_episode = np.load("rewards_esoft_50k_0.02.npy", allow_pickle=True)

    if policy.ndim == 1:
        select_action = lambda s: policy[s]
    else:
        select_action = lambda s: np.random.choice(policy.shape[1], p=policy[s])

    n_episodes_to_show = 6
    episode_frames = []
    episode_rewards = []

    # Collect frames for animation
    for ep in range(n_episodes_to_show):
        obs = env.reset()[0]
        state = disc(obs)
        done = False
        frames = []
        total_reward = 0
        while not done:
            frames.append(env.render())
            action = select_action(state)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = disc(obs)
            total_reward += reward
        episode_frames.append(frames)
        episode_rewards.append(total_reward)

    env.close()

    # --- Figure 1: Animation of episodes ---
    ε = 0.02  # Set epsilon value for title
    fig1, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig1.suptitle(f"Trained (ε={ε}) CartPole Episodes", fontsize=20)
    axes = axes.flatten()
    ims = []

    for i in range(n_episodes_to_show):
        ax = axes[i]
        ax.set_title(f"Episode {i+1}: {episode_rewards[i]*0.02:.2f}s", fontsize=14)
        ax.axis('off')
        im = ax.imshow(episode_frames[i][0])
        ims.append((im, episode_frames[i]))

    axes[-1].axis('off')  # hide unused subplot

    max_frames = max(len(f) for f in episode_frames)
    def update(frame_idx):
        for im, frames in ims:
            idx = min(frame_idx, len(frames)-1)
            im.set_data(frames[idx])
        return [im for im, _ in ims]

    ani = animation.FuncAnimation(fig1, update, frames=max_frames, interval=50, blit=False, repeat=False)

    # --- Figure 2: Learning curve ---
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(rewards_per_episode, label="Episode Reward")
    # plot the moving average
    window = 500
    plt.plot(np.convolve(rewards_per_episode, np.ones(window)/window, mode='valid'), label=f"{window}-Episode Moving Average", color='orange')
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward (Steps)")
    ax2.set_title("Learning Curve of CartPole Agent")
    ax2.grid(True)
    ax2.legend(loc='upper right')

    plt.show()
