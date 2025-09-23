import numpy as np
import pickle
import time
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# =============================================================================
# FULL ENVIRONMENT, AGENT CLASSES & HELPER FUNCTION
# =============================================================================

class FootballEnv_v2(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, render_mode="human"):
        super().__init__()
        self.fig, self.ax = None, None
        self.render_mode = render_mode
        self.field_length, self.field_width, self.ball_x = 25.0, 10.0, 5.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.goal_x, self.goal_width = 25.0, 2.0
        self.goal_y_options = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.phase, self.agent_step_size = 1, 0.5
        self.agent_x_options = np.arange(0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.agent_y_options = np.arange(-5.0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.movement_action_space = spaces.Discrete(4)
        self.angles = np.arange(-45, 50, 5)
        self.shooting_action_space = spaces.Discrete(len(self.angles))
        self.action_space = self.shooting_action_space
        obs_low = np.array([0.0, -5.0, -5.0, -5.0, 1]); obs_high = np.array([5.0, 5.0, 5.0, 5.0, 2])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.traj_x_full, self.traj_y_full = [], []

    def _get_obs(self):
        return np.array([self.agent_x, self.agent_y, self.ball_y, self.goal_y_mid, self.phase], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.phase = 1
        self.ball_y_idx = self.np_random.integers(len(self.ball_y_options))
        self.goal_y_idx = self.np_random.integers(len(self.goal_y_options))
        self.ball_y = float(self.ball_y_options[self.ball_y_idx])
        self.goal_y_mid = float(self.goal_y_options[self.goal_y_idx])
        while True:
            self.agent_x = self.np_random.choice(self.agent_x_options)
            self.agent_y = self.np_random.choice(self.agent_y_options)
            if not np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]): break
        self.traj_x_full, self.traj_y_full = [], []
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated = 0.0, False
        if self.phase == 1:
            if action == 0: self.agent_y += self.agent_step_size
            elif action == 1: self.agent_y -= self.agent_step_size
            elif action == 2: self.agent_x -= self.agent_step_size
            elif action == 3: self.agent_x += self.agent_step_size
            reward = -0.5
            if not (0 <= self.agent_x <= 5 and -5 <= self.agent_y <= 5): reward, terminated = -10.0, True
            elif np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]): reward, self.phase = +10.0, 2
        elif self.phase == 2:
            angle_rad = np.deg2rad(self.angles[action]); terminated = True
            y_final = self.ball_y + np.tan(angle_rad) * (self.goal_x - self.ball_x)
            if not (-self.field_width / 2 < y_final < self.field_width / 2): self.outcome, reward = "Out of Bounds!", -10.0
            else:
                goal_y_min, goal_y_max = self.goal_y_mid - self.goal_width/2, self.goal_y_mid + self.goal_width/2
                if goal_y_min <= y_final <= goal_y_max: self.outcome, reward = "Goal!", +5.0
                else: self.outcome, reward = "Missed!", -3.0
            self.traj_x = np.linspace(self.ball_x, self.goal_x, 100)
            self.traj_y = self.ball_y + np.tan(angle_rad) * (self.traj_x - self.ball_x)
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_facecolor("green"); self.ax.set_xlim(0, self.field_length); self.ax.set_ylim(-5, 5)
            self.agent_artist, = self.ax.plot([], [], 'bo', ms=20, label='Agent')
            self.ball_artist, = self.ax.plot([], [], 'ko', ms=10, label='Ball')
            self.traj_line, = self.ax.plot([], [], color='grey', ls=':', lw=2)
            self.ax.axvline(self.ball_x, color='white', ls='-', lw=2)
            self.goal_top_post, = self.ax.plot([], [], 'w', lw=4)
            self.goal_bottom_post, = self.ax.plot([], [], 'w', lw=4)
            self.goal_crossbar, = self.ax.plot([], [], 'w', lw=4)
            self.ax.legend(fontsize='x-large') # CHANGED: Increased legend font size

        # Update dynamic elements
        self.agent_artist.set_data([self.agent_x], [self.agent_y])

        # Goal visualization
        visual_width = 0.5; goal_top = self.goal_y_mid + self.goal_width / 2; goal_bottom = self.goal_y_mid - self.goal_width / 2
        self.goal_top_post.set_data([self.goal_x - visual_width, self.goal_x], [goal_top, goal_top])
        self.goal_bottom_post.set_data([self.goal_x - visual_width, self.goal_x], [goal_bottom, goal_bottom])
        self.goal_crossbar.set_data([self.goal_x - visual_width, self.goal_x - visual_width], [goal_bottom, goal_top])

        # New logic for ball and trajectory
        if len(self.traj_x_full) > 0:
            self.ball_artist.set_data([self.traj_x_full[-1]], [self.traj_y_full[-1]])
            self.traj_line.set_data(self.traj_x_full, self.traj_y_full)
        else:
            self.ball_artist.set_data([self.ball_x], [self.ball_y])
            self.traj_line.set_data([], [])
            
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def animate_shot(self, fps=60):
        dt = 1.0 / fps
        for x, y in zip(self.traj_x, self.traj_y):
            self.traj_x_full.append(x); self.traj_y_full.append(y); self.render(); time.sleep(dt)

    def close(self):
        if self.fig is not None: plt.close(self.fig); self.fig = None

class QLearningAgent:
    def __init__(self, obs_space_dims, n_actions, epsilon=0.0):
        self.n_actions, self.epsilon = n_actions, epsilon; self.Q = np.zeros((*obs_space_dims, self.n_actions))
    def select_action(self, state): return np.argmax(self.Q[state])
    def load(self, path):
        with open(path, "rb") as f: self.Q = pickle.load(f)

class BanditAgent:
    def __init__(self, n_states, n_actions, epsilon=0.0):
        self.n_actions, self.epsilon = n_actions, epsilon; self.Q = np.zeros(n_states + (n_actions,))
    def select_action(self, state): return int(self.Q[state].argmax())
    def load(self, path):
        with open(path, "rb") as f: self.Q = pickle.load(f)

def discretize_state(obs, env):
    agent_x_idx = int(np.round(obs[0] / env.agent_step_size))
    agent_y_idx = int(np.round((obs[1] + 5.0) / env.agent_step_size))
    return (agent_x_idx, agent_y_idx, env.ball_y_idx, env.goal_y_idx)

# =============================================================================
# MAIN TESTING SCRIPT
# =============================================================================
if __name__ == '__main__':
    env = FootballEnv_v2(render_mode="human")
    
    q_agent_obs_dims = (len(env.agent_x_options), len(env.agent_y_options), len(env.ball_y_options), len(env.goal_y_options))
    bandit_agent_obs_dims = (len(env.ball_y_options), len(env.goal_y_options))
    q_agent = QLearningAgent(obs_space_dims=q_agent_obs_dims, n_actions=env.movement_action_space.n)
    bandit_agent = BanditAgent(n_states=bandit_agent_obs_dims, n_actions=env.shooting_action_space.n)

    try:
        q_agent.load("q_learning_policy.pkl")
        bandit_agent.load("bandit_policy.pkl")
        print("✅ Policies loaded successfully! Starting test...")
    except FileNotFoundError:
        print("❌ ERROR: Policy files not found. Please run train_agents_sequentially.py first.")
        exit()

    for episode in range(5):
        obs, _ = env.reset()
        terminated = False
        print(f"\n--- Episode {episode + 1} ---")
        env.render(); time.sleep(1)
        while not terminated:
            if env.phase == 1:
                state = discretize_state(obs, env)
                action = q_agent.select_action(state)
                obs, reward, terminated, _, _ = env.step(action)
            elif env.phase == 2:
                print("Agent reached the ball, now shooting...")
                state = (env.ball_y_idx, env.goal_y_idx)
                action = bandit_agent.select_action(state)
                obs, reward, terminated, _, _ = env.step(action)
                env.animate_shot()
                print(f"Shot Result: {env.outcome}")
            env.render(); time.sleep(0.05)
        time.sleep(2)
    print("\nTest finished. Close the plot window to exit.")
    env.close(); plt.show(block=True)