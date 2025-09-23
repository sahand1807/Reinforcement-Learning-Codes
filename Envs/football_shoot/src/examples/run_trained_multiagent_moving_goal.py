import numpy as np
import pickle
import time
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

# =============================================================================
# FULL ENVIRONMENT, AGENT & HELPERS (For visual testing)
# =============================================================================

class FootballEnv_v3(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}
    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode, self.fig, self.ax = render_mode, None, None
        self.field_length, self.field_width, self.ball_x = 25.0, 10.0, 5.0
        self.goal_x, self.goal_width = 25.0, 2.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.phase, self.agent_step_size = 1, 0.5
        self.agent_x_options = np.arange(0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.agent_y_options = np.arange(-5.0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.angles = np.arange(-45, 50, 5)
        self.movement_action_space = spaces.Discrete(4)
        self.shooting_action_space = spaces.Discrete(len(self.angles))
        self.action_space = self.shooting_action_space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.traj_x_full, self.traj_y_full = [], []
        self.goal_y_velocity = 0.05

    def _get_obs(self):
        return np.array([self.agent_x, self.agent_y, self.ball_y, self.goal_y_mid, self.goal_y_velocity])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed); self.phase = 1; self.traj_x_full, self.traj_y_full = [], []
        self.ball_y_idx = self.np_random.integers(len(self.ball_y_options))
        self.ball_y = float(self.ball_y_options[self.ball_y_idx])
        self.goal_y_mid = self.np_random.uniform(-4.0, 4.0)
        self.goal_y_velocity = self.np_random.choice([0.05, -0.05])
        while True:
            self.agent_x = self.np_random.choice(self.agent_x_options)
            self.agent_y = self.np_random.choice(self.agent_y_options)
            if not np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]): break
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated = 0.0, False
        self.goal_y_mid += self.goal_y_velocity
        if not (-4.0 < self.goal_y_mid < 4.0): self.goal_y_velocity *= -1
        if self.phase == 1:
            if action == 0: self.agent_y += self.agent_step_size
            elif action == 1: self.agent_y -= self.agent_step_size
            elif action == 2: self.agent_x -= self.agent_step_size
            elif action == 3: self.agent_x += self.agent_step_size
            reward = -0.5
            if not (0 <= self.agent_x <= 5 and -5 <= self.agent_y <= 5): reward, terminated = -10.0, True
            elif np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]): reward, self.phase = +10.0, 2
        elif self.phase == 2:
            terminated = True; angle_rad = np.deg2rad(self.angles[action])
            y_final = self.ball_y + np.tan(angle_rad) * (self.goal_x - self.ball_x)
            self.traj_x = np.linspace(self.ball_x, self.goal_x, 100)
            self.traj_y = self.ball_y + np.tan(angle_rad) * (self.traj_x - self.ball_x)
            if not (-self.field_width / 2 < y_final < self.field_width / 2): self.outcome, reward = "Out of Bounds!", -10.0
            else:
                predicted_goal_y = self.goal_y_mid; temp_velocity = self.goal_y_velocity
                for _ in range(len(self.traj_x)):
                    predicted_goal_y += temp_velocity
                    if not (-4.0 < predicted_goal_y < 4.0): temp_velocity *= -1
                goal_min, goal_max = predicted_goal_y - self.goal_width/2, predicted_goal_y + self.goal_width/2
                self.outcome, reward = ("Goal!", +5.0) if goal_min <= y_final <= goal_max else ("Missed!", -3.0)
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_facecolor("green"); self.ax.set_xlim(0, self.field_length); self.ax.set_ylim(-5, 5)
            self.agent, = self.ax.plot([], [], 'bo', ms=20, label='Agent'); self.ball, = self.ax.plot([], [], 'ko', ms=10, label='Ball')
            self.traj, = self.ax.plot([], [], 'grey', ls=':', lw=2); self.ax.axvline(self.ball_x, color='white', ls='-', lw=2)
            self.g_top, = self.ax.plot([],[],'w',lw=4); self.g_bot, = self.ax.plot([],[],'w',lw=4); self.g_cross, = self.ax.plot([],[],'w',lw=4)
            self.ax.legend(fontsize='x-large')
        self.agent.set_data([self.agent_x], [self.agent_y])
        if len(self.traj_x_full) > 0: self.ball.set_data([self.traj_x_full[-1]], [self.traj_y_full[-1]]); self.traj.set_data(self.traj_x_full, self.traj_y_full)
        else: self.ball.set_data([self.ball_x], [self.ball_y]); self.traj.set_data([], [])
        g_t, g_b = self.goal_y_mid + self.goal_width/2, self.goal_y_mid - self.goal_width/2
        self.g_top.set_data([self.goal_x-0.5, self.goal_x], [g_t, g_t]); self.g_bot.set_data([self.goal_x-0.5, self.goal_x], [g_b, g_b])
        self.g_cross.set_data([self.goal_x-0.5, self.goal_x-0.5], [g_b, g_t]); self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def animate_shot(self, fps=60):
        if self.render_mode != "human" or self.phase != 2: return
        for x, y in zip(self.traj_x, self.traj_y):
            self.goal_y_mid += self.goal_y_velocity
            if not (-4.0 < self.goal_y_mid < 4.0): self.goal_y_velocity *= -1
            self.traj_x_full.append(x); self.traj_y_full.append(y); self.render(); time.sleep(1.0/fps)

    def close(self):
        if self.fig is not None: plt.close(self.fig); self.fig = None; plt.ioff()

class QLearningAgent:
    def __init__(self, obs_dims, n_actions, epsilon=0.0):
        self.n_actions, self.epsilon = n_actions, epsilon; self.Q = np.zeros((*obs_dims, self.n_actions))
    def select_action(self, state): return np.argmax(self.Q[state])
    def load(self, path):
        with open(path, "rb") as f: self.Q = pickle.load(f)

# --- UPDATED Discretize Function with Velocity ---
def discretize_state(env, obs, num_goal_bins, phase):
    agent_x, agent_y, ball_y, goal_y_mid, goal_y_velocity = obs
    agent_x_idx = int(np.round(agent_x / env.agent_step_size))
    agent_y_idx = int(np.round((agent_y + 5.0) / env.agent_step_size))
    ball_y_idx = np.argmin(np.abs(env.ball_y_options - ball_y))
    goal_y_range = (env.field_width - env.goal_width)
    pos_in_range = goal_y_mid + (goal_y_range / 2)
    goal_y_bin = np.clip(int((pos_in_range / goal_y_range) * num_goal_bins), 0, num_goal_bins - 1)
    velocity_idx = 1 if goal_y_velocity > 0 else 0
    if phase == 1:
        return (agent_x_idx, agent_y_idx, ball_y_idx, goal_y_bin, velocity_idx)
    else: # Phase 2
        return (ball_y_idx, goal_y_bin, velocity_idx)

# =============================================================================
# MAIN TESTING SCRIPT
# =============================================================================
if __name__ == '__main__':
    env = FootballEnv_v3(render_mode="human")
    NUM_GOAL_BINS = 20
    
    # Define Q-table dimensions including the new velocity dimension
    p1_obs_dims = (len(env.agent_x_options), len(env.agent_y_options), len(env.ball_y_options), NUM_GOAL_BINS, 2)
    q_agent1 = QLearningAgent(obs_dims=p1_obs_dims, n_actions=env.movement_action_space.n)
    
    p2_obs_dims = (len(env.ball_y_options), NUM_GOAL_BINS, 2)
    q_agent2 = QLearningAgent(obs_dims=p2_obs_dims, n_actions=env.shooting_action_space.n)

    try:
        q_agent1.load("q_policy_phase1_v3.pkl"); q_agent2.load("q_policy_phase2_v3.pkl")
        print("✅ Policies for v3 loaded successfully! Starting test...")
    except FileNotFoundError:
        print("❌ ERROR: Policy files not found. Please run train_v3.py first.")
        exit()

    for episode in range(5):
        obs, _ = env.reset()
        terminated = False
        print(f"\n--- Episode {episode + 1} ---")
        env.render(); time.sleep(1)
        while not terminated:
            if env.phase == 1:
                state = discretize_state(env, obs, NUM_GOAL_BINS, phase=1)
                action = q_agent1.select_action(state)
            else:
                state = discretize_state(env, obs, NUM_GOAL_BINS, phase=2)
                action = q_agent2.select_action(state)
            obs, reward, terminated, _, _ = env.step(action)
            env.render(); time.sleep(0.05)
            if env.phase == 2 and not terminated: print("Agent reached ball, now shooting...")
        if env.phase == 2:
            env.animate_shot(); print(f"Shot Result: {env.outcome}")
        else:
            print("Agent hit a boundary.")
        time.sleep(2)
        
    print("\nTest finished. Close the plot window to exit.")
    env.close(); plt.show(block=True)