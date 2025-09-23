import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import time

class FootballEnv_v3(gym.Env):
    """
    A standalone, two-phase football environment with a MOVING GOAL.
    - The goal moves vertically and bounces off the field edges.
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode
        self.fig, self.ax = None, None
        
        # --- Field & Game Constants ---
        self.field_length, self.field_width, self.ball_x = 25.0, 10.0, 5.0
        self.goal_x, self.goal_width = 25.0, 2.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.goal_y_options = np.array([-2.0, -1.0, 0.0, 1.0, 2.0]) # Used for initial position
        
        # --- Agent & Phase Mechanics ---
        self.phase, self.agent_step_size = 1, 0.5
        self.agent_x_options = np.arange(0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.agent_y_options = np.arange(-5.0, 5.0 + self.agent_step_size, self.agent_step_size)
        
        # --- Action / Observation Spaces ---
        self.angles = np.arange(-45, 50, 5)
        self.movement_action_space = spaces.Discrete(4)
        self.shooting_action_space = spaces.Discrete(len(self.angles))
        self.action_space = self.shooting_action_space
        
        obs_low = np.array([0.0, -5.0, -5.0, -5.0, 1])
        obs_high = np.array([5.0, 5.0, 5.0, 5.0, 2])
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # --- State Tracking ---
        self.traj_x_full, self.traj_y_full = [], []
        self.outcome = ""
        
        # --- NEW: Goal Movement ---
        self.goal_y_velocity = 0.05

    def _get_obs(self):
        return np.array([self.agent_x, self.agent_y, self.ball_y, self.goal_y_mid, self.phase], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = 1
        self.traj_x_full, self.traj_y_full = [], []
        
        self.ball_y_idx = self.np_random.integers(len(self.ball_y_options))
        self.goal_y_idx = self.np_random.integers(len(self.goal_y_options))
        self.ball_y = float(self.ball_y_options[self.ball_y_idx])
        self.goal_y_mid = float(self.goal_y_options[self.goal_y_idx])

        # --- NEW: Randomize starting direction of goal movement ---
        self.goal_y_velocity = np.random.choice([0.05, -0.05])

        while True:
            self.agent_x = self.np_random.choice(self.agent_x_options)
            self.agent_y = self.np_random.choice(self.agent_y_options)
            if not np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]):
                break
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated = 0.0, False
        
        # --- NEW: Update goal position each step ---
        self.goal_y_mid += self.goal_y_velocity
        goal_top_edge = self.goal_y_mid + self.goal_width / 2
        goal_bottom_edge = self.goal_y_mid - self.goal_width / 2

        # Reverse direction if goal hits the boundary
        if goal_top_edge >= self.field_width / 2 or goal_bottom_edge <= -self.field_width / 2:
            self.goal_y_velocity *= -1

        if self.phase == 1:
            if action == 0: self.agent_y += self.agent_step_size
            elif action == 1: self.agent_y -= self.agent_step_size
            elif action == 2: self.agent_x -= self.agent_step_size
            elif action == 3: self.agent_x += self.agent_step_size
            
            reward = -0.5
            if not (0 <= self.agent_x <= 5 and -5 <= self.agent_y <= 5):
                reward, terminated = -10.0, True
            elif np.allclose([self.agent_x, self.agent_y], [self.ball_x, self.ball_y]):
                reward, self.phase = +10.0, 2
        
        elif self.phase == 2:
            terminated = True
            angle_rad = np.deg2rad(self.angles[action])
            y_final = self.ball_y + np.tan(angle_rad) * (self.goal_x - self.ball_x)
            
            if not (-self.field_width / 2 < y_final < self.field_width / 2):
                self.outcome, reward = "Out of Bounds!", -10.0
            else:
                goal_y_min, goal_y_max = self.goal_y_mid - self.goal_width/2, self.goal_y_mid + self.goal_width/2
                if goal_y_min <= y_final <= goal_y_max:
                    self.outcome, reward = "Goal!", +5.0
                else:
                    self.outcome, reward = "Missed!", -3.0

            self.traj_x = np.linspace(self.ball_x, self.goal_x, 100)
            self.traj_y = self.ball_y + np.tan(angle_rad) * (self.traj_x - self.ball_x)

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        # Rendering code is unchanged from the previous version
        if self.fig is None:
            plt.ion(); self.fig, self.ax = plt.subplots(figsize=(12, 6))
            self.ax.set_facecolor("green"); self.ax.set_xlim(0, self.field_length); self.ax.set_ylim(-5, 5)
            self.agent_artist, = self.ax.plot([], [], 'bo', ms=20, label='Agent')
            self.ball_artist, = self.ax.plot([], [], 'ko', ms=10, label='Ball')
            self.traj_line, = self.ax.plot([], [], color='grey', ls=':', lw=2)
            self.ax.axvline(self.ball_x, color='white', ls='-', lw=2)
            self.goal_top_post, = self.ax.plot([], [], 'w', lw=4); self.goal_bottom_post, = self.ax.plot([], [], 'w', lw=4)
            self.goal_crossbar, = self.ax.plot([], [], 'w', lw=4); self.ax.legend(fontsize='x-large')
        self.agent_artist.set_data([self.agent_x], [self.agent_y])
        if len(self.traj_x_full) > 0:
            self.ball_artist.set_data([self.traj_x_full[-1]], [self.traj_y_full[-1]])
            self.traj_line.set_data(self.traj_x_full, self.traj_y_full)
        else:
            self.ball_artist.set_data([self.ball_x], [self.ball_y]); self.traj_line.set_data([], [])
        visual_width = 0.5; goal_top = self.goal_y_mid + self.goal_width / 2; goal_bottom = self.goal_y_mid - self.goal_width / 2
        self.goal_top_post.set_data([self.goal_x - visual_width, self.goal_x], [goal_top, goal_top])
        self.goal_bottom_post.set_data([self.goal_x - visual_width, self.goal_x], [goal_bottom, goal_bottom])
        self.goal_crossbar.set_data([self.goal_x - visual_width, self.goal_x - visual_width], [goal_bottom, goal_top])
        self.fig.canvas.draw(); self.fig.canvas.flush_events()

    def animate_shot(self, fps=60):
        if self.render_mode != "human" or self.phase != 2: return
        dt = 1.0 / fps
        for x, y in zip(self.traj_x, self.traj_y):
            self.traj_x_full.append(x); self.traj_y_full.append(y); self.render(); time.sleep(dt)

    def close(self):
        if self.fig is not None: plt.close(self.fig); self.fig = None; plt.ioff()

# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':
    env = FootballEnv_v3(render_mode="human")
    for episode in range(5):
        obs, info = env.reset()
        env.render(); print(f"\n{'='*10} Episode {episode + 1} {'='*10}")
        terminated = False
        while not terminated:
            action = env.movement_action_space.sample() if env.phase == 1 else env.shooting_action_space.sample()
            obs, reward, terminated, _, _ = env.step(action)
            if env.phase == 1: print(f"Move Reward: {reward:.1f}", end='\r')
            env.render(); time.sleep(0.05)
        if env.phase == 2:
            print("\nAgent reached ball and shot!"); print("Animating shot..."); env.animate_shot(); print(f"Shot Result: {env.outcome}")
        else: print("\nAgent hit a boundary.")
        time.sleep(2)
    print("\nTest finished. Close the plot window to exit.")
    env.close(); plt.show(block=True)