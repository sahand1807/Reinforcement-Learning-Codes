import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class FootballShootEnv(gym.Env):
    """Soccer shooting environment with animated ball trajectory and visible goal."""
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode="human"):
        super().__init__()
        self.render_mode = render_mode

        # --- Field geometry ---
        self.field_length = 25.0
        self.field_width = 10.0
        self.ball_x = 5.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.goal_x = self.field_length
        self.goal_width = 2.0
        self.goal_y_options = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

        # --- Action / Observation spaces ---
        self.angles = np.arange(-45, 50, 5)
        self.action_space = spaces.Discrete(len(self.angles))
        self.observation_space = spaces.MultiDiscrete([len(self.ball_y_options), len(self.goal_y_options)])

        # --- Rendering handles ---
        self.fig = None
        self.ax = None
        self.ball_artist = None
        self.traj_line = None
        self.start_line = None
        self.start_spot = None
        self.traj_x_full = []
        self.traj_y_full = []

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ball_y_idx = self.np_random.integers(len(self.ball_y_options))
        self.goal_y_idx = self.np_random.integers(len(self.goal_y_options))
        self.ball_y = float(self.ball_y_options[self.ball_y_idx])
        self.goal_y_mid = float(self.goal_y_options[self.goal_y_idx])
        self.done = False
        self.traj_x_full = []
        self.traj_y_full = []
        return np.array([self.ball_y_idx, self.goal_y_idx], dtype=np.int32), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset() before stepping again")

        angle_deg = self.angles[action]
        angle_rad = np.deg2rad(angle_deg)
        dx = self.goal_x - self.ball_x
        dy = np.tan(angle_rad) * dx
        y_final = self.ball_y + dy

        reward = 0.0
        terminated = False

        # Ball hits field sides
        if y_final < -self.field_width / 2:
            y_final = -self.field_width / 2
            reward = -10.0
            terminated = True
        elif y_final > self.field_width / 2:
            y_final = self.field_width / 2
            reward = -10.0
            terminated = True
        else:
            goal_y_min = self.goal_y_mid - self.goal_width / 2
            goal_y_max = self.goal_y_mid + self.goal_width / 2
            if goal_y_min <= y_final <= goal_y_max:
                reward = +5.0
                terminated = True
            else:
                reward = -3.0
                terminated = True

        self.done = terminated

        # full trajectory for animation
        self.traj_x = np.linspace(self.ball_x, self.goal_x, 100)
        self.traj_y = self.ball_y + np.tan(angle_rad) * (self.traj_x - self.ball_x)

        # stop trajectory if ball hits sides
        if y_final <= -self.field_width/2 or y_final >= self.field_width/2:
            boundary_index = np.argmax(
                (self.traj_y <= self.field_width/2) & (self.traj_y >= -self.field_width/2) == False
            )
            self.traj_x = self.traj_x[:boundary_index+1]
            self.traj_y = self.traj_y[:boundary_index+1]

        self.traj_x_full = []
        self.traj_y_full = []

        return np.array([self.ball_y_idx, self.goal_y_idx], dtype=np.int32), reward, terminated, False, {}

    def render(self):
        """Render the field, empty goal with posts and crossbar, ball, trajectory, start line, and spot."""
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 5))
            self.ax.set_xlim(0, self.field_length)
            self.ax.set_ylim(-self.field_width/2, self.field_width/2)
            self.ax.set_facecolor("green")

            # Ball artist
            self.ball_artist, = self.ax.plot(self.ball_x, self.ball_y, 'ro', markersize=10)

            # Trajectory line (grey dotted)
            self.traj_line, = self.ax.plot([], [], color='grey', linestyle=':', linewidth=2)

        # --- Remove previous start markers if they exist ---
        if hasattr(self, "start_line") and self.start_line is not None:
            self.start_line.remove()
        if hasattr(self, "start_spot") and self.start_spot is not None:
            self.start_spot.remove()

        # Solid white vertical line at current ball_x
        self.start_line = self.ax.axvline(self.ball_x, color='white', linestyle='-', linewidth=2)
        # White circle at current ball start position (double size)
        self.start_spot = self.ax.plot(
            self.ball_x, self.ball_y, 'wo', markersize=20, markeredgecolor='black'
        )[0]

        # Update ball and trajectory
        if len(self.traj_x_full) > 0:
            self.ball_artist.set_data([self.traj_x_full[-1]], [self.traj_y_full[-1]])
            self.traj_line.set_data(self.traj_x_full, self.traj_y_full)
        else:
            self.ball_artist.set_data([self.ball_x], [self.ball_y])
            self.traj_line.set_data([], [])

        # --- Draw empty goal with thicker visual posts ---
        visual_width = 0.5  # how far posts extend toward ball
        goal_top = self.goal_y_mid + self.goal_width / 2
        goal_bottom = self.goal_y_mid - self.goal_width / 2

        # Remove old goal parts if exist
        for attr in ["goal_top_post", "goal_bottom_post", "goal_cross"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                getattr(self, attr).remove()

        # Top horizontal post (extends from goal line toward ball)
        self.goal_top_post, = self.ax.plot(
            [self.goal_x - visual_width, self.goal_x], [goal_top, goal_top], 'k', linewidth=4, zorder=5
        )
        # Bottom horizontal post
        self.goal_bottom_post, = self.ax.plot(
            [self.goal_x - visual_width, self.goal_x], [goal_bottom, goal_bottom], 'k', linewidth=4, zorder=5
        )
        # Vertical cross connecting left ends of horizontal posts
        self.goal_cross, = self.ax.plot(
            [self.goal_x - visual_width, self.goal_x - visual_width], [goal_bottom, goal_top], 'k', linewidth=4, zorder=5
        )




        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def animate_shot(self, fps=60):
        if self.render_mode != "human":
            return
        dt = 1 / fps
        for x, y in zip(self.traj_x, self.traj_y):
            self.traj_x_full.append(x)
            self.traj_y_full.append(y)
            self.render()
            plt.pause(dt)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.ball_artist = None
            self.traj_line = None
            self.start_line = None
            self.start_spot = None
