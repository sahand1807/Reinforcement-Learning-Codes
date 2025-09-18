import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import colors
import gymnasium as gym
from gymnasium import spaces

# ------------------- Graphical Racetrack Environment -------------------------

class RacetrackEnv(gym.Env):
    """
    Graphical racetrack environment (matplotlib) for RL experiments.
    Observation: (row, col, vx, vy)
    Action: Discrete(9) mapping to (ax, ay) in {-1,0,1}^2
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, track_map, max_speed=5):
        super().__init__()
        self.grid, self.start_positions, self.finish_positions = self.parse_map(track_map)
        self.h, self.w = self.grid.shape
        self.max_speed = max_speed

        # actions: acceleration deltas
        self.action_list = [(ax, ay) for ax in (-1, 0, 1) for ay in (-1, 0, 1)]
        self.action_space = spaces.Discrete(len(self.action_list))

        # observation: (row, col, vx, vy)
        self.observation_space = spaces.MultiDiscrete([self.h, self.w, max_speed, max_speed])

        # state
        self.pos = None
        self.vx = 0
        self.vy = 0

        # matplotlib figure
        self.fig, self.ax = None, None

    def parse_map(self, track_map):
        h = len(track_map)
        w = len(track_map[0])
        grid = np.zeros((h, w), dtype=int)
        start_positions = []
        finish_positions = []

        for r, line in enumerate(track_map):
            for c, ch in enumerate(line):
                if ch == 'X':
                    grid[r, c] = 1  # wall
                elif ch == '.':
                    grid[r, c] = 0  # free
                elif ch == 'S':
                    grid[r, c] = 2
                    start_positions.append((r, c))
                elif ch == 'F':
                    grid[r, c] = 3
                    finish_positions.append((r, c))
                else:
                    raise ValueError("Unknown map char: " + ch)
        return grid, start_positions, finish_positions

    def is_start(self, pos):
        return pos in self.start_positions

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.pos = random.choice(self.start_positions)
        self.vx, self.vy = 0, 0
        return np.array([self.pos[0], self.pos[1], self.vx, self.vy], dtype=int), {}

    def path_cells(self, start, end):
        r0, c0 = start
        r1, c1 = end
        dr = r1 - r0
        dc = c1 - c0
        steps = max(abs(dr), abs(dc), 1)
        cells = []
        for i in range(1, steps + 1):
            t = i / steps
            rr = int(round(r0 + dr * t))
            cc = int(round(c0 + dc * t))
            cells.append((rr, cc))
        return cells

    def step(self, action):
        ax, ay = self.action_list[action]
        new_vx = int(np.clip(self.vx + ax, 0, self.max_speed - 1))
        new_vy = int(np.clip(self.vy + ay, 0, self.max_speed - 1))

        r, c = self.pos
        new_r = r - new_vy  # vy positive moves up
        new_c = c + new_vx  # vx positive moves right

        new_r = int(new_r)
        new_c = int(new_c)

        projected_cells = self.path_cells(self.pos, (new_r, new_c))

        terminated = False
        hit_wall = False
        reached_finish = False

        for rr, cc in projected_cells:
            if rr < 0 or rr >= self.h or cc < 0 or cc >= self.w:
                hit_wall = True
                break
            tile = self.grid[rr, cc]
            if tile == 1:
                hit_wall = True
                break
            if tile == 3:
                reached_finish = True
                break

        reward = -1

        if reached_finish:
            terminated = True
            self.pos = (rr, cc)
        elif hit_wall:
            self.pos = random.choice(self.start_positions)
            self.vx, self.vy = 0, 0
        else:
            self.pos = (new_r, new_c)
            self.vx, self.vy = new_vx, new_vy

        obs = np.array([self.pos[0], self.pos[1], self.vx, self.vy], dtype=int)
        return obs, reward, terminated, False, {}

    def render(self, mode="human", pause_time=0.3, episode=None):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()  # interactive mode

        self.ax.clear()

        # color map: 0-free, 1-wall, 2-start, 3-finish
        cmap = colors.ListedColormap(['white', 'black', 'green', 'red'])
        self.ax.imshow(self.grid, cmap=cmap)

        # plot car
        r, c = self.pos
        self.ax.plot(c, r, 'bo', markersize=12)  # blue circle for car

        # draw grid
        self.ax.set_xticks(np.arange(-0.5, self.grid.shape[1], 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.grid.shape[0], 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # dynamic episode text
        if episode is not None:
            self.ax.text(0, -1, f"Episode: {episode}", fontsize=12, color='blue', ha='left', va='center')

        plt.pause(pause_time)
        plt.draw()

# ------------------- Example Map ---------------------------------------------
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

# ------------------- Demo Run -----------------------------------------------
if __name__ == "__main__":
    env = RacetrackEnv(example_map, max_speed=5)
    obs, _ = env.reset()

    done = False
    steps = 0
    while not done and steps < 50:
        # naive random valid action
        r, c, vx, vy = obs
        valid_actions = []
        for i, (ax, ay) in enumerate(env.action_list):
            new_vx = np.clip(vx + ax, 0, env.max_speed-1)
            new_vy = np.clip(vy + ay, 0, env.max_speed-1)
            if new_vx == 0 and new_vy == 0 and not env.is_start((r,c)):
                continue
            valid_actions.append(i)
        a = random.choice(valid_actions)

        obs, reward, terminated, truncated, info = env.step(a)
        env.render()
        steps += 1
        if terminated:
            done = True
    print("Episode finished in", steps, "steps.")
    plt.ioff()
    plt.show()
