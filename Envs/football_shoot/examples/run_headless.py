import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add parent dir to path

from src.envs.football_shoot_env import FootballShootEnv
import numpy as np

env = FootballShootEnv(render_mode=None)
obs, info = env.reset(seed=42)
print("Initial obs:", obs, "info:", info)

# try all actions once and print rewards
for a in range(env.action_space.n):
    obs, reward, terminated, truncated, info = env.reset(seed=1)  # reset each time to keep new random states if desired
    obs, reward, terminated, truncated, info = env.step(a)
    print(f"action {a} (angle {env.action_to_angle_deg(a):.1f} deg) -> reward {reward}, event {info.get('event')}, final_y {info.get('final_y')}")
