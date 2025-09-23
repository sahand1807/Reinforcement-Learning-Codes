import numpy as np
from src.envs.football_shoot_env import FootballShootEnv

def test_goal_scoring():
    env = FootballShootEnv(render_mode=None)
    # pick ball in center and goal center 0
    _obs, _ = env.reset(seed=0)
    env._ball_y = 0.0
    env._goal_mid_y = 0.0
    # choose angle 0 -> goes straight, should score because goal covers [-1, +1]
    action = 9  # because action_to_angle_deg(9) = -45 + 5*9 = 0 deg
    obs, reward, terminated, truncated, info = env.step(action)
    assert reward == 5.0
    assert terminated
    assert info["event"] == "scored"
