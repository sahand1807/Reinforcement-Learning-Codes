import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # add parent dir to path
from src.envs.football_shoot_env import FootballShootEnv

env = FootballShootEnv(render_mode="human")
obs, _ = env.reset()

done = False
while not done:
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    env.animate_shot(fps=60)  # ball moves with trajectory
    done = terminated or truncated

print("Episode ended with reward:", reward)
input("Press Enter to close…")
env.close()
