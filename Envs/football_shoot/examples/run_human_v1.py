import sys
import os

# add parent directory (so src/ is on the path)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# import the environment
from src.envs.football_shoot_env_v1 import FootballShootEnvPhaseSeparated

# instantiate the env
env = FootballShootEnvPhaseSeparated(render_mode="human")  # or strict_action_space=False if you prefer

# reset
obs, _ = env.reset()

done = False
while not done:
    # sample a random valid action for the current phase
    action = env.action_space.sample()

    # take a step
    obs, reward, terminated, truncated, info = env.step(action)

    # render field + agent
    env.render()

    # animate the ball if we just shot
    env.animate_shot(fps=60)

    done = terminated or truncated

print("Episode ended with reward:", reward)
input("Press Enter to close…")
env.close()
