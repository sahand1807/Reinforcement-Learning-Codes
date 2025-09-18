import numpy as np
import matplotlib.pyplot as plt
from racetrack_env import RacetrackEnv

# ------------------- Example Map --------------------------
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
# ------------------- Racetrack Simulation --------------------------
def state_to_key(obs):
    return tuple(int(x) for x in obs)

def valid_actions_for_state(env, obs):
    r, c, vx, vy = obs
    valid = []
    for i, (ax, ay) in enumerate(env.action_list):
        new_vx = int(np.clip(vx + ax, 0, env.max_speed - 1))
        new_vy = int(np.clip(vy + ay, 0, env.max_speed - 1))
        if new_vx == 0 and new_vy == 0 and not env.is_start((r, c)):
            continue
        valid.append(i)
    return valid

# ------------------- Simulation -------------------
if __name__ == "__main__":
    env = RacetrackEnv(example_map, max_speed=5)
    Q = np.load("q_values.npy", allow_pickle=True).item()

    obs, _ = env.reset()
    done = False
    steps = 0
    while not done and steps < 200:
        s = state_to_key(obs)
        if s in Q:
            valid = valid_actions_for_state(env, obs)
            qvals = Q[s]
            best = max(valid, key=lambda a: qvals[a])
            action = best
        else:
            valid = valid_actions_for_state(env, obs)
            action = np.random.choice(valid)

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()
        plt.pause(0.2)  # slow down for visualization
        steps += 1
        if terminated or truncated:
            done = True

    print("Simulation finished in", steps, "steps.")
    plt.ioff()
    plt.show()
