import numpy as np
import pickle
from tqdm import tqdm
from gymnasium import spaces

# =============================================================================
# ENVIRONMENT, AGENT & HELPERS
# =============================================================================

class FootballEnv_v3_Sim:
    """A minimal simulation of the v3 environment for fast, headless training."""
    def __init__(self):
        self.field_width, self.ball_x, self.goal_width = 10.0, 5.0, 2.0
        self.goal_x = 25.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.agent_step_size = 0.5
        self.agent_x_options = np.arange(0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.agent_y_options = np.arange(-5.0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.angles = np.arange(-45, 50, 5)
        self.movement_action_space = spaces.Discrete(4)
        self.shooting_action_space = spaces.Discrete(len(self.angles))
        self.phase1_env = self._create_gym_env()
    
    def _create_gym_env(self):
        from gymnasium import Env
        class Phase1GymEnv(Env):
            def __init__(self, sim):
                super().__init__(); self.sim = sim
                self.action_space = self.sim.movement_action_space
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,))
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed); s = self.sim
                s.ball_y_idx = self.np_random.integers(len(s.ball_y_options))
                s.ball_y = float(s.ball_y_options[s.ball_y_idx])
                s.goal_y_mid = self.np_random.uniform(-4.0, 4.0) 
                s.goal_y_velocity = self.np_random.choice([0.05, -0.05])
                while True:
                    s.agent_x = self.np_random.choice(s.agent_x_options)
                    s.agent_y = self.np_random.choice(s.agent_y_options)
                    if not np.allclose([s.agent_x, s.agent_y], [s.ball_x, s.ball_y]): break
                return s._get_obs(), {}
            def step(self, action):
                s = self.sim
                s.goal_y_mid += s.goal_y_velocity
                if not (-4.0 < s.goal_y_mid < 4.0): s.goal_y_velocity *= -1
                if action == 0: s.agent_y += s.agent_step_size
                elif action == 1: s.agent_y -= s.agent_step_size
                elif action == 2: s.agent_x -= s.agent_step_size
                elif action == 3: s.agent_x += s.agent_step_size
                reward, terminated = -0.5, False
                if not (0 <= s.agent_x <= 5 and -5 <= s.agent_y <= 5): reward, terminated = -10.0, True
                elif np.allclose([s.agent_x, s.agent_y], [s.ball_x, s.ball_y]): reward, terminated = +10.0, True
                return s._get_obs(), reward, terminated, False, {}
        return Phase1GymEnv(self)

    def _get_obs(self):
        return np.array([self.agent_x, self.agent_y, self.ball_y, self.goal_y_mid, self.goal_y_velocity])

    def calculate_shot_reward(self, ball_y, goal_y_mid, goal_y_velocity, action):
        angle_rad = np.deg2rad(self.angles[action])
        y_final = ball_y + np.tan(angle_rad) * (self.goal_x - self.ball_x)
        if not (-self.field_width / 2 < y_final < self.field_width / 2): return -10.0
        predicted_goal_y = goal_y_mid; temp_velocity = goal_y_velocity
        for _ in range(100):
            predicted_goal_y += temp_velocity
            if not (-4.0 < predicted_goal_y < 4.0): temp_velocity *= -1
        goal_min, goal_max = predicted_goal_y - self.goal_width/2, predicted_goal_y + self.goal_width/2
        return +5.0 if goal_min <= y_final <= goal_max else -3.0

# --- UPDATED QLearningAgent with Alpha and Epsilon Decay ---
class QLearningAgent:
    def __init__(self, obs_dims, n_actions, alpha=1.0, gamma=0.99, epsilon=1.0, 
                 alpha_decay=0.99999, min_alpha=0.01,
                 epsilon_decay=0.99999, min_epsilon=0.01):
        self.n_actions, self.gamma = n_actions, gamma
        self.alpha, self.alpha_decay, self.min_alpha = alpha, alpha_decay, min_alpha
        self.epsilon, self.epsilon_decay, self.min_epsilon = epsilon, epsilon_decay, min_epsilon
        self.Q = np.zeros((*obs_dims, self.n_actions))
    def select_action(self, state):
        return np.random.randint(self.n_actions) if np.random.rand() < self.epsilon else np.argmax(self.Q[state])
    def update(self, state, action, reward, next_state, done):
        best_next = 0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
    def decay(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.alpha = max(self.min_alpha, self.alpha * self.alpha_decay)
    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self.Q, f)

# --- UPDATED Discretize Function with Velocity ---
def discretize_state(sim, obs, num_goal_bins, phase):
    agent_x, agent_y, ball_y, goal_y_mid, goal_y_velocity = obs
    agent_x_idx = int(np.round(agent_x / sim.agent_step_size))
    agent_y_idx = int(np.round((agent_y + 5.0) / sim.agent_step_size))
    ball_y_idx = np.argmin(np.abs(sim.ball_y_options - ball_y))
    goal_y_range = (sim.field_width - sim.goal_width)
    pos_in_range = goal_y_mid + (goal_y_range / 2)
    goal_y_bin = np.clip(int((pos_in_range / goal_y_range) * num_goal_bins), 0, num_goal_bins - 1)
    # Convert velocity to a binary index (0 for down, 1 for up)
    velocity_idx = 1 if goal_y_velocity > 0 else 0
    
    if phase == 1:
        return (agent_x_idx, agent_y_idx, ball_y_idx, goal_y_bin, velocity_idx)
    else: # Phase 2
        return (ball_y_idx, goal_y_bin, velocity_idx)

# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================
if __name__ == '__main__':
    sim = FootballEnv_v3_Sim()
    NUM_GOAL_BINS = 20

    # --- Phase 1: Train Movement Agent ---
    p1_obs_dims = (len(sim.agent_x_options), len(sim.agent_y_options), len(sim.ball_y_options), NUM_GOAL_BINS, 2) # Added velocity dim
    q_agent1 = QLearningAgent(obs_dims=p1_obs_dims, n_actions=sim.movement_action_space.n)
    N_PHASE1_EPISODES = 200000 
    print(f"--- Training Phase 1: Movement for {N_PHASE1_EPISODES} episodes ---")
    for _ in tqdm(range(N_PHASE1_EPISODES), desc="Phase 1 Training"):
        obs, _ = sim.phase1_env.reset()
        terminated = False
        while not terminated:
            state = discretize_state(sim, obs, NUM_GOAL_BINS, phase=1)
            action = q_agent1.select_action(state)
            next_obs, reward, terminated, _, _ = sim.phase1_env.step(action)
            next_state = discretize_state(sim, next_obs, NUM_GOAL_BINS, phase=1)
            q_agent1.update(state, action, reward, next_state, terminated)
            obs = next_obs
        q_agent1.decay()
    q_agent1.save("q_policy_phase1_v3.pkl")
    print("Phase 1 training complete.")

    # --- Phase 2: Train Shooting Agent ---
    p2_obs_dims = (len(sim.ball_y_options), NUM_GOAL_BINS, 2) # Added velocity dim
    q_agent2 = QLearningAgent(obs_dims=p2_obs_dims, n_actions=sim.shooting_action_space.n)
    N_PHASE2_EPISODES = 400000
    print(f"\n--- Training Phase 2: Shooting for {N_PHASE2_EPISODES} episodes ---")
    for _ in tqdm(range(N_PHASE2_EPISODES), desc="Phase 2 Training"):
        ball_y_idx = np.random.randint(len(sim.ball_y_options))
        ball_y = sim.ball_y_options[ball_y_idx]
        goal_y_mid = np.random.uniform(-4.0, 4.0)
        goal_y_velocity = np.random.choice([0.05, -0.05])
        
        state = discretize_state(sim, [0,0,ball_y,goal_y_mid,goal_y_velocity], NUM_GOAL_BINS, phase=2)
        action = q_agent2.select_action(state)
        reward = sim.calculate_shot_reward(ball_y, goal_y_mid, goal_y_velocity, action)
        q_agent2.update(state, action, reward, None, done=True)
        q_agent2.decay()
    q_agent2.save("q_policy_phase2_v3.pkl")
    print("Phase 2 training complete.")
    
    print("\n✅ Both agent policies saved successfully!")