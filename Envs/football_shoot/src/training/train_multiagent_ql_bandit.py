import numpy as np
import pickle
import time
from tqdm import tqdm # Import the progress bar library

# Only import what's needed for training to keep it clean
from gymnasium import spaces

# =============================================================================
# ENVIRONMENT, AGENT CLASSES & HELPERS
# (The definitions are needed to create the objects for training)
# =============================================================================

class FootballEnv_v2:
    """A minimal version of the env for accessing constants and logic."""
    def __init__(self):
        # We only need the constants and logic, not the full Gym setup
        self.field_width, self.ball_x = 10.0, 5.0
        self.ball_y_options = np.array([-3.0, 0.0, 3.0])
        self.goal_x, self.goal_width = 25.0, 2.0
        self.goal_y_options = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        self.agent_step_size = 0.5
        self.agent_x_options = np.arange(0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.agent_y_options = np.arange(-5.0, 5.0 + self.agent_step_size, self.agent_step_size)
        self.movement_action_space = spaces.Discrete(4)
        self.angles = np.arange(-45, 50, 5)
        self.shooting_action_space = spaces.Discrete(len(self.angles))
        
        # We need a separate Gym-compliant env for Phase 1 training
        self.phase1_env = self._create_gym_env()
    
    def _create_gym_env(self):
        # This inner class is a lightweight, Gym-compliant env for Phase 1
        from gymnasium import Env
        class Phase1Env(Env):
            def __init__(self, outer_instance):
                super().__init__()
                self.outer = outer_instance
                self.action_space = self.outer.movement_action_space
                self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.outer.phase = 1
                self.outer.ball_y_idx = self.np_random.integers(len(self.outer.ball_y_options))
                self.outer.goal_y_idx = self.np_random.integers(len(self.outer.goal_y_options))
                self.outer.ball_y = float(self.outer.ball_y_options[self.outer.ball_y_idx])
                while True:
                    self.outer.agent_x = self.np_random.choice(self.outer.agent_x_options)
                    self.outer.agent_y = self.np_random.choice(self.outer.agent_y_options)
                    if not np.allclose([self.outer.agent_x, self.outer.agent_y], [self.outer.ball_x, self.outer.ball_y]):
                        break
                return self.outer._get_obs(), {}

            def step(self, action):
                o = self.outer
                if action == 0: o.agent_y += o.agent_step_size
                elif action == 1: o.agent_y -= o.agent_step_size
                elif action == 2: o.agent_x -= o.agent_step_size
                elif action == 3: o.agent_x += o.agent_step_size
                
                reward, terminated = -0.5, False
                if not (0 <= o.agent_x <= 5 and -5 <= o.agent_y <= 5):
                    reward, terminated = -10.0, True
                elif np.allclose([o.agent_x, o.agent_y], [o.ball_x, o.ball_y]):
                    reward, terminated = +10.0, True # End episode on success
                return o._get_obs(), reward, terminated, False, {}

            def _get_obs(self):
                # Dummy obs, the real state is in the outer instance
                return np.array([self.outer.agent_x, self.outer.agent_y, self.outer.ball_y_idx, self.outer.goal_y_idx, self.outer.phase])

        return Phase1Env(self)

    def _get_obs(self): # Helper for the inner env
        return np.array([self.agent_x, self.agent_y, self.ball_y_idx, self.goal_y_idx, self.phase])

    def calculate_shot_reward(self, ball_y_idx, goal_y_idx, action):
        """Calculates reward for a shot, assuming agent is at the ball."""
        ball_y = self.ball_y_options[ball_y_idx]
        goal_y_mid = self.goal_y_options[goal_y_idx]
        angle_rad = np.deg2rad(self.angles[action])
        
        y_final = ball_y + np.tan(angle_rad) * (self.goal_x - self.ball_x)
        
        if not (-self.field_width / 2 < y_final < self.field_width / 2):
            return -10.0  # Out of bounds
        else:
            goal_y_min = goal_y_mid - self.goal_width / 2
            goal_y_max = goal_y_mid + self.goal_width / 2
            return +5.0 if goal_y_min <= y_final <= goal_y_max else -3.0 # Goal or miss

class QLearningAgent:
    def __init__(self, obs_space_dims, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_actions, self.alpha, self.gamma, self.epsilon = n_actions, alpha, gamma, epsilon
        self.Q = np.zeros((*obs_space_dims, self.n_actions))
    def select_action(self, state):
        return np.random.randint(self.n_actions) if np.random.rand() < self.epsilon else np.argmax(self.Q[state])
    def update(self, state, action, reward, next_state, done):
        best_next = 0 if done else np.max(self.Q[next_state])
        td_target = reward + self.gamma * best_next
        self.Q[state][action] += self.alpha * (td_target - self.Q[state][action])
    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self.Q, f)

class BanditAgent:
    def __init__(self, n_states, n_actions, epsilon=0.1):
        self.n_actions, self.epsilon = n_actions, epsilon
        self.Q = np.zeros(n_states + (n_actions,))
        self.N = np.zeros(n_states + (n_actions,), dtype=int)
    def select_action(self, state):
        return np.random.randint(self.n_actions) if np.random.rand() < self.epsilon else int(self.Q[state].argmax())
    def update(self, state, action, reward):
        self.N[state + (action,)] += 1
        self.Q[state + (action,)] += (reward - self.Q[state + (action,)]) / self.N[state + (action,)]
    def save(self, path):
        with open(path, "wb") as f: pickle.dump(self.Q, f)

def discretize_state(env_instance):
    # Uses indices directly from the environment instance
    agent_x_idx = int(np.round(env_instance.agent_x / env_instance.agent_step_size))
    agent_y_idx = int(np.round((env_instance.agent_y + 5.0) / env_instance.agent_step_size))
    return (agent_x_idx, agent_y_idx, env_instance.ball_y_idx, env_instance.goal_y_idx)

# =============================================================================
# MAIN SEQUENTIAL TRAINING SCRIPT
# =============================================================================
if __name__ == '__main__':
    # --- Setup ---
    master_env = FootballEnv_v2()
    phase1_env = master_env.phase1_env # Use the inner, Gym-compliant env for Phase 1

    q_agent_obs_dims = (len(master_env.agent_x_options), len(master_env.agent_y_options), len(master_env.ball_y_options), len(master_env.goal_y_options))
    bandit_agent_obs_dims = (len(master_env.ball_y_options), len(master_env.goal_y_options))

    q_agent = QLearningAgent(obs_space_dims=q_agent_obs_dims, n_actions=master_env.movement_action_space.n)
    bandit_agent = BanditAgent(n_states=bandit_agent_obs_dims, n_actions=master_env.shooting_action_space.n)

    # --- 1. Train Phase 1 Agent (Q-Learning) ---
    N_PHASE1_EPISODES = 75000
    print(f"--- Training Phase 1: Movement (Q-Learning) for {N_PHASE1_EPISODES} episodes ---")
    
    for episode in tqdm(range(N_PHASE1_EPISODES), desc="Phase 1 Training"):
        obs, _ = phase1_env.reset()
        terminated = False
        while not terminated:
            state = discretize_state(master_env)
            action = q_agent.select_action(state)
            next_obs, reward, terminated, _, _ = phase1_env.step(action)
            next_state = discretize_state(master_env)
            q_agent.update(state, action, reward, next_state, terminated)
    
    print("Phase 1 training complete.")

    # --- 2. Train Phase 2 Agent (Bandit) ---
    N_PHASE2_EPISODES = 20000
    print(f"\n--- Training Phase 2: Shooting (Bandit) for {N_PHASE2_EPISODES} episodes ---")

    for episode in tqdm(range(N_PHASE2_EPISODES), desc="Phase 2 Training"):
        # Simulate a scenario: randomly pick ball and goal positions
        ball_y_idx = np.random.randint(len(master_env.ball_y_options))
        goal_y_idx = np.random.randint(len(master_env.goal_y_options))
        state = (ball_y_idx, goal_y_idx)
        
        # Select action and get reward
        action = bandit_agent.select_action(state)
        reward = master_env.calculate_shot_reward(ball_y_idx, goal_y_idx, action)
        
        # Update the agent
        bandit_agent.update(state, action, reward)
        
    print("Phase 2 training complete.")

    # --- 3. Save Policies ---
    q_agent.save("q_learning_policy.pkl")
    bandit_agent.save("bandit_policy.pkl")
    print("\n✅ Both agent policies saved successfully!")