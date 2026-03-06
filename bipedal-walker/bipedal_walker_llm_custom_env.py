# Run `pip install "gymnasium[box2d]" ollama stable-baselines3` for this example.
import re

import gymnasium as gym
import ollama
from stable_baselines3 import PPO


class CustomBipedalWalkerLLMEnv(gym.Env):
    """
    Custom environment created from scratch (not a wrapper).
    - State/action transitions come from BipedalWalker-v3
    - Reward is ignored from env and recomputed by LLM every step
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(self, model_name="qwen2.5:7b", render_mode="human", llm_query_interval=10):
        super().__init__()
        self.model_name = model_name
        self.llm_query_interval = max(1, int(llm_query_interval))

        # Internal source env for transitions/state/action/termination
        self.source_env = gym.make("BipedalWalker-v3", render_mode=render_mode)

        # Expose same spaces
        self.action_space = self.source_env.action_space
        self.observation_space = self.source_env.observation_space

        self.step_count = 0
        self.total_llm_reward = 0.0
        self.prev_state = None
        self.last_llm_text = ""

    def reset(self, **kwargs):
        observation, info = self.source_env.reset(**kwargs)
        self.step_count = 0
        self.total_llm_reward = 0.0
        self.prev_state = observation
        self.last_llm_text = ""
        return observation, info

    def step(self, action):
        # Transition from source environment
        next_state, _env_reward, terminated, truncated, info = self.source_env.step(action)
        self.step_count += 1

        # We ignore _env_reward.
        # For speed: call LLM every N steps and use a cheap proxy reward otherwise.
        if self.step_count % self.llm_query_interval == 0 or terminated or truncated:
            llm_reward, llm_text = self._compute_llm_step_reward(
                prev_state=self.prev_state,
                state=next_state,
                action=action,
                terminated=terminated,
                truncated=truncated,
                step_count=self.step_count,
            )
            self.last_llm_text = llm_text
        else:
            llm_reward = self._fast_proxy_reward(next_state, action, terminated)
            llm_text = f"proxy_reward(interval={self.llm_query_interval})"
        self.prev_state = next_state

        self.total_llm_reward += llm_reward

        info["llm_reward"] = llm_reward
        info["llm_text"] = llm_text
        info["last_llm_text"] = self.last_llm_text
        info["llm_total_reward"] = self.total_llm_reward

        return next_state, llm_reward, terminated, truncated, info

    def close(self):
        self.source_env.close()

    def _compute_llm_step_reward(self, prev_state, state, action, terminated, truncated, step_count):
        """
        Ask LLM to compute per-step reward using BipedalWalker reward ideas:
        - forward movement reward
        - fall penalty
        - motor torque penalty
        """
        prev_state_named = self._state_to_named_dict(prev_state)
        state_named = self._state_to_named_dict(state)
        action_named = self._action_to_named_dict(action)

        prompt = (
            "You are a reinforcement learning reward function for BipedalWalker-v3.\n"
            "Compute ONLY ONE step reward as a single float.\n"
            "Use this reward logic (detailed):\n"
            "1) Forward progress reward:\n"
            "   - reward should increase when hull_velocity_x is positive and stable.\n"
            "2) Balance/stability penalty:\n"
            "   - penalize large abs(hull_angle), high hull_angular_velocity, and unstable vertical motion.\n"
            "3) Motor effort penalty:\n"
            "   - penalize high absolute torque usage from 4 actions.\n"
            "4) Contact and gait quality:\n"
            "   - reasonable alternation of leg contacts is good.\n"
            "   - chaotic contacts and extreme joint speeds should be penalized.\n"
            "5) Termination rule:\n"
            "   - if terminated=True because of failure/fall, return a strong negative reward close to -100.\n"
            "6) If truncated=True (time limit), avoid strong failure penalty unless state is clearly unstable.\n"
            "7) Keep output in range [-100, 100].\n"
            "\n"
            "Observation mapping (24 dims):\n"
            "[0] hull_angle, [1] hull_angular_velocity, [2] hull_velocity_x, [3] hull_velocity_y,\n"
            "[4] hip1_joint_angle, [5] hip1_joint_speed, [6] knee1_joint_angle, [7] knee1_joint_speed,\n"
            "[8] leg1_ground_contact, [9] hip2_joint_angle, [10] hip2_joint_speed,\n"
            "[11] knee2_joint_angle, [12] knee2_joint_speed, [13] leg2_ground_contact,\n"
            "[14] lidar_1, [15] lidar_2, [16] lidar_3, [17] lidar_4, [18] lidar_5,\n"
            "[19] lidar_6, [20] lidar_7, [21] lidar_8, [22] lidar_9, [23] lidar_10.\n"
            "Action mapping (4 dims):\n"
            "[0] hip1_torque, [1] knee1_torque, [2] hip2_torque, [3] knee2_torque.\n"
            f"terminated: {terminated}\n"
            f"truncated: {truncated}\n"
            f"step_count: {step_count}\n"
            f"prev_state_named: {prev_state_named}\n"
            f"current_state_named: {state_named}\n"
            f"action_named: {action_named}\n"
            "Return only the float number, no text."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            raw_text = response["message"]["content"].strip()
            reward = self._safe_parse_float(raw_text)
            reward = max(-100.0, min(100.0, reward))
            return reward, raw_text
        except Exception as e:
            return 0.0, f"error: {e}"

    @staticmethod
    def _fast_proxy_reward(state, action, terminated):
        # Quick deterministic fallback for non-LLM steps.
        vx = float(state[2])
        abs_angle = abs(float(state[0]))
        effort = sum(abs(float(x)) for x in action.tolist()) / 4.0
        reward = (10.0 * vx) - (2.0 * abs_angle) - (0.5 * effort)
        if terminated:
            reward -= 50.0
        return max(-100.0, min(100.0, reward))

    @staticmethod
    def _safe_parse_float(text):
        try:
            return float(text)
        except ValueError:
            m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if m:
                try:
                    return float(m.group(0))
                except ValueError:
                    return 0.0
            return 0.0

    @staticmethod
    def _state_to_named_dict(state):
        s = state.tolist() if state is not None else [0.0] * 24
        return {
            "hull_angle": float(s[0]),
            "hull_angular_velocity": float(s[1]),
            "hull_velocity_x": float(s[2]),
            "hull_velocity_y": float(s[3]),
            "hip1_joint_angle": float(s[4]),
            "hip1_joint_speed": float(s[5]),
            "knee1_joint_angle": float(s[6]),
            "knee1_joint_speed": float(s[7]),
            "leg1_ground_contact": float(s[8]),
            "hip2_joint_angle": float(s[9]),
            "hip2_joint_speed": float(s[10]),
            "knee2_joint_angle": float(s[11]),
            "knee2_joint_speed": float(s[12]),
            "leg2_ground_contact": float(s[13]),
            "lidar_1": float(s[14]),
            "lidar_2": float(s[15]),
            "lidar_3": float(s[16]),
            "lidar_4": float(s[17]),
            "lidar_5": float(s[18]),
            "lidar_6": float(s[19]),
            "lidar_7": float(s[20]),
            "lidar_8": float(s[21]),
            "lidar_9": float(s[22]),
            "lidar_10": float(s[23]),
        }

    @staticmethod
    def _action_to_named_dict(action):
        a = action.tolist()
        return {
            "hip1_torque": float(a[0]),
            "knee1_torque": float(a[1]),
            "hip2_torque": float(a[2]),
            "knee2_torque": float(a[3]),
        }


# ------------------------
# PPO train + demo run
# ------------------------

# Train without render for speed
train_env = CustomBipedalWalkerLLMEnv(model_name="qwen2.5:7b", render_mode=None, llm_query_interval=10)
model = PPO("MlpPolicy", train_env, verbose=1, n_steps=64, batch_size=64, n_epochs=1)
model.learn(total_timesteps=256)
train_env.close()

# Evaluate with render
env = CustomBipedalWalkerLLMEnv(model_name="qwen2.5:7b", render_mode="human", llm_query_interval=10)

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")
print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

observation, info = env.reset()
print(f"Starting observation: {observation}")

episode_over = False
step_count = 0
total_reward = 0.0

while not episode_over:
    # Use PPO policy action
    action, _ = model.predict(observation, deterministic=True)
    step_count += 1

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

    print(f"Step {step_count} action: {action} | llm_reward: {reward:.4f}")

print(f"Episode finished! Total LLM reward: {total_reward:.4f}")
print(f"Last LLM raw output: {info.get('llm_text', '')}")

env.close()
