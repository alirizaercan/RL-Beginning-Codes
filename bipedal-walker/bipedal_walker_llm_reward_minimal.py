# Run `pip install "gymnasium[box2d]" ollama` for this example.
import re

import gymnasium as gym
import ollama


class LLMRewardWrapper(gym.Wrapper):
    # We only change reward logic.
    # Environment reward is ignored.
    def __init__(self, env, model_name="qwen2.5:7b"):
        super().__init__(env)
        self.model_name = model_name
        self.step_count = 0
        self.sum_vx = 0.0
        self.sum_abs_angle = 0.0
        self.last_state = None
        self.last_action = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.step_count = 0
        self.sum_vx = 0.0
        self.sum_abs_angle = 0.0
        self.last_state = observation
        self.last_action = None
        return observation, info

    def step(self, action):
        # Normal environment step
        next_observation, _env_reward, terminated, truncated, info = self.env.step(action)

        # Keep simple episode stats
        self.step_count += 1
        self.sum_vx += float(next_observation[2])          # forward velocity
        self.sum_abs_angle += abs(float(next_observation[0]))  # hull angle
        self.last_state = next_observation
        self.last_action = action

        episode_over = terminated or truncated
        if not episode_over:
            # During episode, reward is always 0
            return next_observation, 0.0, terminated, truncated, info

        # Episode ended -> ask LLM for final reward
        avg_vx = self.sum_vx / self.step_count if self.step_count > 0 else 0.0
        avg_abs_angle = self.sum_abs_angle / self.step_count if self.step_count > 0 else 0.0

        prompt = (
            "You are a reinforcement learning reward function.\n"
            "Environment: BipedalWalker-v3.\n"
            "Higher forward velocity is better. Higher absolute hull angle is worse.\n"
            "Given the episode summary below, return ONLY one float between -100 and 100.\n"
            f"average_forward_velocity: {avg_vx:.6f}\n"
            f"average_absolute_hull_angle: {avg_abs_angle:.6f}\n"
            f"total_steps: {self.step_count}\n"
            f"last_state: {self.last_state.tolist() if self.last_state is not None else []}\n"
            f"last_action: {self.last_action.tolist() if self.last_action is not None else []}\n"
        )

        llm_text = ""
        llm_reward = 0.0

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            llm_text = response["message"]["content"].strip()
            llm_reward = self._safe_parse_float(llm_text)
            llm_reward = max(-100.0, min(100.0, llm_reward))
        except Exception as e:
            llm_text = f"error: {e}"
            llm_reward = 0.0

        # Keep extra info for end-of-episode print
        info["llm_text"] = llm_text
        info["llm_reward"] = llm_reward
        info["avg_vx"] = avg_vx
        info["avg_abs_angle"] = avg_abs_angle

        return next_observation, llm_reward, terminated, truncated, info

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


# Create our environment - a 2D biped robot that tries to walk forward
env = gym.make("BipedalWalker-v3", render_mode="human")
env = LLMRewardWrapper(env, model_name="qwen2.5:7b")

print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")
print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

# If you want only one episode, set MAX_EPISODES = 1
MAX_EPISODES = 3

for episode in range(1, MAX_EPISODES + 1):
    # Reset environment to start a new episode
    observation, info = env.reset()
    print(f"\nEpisode {episode} started")
    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0.0
    step_count = 0

    while not episode_over:
        # Choose an action: 4 continuous motor commands in range [-1, 1]
        action = env.action_space.sample()  # Random action for now
        step_count += 1

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)

        # During episode reward is 0, final step gets LLM reward
        total_reward += reward
        episode_over = terminated or truncated

        print(f"Step {step_count} action: {action} | reward: {reward:.4f}")

    print(f"Episode finished! Total reward: {total_reward:.4f}")
    print(f"LLM raw output: {info.get('llm_text', '')}")
    print(f"LLM reward: {info.get('llm_reward', 0.0):.4f}")
    print(f"Summary -> avg_vx: {info.get('avg_vx', 0.0):.4f}, avg_abs_angle: {info.get('avg_abs_angle', 0.0):.4f}")

env.close()
