# Run `pip install "gymnasium[box2d]" ollama stable-baselines3` for this example.
import re

import gymnasium as gym
import ollama

from lunar_lander_env import LunarLander


class CustomLunarLanderLLMEnv(gym.Env):
    """
    Custom LunarLander environment.
    - State/action/termination come from LunarLander physics
    - Reward is ignored from env and recomputed by LLM
    - PPO can train on either discrete or continuous action mode
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(
        self,
        model_name="qwen2.5:3b",
        render_mode=None,
        continuous=False,
        llm_update_every=1,
    ):
        super().__init__()
        self.model_name = model_name
        self.continuous = continuous
        self.llm_update_every = max(1, int(llm_update_every))

        # This is our project-local LunarLander physics environment.
        self.source_env = LunarLander(render_mode=render_mode, continuous=continuous)

        self.action_space = self.source_env.action_space
        self.observation_space = self.source_env.observation_space

        self.step_count = 0
        self.total_llm_reward = 0.0
        self.last_llm_text = ""

    def reset(self, **kwargs):
        observation, info = self.source_env.reset(**kwargs)
        self.step_count = 0
        self.total_llm_reward = 0.0
        self.last_llm_text = ""
        return observation, info

    def step(self, action):
        # Transition comes from physics env, but reward is ignored.
        next_state, _env_reward, terminated, truncated, info = self.source_env.step(action)
        self.step_count += 1

        should_query_llm = (
            self.step_count % self.llm_update_every == 0 or terminated or truncated
        )
        if should_query_llm:
            llm_reward, llm_text = self._compute_llm_step_reward(
                state=next_state,
                action=action,
                terminated=terminated,
                truncated=truncated,
                step_count=self.step_count,
            )
            self.last_llm_text = llm_text
        else:
            llm_reward = 0.0
            llm_text = f"waiting_for_llm(step_interval={self.llm_update_every})"

        self.total_llm_reward += llm_reward

        info["llm_reward"] = llm_reward
        info["llm_text"] = llm_text
        info["last_llm_text"] = self.last_llm_text
        info["llm_total_reward"] = self.total_llm_reward
        return next_state, llm_reward, terminated, truncated, info

    def close(self):
        self.source_env.close()

    def _compute_llm_step_reward(self, state, action, terminated, truncated, step_count):
        state_list = state.tolist() if state is not None else [0.0] * 8
        action_value = action.tolist() if self.continuous else int(action)

        prompt = (
            "You are a reinforcement learning reward function for LunarLander-v3.\n"
            "Compute ONLY ONE step reward as a single float.\n"
            "Reward rules:\n"
            "- If the current state looks better for landing, increase reward clearly.\n"
            "- If the current state looks worse for landing, decrease reward clearly.\n"
            "- closer/slower to landing pad => higher reward\n"
            "- further/faster from landing pad => lower reward\n"
            "- more tilted => lower reward\n"
            "- +10 for each leg in contact with the ground\n"
            "- -0.03 if a side engine fires\n"
            "- -0.30 if the main engine fires\n"
            "- -100 for crashing\n"
            "- +100 for landing safely\n"
            "Goal: cumulative episode reward >= 200.\n"
            f"terminated: {terminated}\n"
            f"truncated: {truncated}\n"
            f"step_count: {step_count}\n"
            f"state: {state_list}\n"
            f"action: {action_value}\n"
            "Return only the float number with 4 decimal places, no text."
        )

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
            )
            raw_text = response["message"]["content"].strip()
            reward = self._safe_parse_float(raw_text)
            reward = max(-100.0, min(100.0, reward))
            return reward, raw_text
        except Exception as error:
            return 0.0, f"error: {error}"

    @staticmethod
    def _safe_parse_float(text):
        try:
            return float(text)
        except ValueError:
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return 0.0
            return 0.0
