from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from env import LunarLanderEnv

EXTRA_RULES = """
"""

env = TimeLimit(
    LunarLanderEnv(extra_rules=EXTRA_RULES, render_mode="human"),
    max_episode_steps=500
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu"
)

model.learn(total_timesteps=1_000_000)
model.save("ppo_lunarlander_llm")
print("Model saved!")
env.close()