from stable_baselines3 import PPO
from env import LunarLanderEnv

EXTRA_RULES = """
"""

env = LunarLanderEnv(extra_rules=EXTRA_RULES, render_mode="human")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device="cpu"
)

print("Training PPO agent...")
model.learn(total_timesteps=200_000)
model.save("ppo_lunarlander_llm")
print("Model saved!")
env.close()