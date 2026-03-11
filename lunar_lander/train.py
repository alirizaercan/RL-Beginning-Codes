import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("LunarLander-v3")

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    device='cpu'
)

print("Training PPO agent...")
model.learn(total_timesteps=1000_000)
model.save("ppo_lunarlander")
print("Model saved!")
env.close()