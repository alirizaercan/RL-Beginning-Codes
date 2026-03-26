import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

model = PPO(
    "MlpPolicy",
    env,
    n_steps=1024,
    batch_size=64,
    n_epochs=4,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    verbose=1,
    device='cpu'
)

model.learn(total_timesteps=3_000_000)
model.save("hf")
print("Model saved!")
env.close()