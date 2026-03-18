import gymnasium as gym
import torch.nn as nn
from stable_baselines3 import PPO

env = gym.make("LunarLander-v3")

policy_kwargs = dict(
    net_arch=[64, 64, 64],
    activation_fn=nn.SiLU
)

model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device='cpu'
)

model.learn(total_timesteps=2_000_000)
model.save("3_layer")
print("Model saved!")
env.close()