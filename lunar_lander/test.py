import gymnasium as gym
from stable_baselines3 import PPO


env = gym.make("LunarLander-v3", render_mode="human")

model = PPO.load("LunarLander-v3_PPO_ne128_ns1024_b64_e4_cpu_TotalStep9000K", device="cpu")

observation, info = env.reset()

total_reward = 0
episode = 1

while True:
    action, _states = model.predict(observation, deterministic=True)

    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"\nEpisode {episode} | Total Reward: {total_reward:.2f}")
        total_reward = 0
        episode += 1
        observation, info = env.reset()

env.close()