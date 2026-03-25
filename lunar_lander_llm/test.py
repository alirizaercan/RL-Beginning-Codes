from env import LunarLanderEnv

env = LunarLanderEnv(render_mode="human")

observation, info = env.reset()
total_reward = 0
episode = 1

while True:
    observation, reward, terminated, truncated, info = env.step()
    total_reward += reward

    if terminated or truncated:
        print(f"\nEpisode {episode} | Total Reward: {total_reward:.2f}")
        total_reward = 0
        episode += 1
        observation, info = env.reset()

env.close()