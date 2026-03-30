from env import LunarLanderEnv

MAX_STEPS = 500

env = LunarLanderEnv(render_mode="human")

observation, info = env.reset()
total_reward = 0
episode = 1
step = 0

while True:
    observation, reward, terminated, truncated, info = env.step()
    total_reward += reward
    step += 1

    if terminated or truncated or step >= MAX_STEPS:
        print(f"\nEpisode {episode} | Total Reward: {total_reward:.2f} | Steps: {step}")
        total_reward = 0
        episode += 1
        step = 0
        observation, info = env.reset()

env.close()