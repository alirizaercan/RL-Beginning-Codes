import gymnasium as gym
import numpy as np
import sys
from stable_baselines3 import PPO
from qwen_policy import QwenActorCriticPolicy


if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    model_name = "ppo_qwen_finetuned"

print(f"\nTesting model: {model_name}")
print("=" * 60)

env = gym.make("LunarLander-v3", render_mode="human")

print("Loading trained model...")
try:
    model = PPO.load(
        model_name,
        env=env,
        custom_objects={"policy_class": QwenActorCriticPolicy}
    )
    print(f"Model '{model_name}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: Model '{model_name}.zip' not found!")
    env.close()
    sys.exit(1)

print("\nTesting the model...")
print("=" * 60)

n_episodes = 20
episode_rewards = []

for episode in range(n_episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False
    step = 0
    
    print(f"\nEpisode {episode + 1}/{n_episodes}")
    print("-" * 40)
    
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        
        if step % 50 == 0:
            print(f"Step {step:3d} | Reward so far: {episode_reward:7.2f}")
    
    episode_rewards.append(episode_reward)
    print(f"Episode finished in {step} steps")
    print(f"Final reward: {episode_reward:.2f}")

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
print(f"Model: {model_name}")
print(f"Episodes: {n_episodes}")
print(f"Average reward: {np.mean(episode_rewards):.2f}")
print(f"Std deviation: {np.std(episode_rewards):.2f}")
print(f"Min reward: {np.min(episode_rewards):.2f}")
print(f"Max reward: {np.max(episode_rewards):.2f}")
print("=" * 60)

env.close()
