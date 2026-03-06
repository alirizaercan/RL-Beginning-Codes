# Run `pip install "gymnasium[box2d]" stable-baselines3` for this example.
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

# Create training environment (no render for faster training)
train_env = gym.make("BipedalWalker-v3")
print(f"Action space: {train_env.action_space}")
print(f"Sample action: {train_env.action_space.sample()}")
print(f"Observation space: {train_env.observation_space}")
print(f"Sample observation: {train_env.observation_space.sample()}")

# Create PPO model
# MlpPolicy is a basic feed-forward policy network
model = PPO("MlpPolicy", train_env, verbose=1)

# Train model
TOTAL_TIMESTEPS = 300_000
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save trained model for long-term reuse
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "bipedal_walker_ppo_minimal"
model.save(str(MODEL_PATH))
print(f"Model saved to: {MODEL_PATH}.zip")

# Close training env
train_env.close()

# Load model from disk for evaluation
loaded_model = PPO.load(str(MODEL_PATH))

# Create evaluation environment with render
env = gym.make("BipedalWalker-v3", render_mode="human")

# Example observation meaning (24 values total):
# [0]  hull_angle
# [1]  hull_angular_velocity
# [2]  hull_velocity_x
# [3]  hull_velocity_y
# [4]  hip1_joint_angle
# [5]  hip1_joint_speed
# [6]  knee1_joint_angle
# [7]  knee1_joint_speed
# [8]  leg1_ground_contact
# [9]  hip2_joint_angle
# [10] hip2_joint_speed
# [11] knee2_joint_angle
# [12] knee2_joint_speed
# [13] leg2_ground_contact
# [14] lidar_1
# [15] lidar_2
# [16] lidar_3
# [17] lidar_4
# [18] lidar_5
# [19] lidar_6
# [20] lidar_7
# [21] lidar_8
# [22] lidar_9
# [23] lidar_10

# Action meaning (4 values total):
# [0] hip1_torque
# [1] knee1_torque
# [2] hip2_torque
# [3] knee2_torque

# Evaluate on multiple episodes
MAX_EVAL_EPISODES = 3
all_rewards = []

for episode in range(1, MAX_EVAL_EPISODES + 1):
    # Reset environment to start a new episode
    observation, info = env.reset()
    print(f"\nEpisode {episode}/{MAX_EVAL_EPISODES} started")
    print(f"Starting observation: {observation}")

    episode_over = False
    total_reward = 0.0
    step_count = 0

    while not episode_over:
        # Use trained PPO policy instead of random action
        action, _state = loaded_model.predict(observation, deterministic=True)
        step_count += 1

        # Take the action and see what happens
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step_count} action: {action} | reward: {reward:.4f}")

        # reward: positive when moving forward with stable walking, penalties for bad behavior/falling
        # terminated: True if the episode ends naturally (e.g., robot falls)
        # truncated: True if we hit the time limit

        total_reward += reward
        episode_over = terminated or truncated

    all_rewards.append(total_reward)
    print(f"Episode {episode} finished! Total reward: {total_reward:.2f}")

average_reward = sum(all_rewards) / len(all_rewards)
print(f"\nAverage reward over {MAX_EVAL_EPISODES} episodes: {average_reward:.2f}")
env.close()
