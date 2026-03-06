# Run `pip install "gymnasium[box2d]"` for this example.
import gymnasium as gym

# Create our environment - a 2D biped robot that tries to walk forward
env = gym.make("BipedalWalker-v3", render_mode="human")
print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

# Reset environment to start a new episode
observation, info = env.reset()
# observation: what the agent can "see" - body angle, velocities, joint states, ground contacts, lidar, etc.
# info: extra debugging information (usually not needed for basic learning)
print(f"Observation space: {env.observation_space}")
print(f"Sample observation: {env.observation_space.sample()}")

print(f"Starting observation: {observation}")
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

episode_over = False
total_reward = 0.0
step_count = 0

while not episode_over:
    # Choose an action: 4 continuous motor commands in range [-1, 1]
    action = env.action_space.sample()  # Random action for now - real agents will be smarter!
    step_count += 1
    print(f"Step {step_count} action: {action}")

    # Take the action and see what happens
    observation, reward, terminated, truncated, info = env.step(action)

    # reward: positive when moving forward with stable walking, penalties for bad behavior/falling
    # terminated: True if the episode ends naturally (e.g., robot falls)
    # truncated: True if we hit the time limit

    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
