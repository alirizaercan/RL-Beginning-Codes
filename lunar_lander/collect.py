import json
import random
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


NUM_EPISODES = 3500
MODEL_PATH = "LunarLander-v3_PPO_ne128_ns1024_b64_e4_cpu_TotalStep9000K"
OUTPUT_FILE = "ep_3500.json"

ACTION_LABELS = {
    0: "do nothing",
    1: "fire left engine",
    2: "fire main engine",
    3: "fire right engine",
}


def make_conversation(obs, action):
    x, y, vx, vy, angle, angular_vel, left_leg, right_leg = obs
    state_str = (
        f"State: [x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
        f"angle={angle:.4f}, angular_vel={angular_vel:.4f}, "
        f"left_leg={left_leg:.1f}, right_leg={right_leg:.1f}]. "
        f"What action should the lander take?"
    )
    action_str = f"Action: {action} ({ACTION_LABELS[action]})."
    return [
        {"from": "human", "value": state_str},
        {"from": "gpt",   "value": action_str},
    ]


env = gym.make("LunarLander-v3")
model = PPO.load(MODEL_PATH, device="cpu")

conversations = []
plot_data = []

for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        conversations.append(make_conversation(obs, a))
        plot_data.append((float(obs[0]), float(obs[1]), a))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

env.close()
print(f"Collected {len(conversations)} samples over {NUM_EPISODES} episodes.")

random.shuffle(conversations)

with open(OUTPUT_FILE, "w") as f:
    json.dump([{"conversations": conv} for conv in conversations], f, indent=2)
print(f"Saved to {OUTPUT_FILE}")

palette = {
    "do nothing":        "black",
    "fire left engine":  "blue",
    "fire main engine":  "red",
    "fire right engine": "green",
}

action_groups = {a: ([], []) for a in range(4)}
for x, y, a in plot_data:
    action_groups[a][0].append(x)
    action_groups[a][1].append(y)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, a in zip(axes, range(4)):
    label = ACTION_LABELS[a]
    xs, ys = action_groups[a]
    ax.scatter(xs, ys, alpha=0.3, s=10, color=palette[label])
    ax.set_title(f"{label} (n={len(xs)})")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

fig.suptitle(f"X-Y Position per Action (normal, {NUM_EPISODES} episodes)", fontsize=14)
plt.tight_layout()
plt.savefig("ep_3500.png", dpi=150)
plt.show()
