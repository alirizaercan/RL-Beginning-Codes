import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


NUM_EPISODES = 50
TARGET_PER_ACTION = 500
MODEL_PATH = "LunarLander-v3_PPO_ne128_ns1024_b64_e4_cpu_TotalStep9000K"
ACTION_LABELS = {0: "do nothing", 1: "left engine", 2: "main engine", 3: "right engine"}

env = gym.make("LunarLander-v3")
model = PPO.load(MODEL_PATH, device="cpu")

buckets = {0: [], 1: [], 2: [], 3: []}

ep = 0
while not all(len(b) >= TARGET_PER_ACTION for b in buckets.values()):
    obs, _ = env.reset()
    ep += 1
    while True:
        action, _ = model.predict(obs, deterministic=True)
        a = int(action)
        if len(buckets[a]) < TARGET_PER_ACTION:
            buckets[a].append((*obs, a))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

env.close()
print(f"Collected over {ep} episodes")

records = [sample for bucket in buckets.values() for sample in bucket]


state_cols = ["x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg"]
df = pd.DataFrame(records, columns=state_cols + ["action"])
df["action_label"] = df["action"].map(ACTION_LABELS)

print(df["action_label"].value_counts())

palette = {
    "do nothing": "black",
    "left engine": "blue",
    "main engine": "red",
    "right engine": "green",
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for ax, (label, group) in zip(axes, df.groupby("action_label")):
    ax.scatter(group["x"], group["y"], alpha=0.3, s=10, color=palette[label])
    ax.set_title(f"{label} (n={len(group)})")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

fig.suptitle(f"X-Y Position per Action (balanced, {TARGET_PER_ACTION} each, {ep} episodes)", fontsize=14)
plt.tight_layout()
plt.savefig("xy_balanced.png", dpi=150)
plt.show()