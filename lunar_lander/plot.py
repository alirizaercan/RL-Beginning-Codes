import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


NUM_EPISODES = 50
MODEL_PATH = "LunarLander-v3_PPO_ne128_ns1024_b64_e4_cpu_TotalStep9000K"
ACTION_LABELS = {0: "do nothing", 1: "left engine", 2: "main engine", 3: "right engine"}

env = gym.make("LunarLander-v3")
model = PPO.load(MODEL_PATH, device="cpu")

records = []
for ep in range(NUM_EPISODES):
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        records.append((*obs, int(action)))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

env.close()


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

fig.suptitle(f"X-Y Position per Action ({NUM_EPISODES} episodes)", fontsize=14)
plt.tight_layout()
plt.savefig("xy.png", dpi=150)
plt.show()