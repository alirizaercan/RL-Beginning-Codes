import json
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from pathlib import Path
from collections import defaultdict


MODEL_PATH      = "3_layer"  
OUTPUT_FILE     = "test_data.json"
TARGET_SAMPLES  = 100_000               
MIN_EP_REWARD   = 260                   
SEED            = 42

SAMPLES_PER_ACTION = TARGET_SAMPLES // 4  

OBS_RANGES = {
    "x":           (-2.5, 2.5),
    "y":           (-2.5, 2.5),
    "vx":          (-10.0, 10.0),
    "vy":          (-10.0, 10.0),
    "angle":       (-6.2831855, 6.2831855),  
    "angular_vel": (-10.0, 10.0),
    "left_leg":    (0.0, 1.0),
    "right_leg":   (0.0, 1.0),
}

OBS_BINS = {
    "x": 10,           
    "y": 10,           
    "vx": 10,          
    "vy": 10,          
    "angle": 10,       
    "angular_vel": 10, 
    "left_leg": 2,     
    "right_leg": 2,    
}

ACTION_DESCRIPTIONS = {
    0: "do nothing",
    1: "fire left engine",
    2: "fire main engine",
    3: "fire right engine",
}

OBS_KEYS = [
    "x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg",
]


def obs_to_text(obs):
    parts = [f"{k}={v:.4f}" for k, v in zip(OBS_KEYS, obs)]
    return "State: [" + ", ".join(parts) + "]. What action should the lander take?"


def action_to_text(action):
    return f"Action: {action} ({ACTION_DESCRIPTIONS[action]})."

def discretize_obs(obs):
    bins = []
    
    for i, key in enumerate(OBS_KEYS):
        if key in ["left_leg", "right_leg"]:
            bins.append(int(obs[i])) 
        else:
            min_val, max_val = OBS_RANGES[key]
            num_bins = OBS_BINS[key]
            
            normalized = (obs[i] - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0.0, 0.999999) 
            
            bin_idx = int(normalized * num_bins)
            bin_idx = np.clip(bin_idx, 0, num_bins - 1)
            
            bins.append(bin_idx)
    
    return tuple(bins)


def collect_raw_data(min_reward, seed, max_episodes=1000):
    env   = gym.make("LunarLander-v3")
    model = PPO.load(MODEL_PATH, device="cpu")

    raw_data = []
    episode = 0
    episode_buf = []
    episode_reward = 0.0

    obs, _ = env.reset(seed=seed)

    while episode < max_episodes:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward

        episode_buf.append({
            "obs": obs.copy(),
            "action": action,
            "obs_bins": discretize_obs(obs)
        })

        obs = next_obs

        if terminated or truncated:
            episode += 1

            if episode_reward >= min_reward:
                raw_data.extend(episode_buf)

            if episode % 5 == 0:
                print(f"Episode {episode:4d} | reward={episode_reward:8.2f} | total_samples={len(raw_data):6,d}")

            episode_buf = []
            episode_reward = 0.0
            obs, _ = env.reset()

    env.close()
    print(f"\nHam veri toplandı: {len(raw_data):,} örnek")
    return raw_data


def balance_dataset(raw_data, target_samples):
    action_groups = defaultdict(list)
    for sample in raw_data:
        action_groups[sample["action"]].append(sample)
    
    print("\nAksiyon dağılımı (ham):")
    for action in sorted(action_groups.keys()):
        print(f"  Aksiyon {action}: {len(action_groups[action]):,} örnek")
    
    samples_per_action = target_samples // 4
    balanced_data = []
    
    for action in range(4):
        if action not in action_groups:
            print(f"UYARI: Aksiyon {action} için veri yok!")
            continue
            
        samples = action_groups[action]
        
        obs_bins_seen = set()
        diverse_samples = []
        diverse_indices = set()
        
        for idx, sample in enumerate(samples):
            obs_bin = sample["obs_bins"]
            if obs_bin not in obs_bins_seen:
                diverse_samples.append(sample)
                diverse_indices.add(idx)
                obs_bins_seen.add(obs_bin)
        
        if len(diverse_samples) < samples_per_action:
            remaining = [samples[i] for i in range(len(samples)) if i not in diverse_indices]
            diverse_samples.extend(remaining[:samples_per_action - len(diverse_samples)])
        
        if len(diverse_samples) >= samples_per_action:
            selected = np.random.choice(len(diverse_samples), samples_per_action, replace=False)
            balanced_data.extend([diverse_samples[i] for i in selected])
        else:
            selected = np.random.choice(len(diverse_samples), samples_per_action, replace=True)
            balanced_data.extend([diverse_samples[i] for i in selected])
        
        print(f"  Aksiyon {action}: {samples_per_action:,} örnek seçildi ({len(diverse_samples):,} benzersiz durum)")
    
    return balanced_data


def convert_to_dataset(balanced_data):
    dataset = []
    for sample in balanced_data:
        dataset.append({
            "conversations": [
                {"from": "human", "value": obs_to_text(sample["obs"])},
                {"from": "gpt",   "value": action_to_text(sample["action"])},
            ]
        })
    return dataset


def save(dataset, path):
    out = Path(path)
    out.write_text(json.dumps(dataset, ensure_ascii=False, indent=2))
    print(f"\nDataset kaydedildi: {path}")


def analyze_balance(balanced_data):
    print("\n=== Denge Analizi ===")
    
    action_counts = defaultdict(int)
    for sample in balanced_data:
        action_counts[sample["action"]] += 1
    
    print("\nAksiyon dağılımı (dengeli):")
    for action in sorted(action_counts.keys()):
        print(f"  Aksiyon {action}: {action_counts[action]:,} örnek")
    
    for i, key in enumerate(OBS_KEYS):
        bins = defaultdict(int)
        for sample in balanced_data:
            bins[sample["obs_bins"][i]] += 1
        
        print(f"\n{key} dağılımı ({OBS_BINS[key]} kategori):")
        for bin_idx in sorted(bins.keys()):
            print(f"  Kategori {bin_idx}: {bins[bin_idx]:,} örnek")


if __name__ == "__main__":
    np.random.seed(SEED)
    
    raw_data = collect_raw_data(MIN_EP_REWARD, SEED, max_episodes=2000)
    
    balanced_data = balance_dataset(raw_data, TARGET_SAMPLES)
    
    analyze_balance(balanced_data)
    
    dataset = convert_to_dataset(balanced_data)
    np.random.shuffle(dataset) 
    save(dataset, OUTPUT_FILE)
