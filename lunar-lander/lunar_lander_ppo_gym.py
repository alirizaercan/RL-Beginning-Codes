import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import time
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# ======================
# Config
# ======================
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 2.5e-4
POLICY_LR = 2.5e-4
VALUE_LR = 1e-3          # Value network için ayrı, daha yüksek LR
EPOCHS = 10
BATCH_SIZE = 64
TIMESTEPS = 2048
MAX_UPDATES = 1000
RENDER_EVERY = 1       # eğitim sırasında render açık
ENTROPY_COEF = 0.01
VALUE_COEF = 1.0         # Value loss daha önemli
MAX_VALUE_LOSS = 10.0    # Value loss clipping
MAX_GRAD_NORM = 0.5
ANNEAL_LR = True
USE_VALUE_CLIP = True
PLOT_EVERY = 5
PLOT_LIVE = False      # canlı pencere yerine dosyaya kaydet
PLOT_FILE = "ppo_outputs/ppo_training.png"
LOG_DIR = "ppo_outputs"
LOG_REWARD_FILE = os.path.join(LOG_DIR, "episode_rewards.csv")
LOG_UPDATE_FILE = os.path.join(LOG_DIR, "update_metrics.csv")
LOG_CONFIG_FILE = os.path.join(LOG_DIR, "run_config.json")

# 30 dakika hedef modu
TARGET_30_MIN_MODE = True
FAST_MODE = False
EVAL_RENDER = False
EVAL_EVERY = 0
EVAL_EPISODES = 2

if TARGET_30_MIN_MODE:
    POLICY_LR = 2.5e-4
    VALUE_LR = 1e-3
    EPOCHS = 4
    BATCH_SIZE = 256
    TIMESTEPS = 1024
    MAX_UPDATES = 300
    PLOT_EVERY = 1
    ENTROPY_COEF = 0.01
    VALUE_COEF = 1.0

if FAST_MODE:
    LR = 5e-4
    EPOCHS = 4
    BATCH_SIZE = 256
    TIMESTEPS = 1024
    MAX_UPDATES = 200
    PLOT_EVERY = 1
    EVAL_EVERY = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Actor-Critic Network
# ======================
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.policy = nn.Linear(128, act_dim)
        self.value = nn.Linear(128, 1)
        
        # Better initialization
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
        self.value.bias.data.fill_(0)

    def forward(self, x):
        x = self.shared(x)
        return self.policy(x), self.value(x)

# ======================
# GAE
# ======================
def compute_gae(rewards, values, dones, last_value):
    advantages = []
    gae = 0
    values = np.append(values, last_value)

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return np.array(advantages)

# ======================
# Environment (eğitimde render açık)
# ======================
env = gym.make(ENV_NAME, render_mode="human")
eval_env = gym.make(ENV_NAME, render_mode="human") if EVAL_RENDER else None
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=POLICY_LR)

# ======================
# Plot & Log Setup
# ======================
os.makedirs(LOG_DIR, exist_ok=True)

# Eski logları backup yap ve yeniden başla (sıfırdan eğitim)
if os.path.exists(LOG_REWARD_FILE):
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = LOG_REWARD_FILE.replace(".csv", f"_backup_{backup_time}.csv")
    os.rename(LOG_REWARD_FILE, backup_file)
    print(f"Old reward logs backed up to: {backup_file}")

if os.path.exists(LOG_UPDATE_FILE):
    backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = LOG_UPDATE_FILE.replace(".csv", f"_backup_{backup_time}.csv")
    os.rename(LOG_UPDATE_FILE, backup_file)
    print(f"Old update logs backed up to: {backup_file}")

# Yeni config dosyası yaz
with open(LOG_CONFIG_FILE, "w", encoding="utf-8") as f:
    json.dump(
        {
            "ENV_NAME": ENV_NAME,
            "GAMMA": GAMMA,
            "LAMBDA": LAMBDA,
            "CLIP_EPS": CLIP_EPS,
            "LR": LR,
            "EPOCHS": EPOCHS,
            "BATCH_SIZE": BATCH_SIZE,
            "TIMESTEPS": TIMESTEPS,
            "MAX_UPDATES": MAX_UPDATES,
            "ENTROPY_COEF": ENTROPY_COEF,
            "VALUE_COEF": VALUE_COEF,
            "MAX_GRAD_NORM": MAX_GRAD_NORM,
            "ANNEAL_LR": ANNEAL_LR,
            "USE_VALUE_CLIP": USE_VALUE_CLIP,
            "PLOT_EVERY": PLOT_EVERY,
            "TARGET_30_MIN_MODE": TARGET_30_MIN_MODE,
            "FAST_MODE": FAST_MODE,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        f,
        indent=2,
    )

# Yeni CSV dosyalarını başlat (readable datetime ile)
with open(LOG_REWARD_FILE, "w", encoding="utf-8") as f:
    f.write("local_datetime,update,episode_reward\n")

with open(LOG_UPDATE_FILE, "w", encoding="utf-8") as f:
    f.write("local_datetime,update,policy_loss,value_loss,entropy,approx_kl,clipfrac,avg_reward_10\n")

def log_episode(update, ep_reward):
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(LOG_REWARD_FILE, "a", encoding="utf-8") as f:
        f.write(f"{local_time},{update},{ep_reward}\n")

def log_update(update, policy_loss, value_loss, entropy, approx_kl, clipfrac, avg_reward_10):
    local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(LOG_UPDATE_FILE, "a", encoding="utf-8") as f:
        f.write(
            f"{local_time},{update},{policy_loss},{value_loss},{entropy},{approx_kl},{clipfrac},{avg_reward_10}\n"
        )

# ======================
# Plot Setup
# ======================
fig = None
ax1 = ax2 = ax3 = None
if PLOT_LIVE:
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    ax1.set_title("Episode Reward")
    ax2.set_title("Policy/Value/Entropy Loss")
    ax3.set_title("Approx KL & Clip Fraction")
    ax3.set_xlabel("Update")

reward_history = []
policy_loss_history = []
value_loss_history = []
entropy_history = []
approx_kl_history = []
clipfrac_history = []

def update_plots():
    if PLOT_LIVE:
        ax1.clear()
        ax2.clear()
        ax3.clear()

        ax1.set_title("Episode Reward")
        ax1.plot(reward_history, label="reward")
        ax1.legend()

        ax2.set_title("Policy/Value/Entropy Loss")
        ax2.plot(policy_loss_history, label="policy")
        ax2.plot(value_loss_history, label="value")
        ax2.plot(entropy_history, label="entropy")
        ax2.legend()

        ax3.set_title("Approx KL & Clip Fraction")
        ax3.plot(approx_kl_history, label="approx_kl")
        ax3.plot(clipfrac_history, label="clipfrac")
        ax3.legend()

        ax3.set_xlabel("Update")
        plt.tight_layout()
        plt.pause(0.001)
    else:
        fig_local, (a1, a2, a3) = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        a1.set_title("Episode Reward")
        a1.plot(reward_history, label="reward")
        a1.legend()

        a2.set_title("Policy/Value/Entropy Loss")
        a2.plot(policy_loss_history, label="policy")
        a2.plot(value_loss_history, label="value")
        a2.plot(entropy_history, label="entropy")
        a2.legend()

        a3.set_title("Approx KL & Clip Fraction")
        a3.plot(approx_kl_history, label="approx_kl")
        a3.plot(clipfrac_history, label="clipfrac")
        a3.legend()
        a3.set_xlabel("Update")

        plt.tight_layout()
        fig_local.savefig(PLOT_FILE, dpi=150)
        plt.close(fig_local)

def evaluate_policy(model, device, episodes=2):
    if eval_env is None:
        return None
    returns = []
    for _ in range(episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, _ = model(obs_t)
                action = torch.argmax(logits).item()
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
    return float(np.mean(returns))

# ======================
# Training Loop
# ======================
obs, _ = env.reset()
episode_reward = 0
episode_rewards = []
global_step = 0

for update in range(1, MAX_UPDATES + 1):

    obs_buf, act_buf, rew_buf, done_buf = [], [], [], []
    logp_buf, val_buf = [], []

    # -------- Rollout --------
    for step in range(TIMESTEPS):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)

        with torch.no_grad():
            logits, value = model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action.item())
        rew_buf.append(reward)
        done_buf.append(done)
        logp_buf.append(logp.item())
        val_buf.append(value.item())

        obs = next_obs
        episode_reward += reward
        global_step += 1

        if done:
            print(f"Episode reward: {episode_reward:.2f}")
            episode_rewards.append(episode_reward)
            log_episode(update, episode_reward)
            obs, _ = env.reset()
            episode_reward = 0

        # Eğitim sırasında render açık
        if RENDER_EVERY and update % RENDER_EVERY == 0:
            time.sleep(0.01)

    # -------- Advantage --------
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32).to(device)
        _, last_value_t = model(obs_t)
        last_value = last_value_t.item()

    advantages = compute_gae(rew_buf, val_buf, done_buf, last_value)
    returns = advantages + np.array(val_buf)

    # Only normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    obs_buf = torch.tensor(obs_buf, dtype=torch.float32).to(device)
    act_buf = torch.tensor(act_buf).to(device)
    logp_buf = torch.tensor(logp_buf).to(device)
    adv_buf = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_buf = torch.tensor(returns, dtype=torch.float32).to(device)
    old_val_buf = torch.tensor(val_buf, dtype=torch.float32).to(device)

    if ANNEAL_LR:
        frac = 1.0 - (update - 1.0) / MAX_UPDATES
        lr_now = frac * POLICY_LR
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

    # -------- PPO Update --------
    policy_loss_epoch = 0.0
    value_loss_epoch = 0.0
    entropy_epoch = 0.0
    approx_kl_epoch = 0.0
    clipfrac_epoch = 0.0
    num_minibatches = 0

    for _ in range(EPOCHS):
        idx = np.random.permutation(TIMESTEPS)

        for start in range(0, TIMESTEPS, BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]

            logits, values = model(obs_buf[batch])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_buf[batch])

            log_ratio = logp - logp_buf[batch]
            ratio = torch.exp(log_ratio)
            surr1 = ratio * adv_buf[batch]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_buf[batch]

            policy_loss = -torch.min(surr1, surr2).mean()

            if USE_VALUE_CLIP:
                values_clipped = old_val_buf[batch] + torch.clamp(
                    values.squeeze() - old_val_buf[batch], -CLIP_EPS, CLIP_EPS
                )
                value_loss_unclipped = (ret_buf[batch] - values.squeeze()).pow(2)
                value_loss_clipped = (ret_buf[batch] - values_clipped).pow(2)
                value_loss_raw = torch.max(value_loss_unclipped, value_loss_clipped).mean()
                # Clip extreme value losses
                value_loss = torch.clamp(value_loss_raw, max=MAX_VALUE_LOSS)
            else:
                value_loss_raw = (ret_buf[batch] - values.squeeze()).pow(2).mean()
                value_loss = torch.clamp(value_loss_raw, max=MAX_VALUE_LOSS)

            entropy = dist.entropy().mean()

            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (ratio - 1 - log_ratio).mean().item()
                clipfrac = (torch.abs(ratio - 1.0) > CLIP_EPS).float().mean().item()

            policy_loss_epoch += policy_loss.item()
            value_loss_epoch += value_loss.item()
            entropy_epoch += entropy.item()
            approx_kl_epoch += approx_kl
            clipfrac_epoch += clipfrac
            num_minibatches += 1

    policy_loss_history.append(policy_loss_epoch / max(1, num_minibatches))
    value_loss_history.append(value_loss_epoch / max(1, num_minibatches))
    entropy_history.append(entropy_epoch / max(1, num_minibatches))
    approx_kl_history.append(approx_kl_epoch / max(1, num_minibatches))
    clipfrac_history.append(clipfrac_epoch / max(1, num_minibatches))

    if episode_rewards:
        avg_reward_10 = float(np.mean(episode_rewards[-10:]))
        reward_history.append(avg_reward_10)
    else:
        avg_reward_10 = 0.0
        reward_history.append(0.0)

    if update % PLOT_EVERY == 0:
        update_plots()

    log_update(
        update,
        policy_loss_history[-1],
        value_loss_history[-1],
        entropy_history[-1],
        approx_kl_history[-1],
        clipfrac_history[-1],
        avg_reward_10,
    )

    if EVAL_EVERY and update % EVAL_EVERY == 0:
        eval_return = evaluate_policy(model, device, episodes=EVAL_EPISODES)
        if eval_return is not None:
            print(f"Eval return (avg {EVAL_EPISODES} ep): {eval_return:.2f}")

    print(f"Update {update} completed")

env.close()
if eval_env is not None:
    eval_env.close()
if PLOT_LIVE:
    plt.ioff()
    plt.show()
