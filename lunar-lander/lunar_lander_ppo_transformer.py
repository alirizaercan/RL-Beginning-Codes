import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import math

# ======================
# Config
# ======================
ENV_NAME = "LunarLander-v3"
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
EPOCHS = 4
BATCH_SIZE = 64
TIMESTEPS = 512  # Transformer için daha kısa
MAX_UPDATES = 100  # Demo için kısa
SEQUENCE_LENGTH = 16  # Transformer için sequence length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================
# Transformer Actor-Critic
# ======================
class TransformerActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        
        # Input embedding
        self.input_embed = nn.Linear(obs_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.policy = nn.Linear(hidden_dim, act_dim)
        self.value = nn.Linear(hidden_dim, 1)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.policy.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def forward(self, x, mask=None):
        # x: (batch, seq_len, obs_dim)
        x = self.input_embed(x)
        x = self.pos_encoder(x)
        
        # Transformer expects (batch, seq, features)
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Use last timestep for action/value
        x = x[:, -1, :]
        
        logits = self.policy(x)
        value = self.value(x)
        
        return logits, value

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# ======================
# GAE (same as before)
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
# Sequence Buffer
# ======================
class SequenceBuffer:
    def __init__(self, seq_len):
        self.seq_len = seq_len
        self.reset()
    
    def reset(self):
        self.obs_history = []
    
    def add(self, obs):
        self.obs_history.append(obs)
        if len(self.obs_history) > self.seq_len:
            self.obs_history.pop(0)
    
    def get_sequence(self):
        # Pad if needed
        while len(self.obs_history) < self.seq_len:
            self.obs_history.insert(0, np.zeros_like(self.obs_history[0] if self.obs_history else np.zeros(8)))
        return np.array(self.obs_history[-self.seq_len:])

# ======================
# Environment
# ======================
env = gym.make(ENV_NAME)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n

model = TransformerActorCritic(obs_dim, act_dim, hidden_dim=128, num_heads=4, num_layers=2).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"Transformer PPO initialized")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ======================
# Training Loop
# ======================
obs, _ = env.reset()
seq_buffer = SequenceBuffer(SEQUENCE_LENGTH)
seq_buffer.add(obs)
episode_reward = 0
episode_rewards = []

for update in range(1, MAX_UPDATES + 1):
    obs_sequences = []
    act_buf, rew_buf, done_buf = [], [], []
    logp_buf, val_buf = [], []

    # -------- Rollout --------
    for step in range(TIMESTEPS):
        # Get sequence
        obs_seq = seq_buffer.get_sequence()
        obs_sequences.append(obs_seq)
        
        # Forward pass with sequence
        obs_seq_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, value = model(obs_seq_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logp = dist.log_prob(action)

        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        act_buf.append(action.item())
        rew_buf.append(reward)
        done_buf.append(done)
        logp_buf.append(logp.item())
        val_buf.append(value.item())

        obs = next_obs
        seq_buffer.add(obs)
        episode_reward += reward

        if done:
            print(f"Episode reward: {episode_reward:.2f}")
            episode_rewards.append(episode_reward)
            obs, _ = env.reset()
            seq_buffer.reset()
            seq_buffer.add(obs)
            episode_reward = 0

    # -------- Advantage --------
    obs_seq = seq_buffer.get_sequence()
    obs_seq_t = torch.tensor(obs_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        _, last_value_t = model(obs_seq_t)
        last_value = last_value_t.item()

    advantages = compute_gae(rew_buf, val_buf, done_buf, last_value)
    returns = advantages + np.array(val_buf)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Convert to tensors
    obs_seq_buf = torch.tensor(np.array(obs_sequences), dtype=torch.float32).to(device)
    act_buf = torch.tensor(act_buf).to(device)
    logp_buf = torch.tensor(logp_buf).to(device)
    adv_buf = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_buf = torch.tensor(returns, dtype=torch.float32).to(device)

    # -------- PPO Update --------
    for _ in range(EPOCHS):
        idx = np.random.permutation(TIMESTEPS)

        for start in range(0, TIMESTEPS, BATCH_SIZE):
            batch = idx[start:start + BATCH_SIZE]

            logits, values = model(obs_seq_buf[batch])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(act_buf[batch])

            ratio = torch.exp(logp - logp_buf[batch])
            surr1 = ratio * adv_buf[batch]
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_buf[batch]

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (ret_buf[batch] - values.squeeze()).pow(2).mean()
            entropy = dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

    if episode_rewards:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Update {update}/{MAX_UPDATES} - Avg Reward: {avg_reward:.2f}")

env.close()
print(f"\nTransformer PPO training completed!")
print(f"Final avg reward: {np.mean(episode_rewards[-10:]):.2f}" if episode_rewards else "No episodes")
