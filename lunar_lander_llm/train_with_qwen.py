import gymnasium as gym
import torch
from stable_baselines3 import PPO
from qwen_policy import QwenActorCriticPolicy


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print("WARNING: Fine-tuning Qwen on CPU will be VERY slow!")

env = gym.make("LunarLander-v3")

model = PPO(
    policy=QwenActorCriticPolicy,
    env=env,
    policy_kwargs={
        "qwen_model_path": "./models/hf_model_ep_3500",
    },
    learning_rate=1e-5,
    n_steps=512,  
    batch_size=16, 
    n_epochs=2,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
    device=device
)

total_timesteps = 100_000 
print(f"Starting training for {total_timesteps:,} timesteps...")

model.learn(
    total_timesteps=total_timesteps,
    progress_bar=True
)

model.save("ppo_qwen_finetuned")

print("\nSaving fine-tuned Qwen model...")
model.policy.qwen_model.save_pretrained("./models/qwen_ppo_finetuned")
model.policy.qwen_tokenizer.save_pretrained("./models/qwen_ppo_finetuned")

env.close()
