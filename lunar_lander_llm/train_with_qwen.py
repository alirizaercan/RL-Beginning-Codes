import gymnasium as gym
import torch
from stable_baselines3 import PPO
from qwen_policy import QwenActorCriticPolicy


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

env = gym.make("LunarLander-v3")

model = PPO(
    policy=QwenActorCriticPolicy,
    env=env,
    policy_kwargs={
        "qwen_model_path": "./models/hf_model_ep_3500",
    },
    batch_size=4,
    n_steps=128,
    n_epochs=1,
    ent_coef=0.01,
    learning_rate=1e-5,   
    verbose=1,
    device=device
)

total_timesteps = 50_000  
print(f"\nStarting training for {total_timesteps:,} timesteps...")

try:
    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True
    )
    
    model.save("ppo_qwen_finetuned_2")
    print("Model saved!")
    
    print("Saving fine-tuned Qwen model...")
    model.policy.qwen_model.save_pretrained("./models/qwen_ppo_finetuned_2")
    model.policy.qwen_tokenizer.save_pretrained("./models/qwen_ppo_finetuned_2")
    print("Qwen model saved!")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print("OUT OF MEMORY!")
    raise

env.close()
print("Training completed!")
