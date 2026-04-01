import os
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from qwen_policy import QwenActorCriticPolicy


class CheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            step = self.num_timesteps

            self.model.save(f"{self.save_path}/ppo_{step}")

            self.model.policy.qwen_model.save_pretrained(
                f"{self.save_path}/qwen_{step}"
            )
            self.model.policy.qwen_tokenizer.save_pretrained(
                f"{self.save_path}/qwen_{step}"
            )
            print(f"[Checkpoint] {step} timestep saved → {self.save_path}")
        return True


CHECKPOINT_PATH  = "./checkpoints"
TOTAL_TIMESTEPS  = 300_000
SAVE_FREQ        = 1024

RESUME_STEP      = 0     

QWEN_BASE_PATH   = "./models/hf_model_ep_3500" 
device           = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


env = gym.make("LunarLander-v3")

if RESUME_STEP > 0:
    qwen_ckpt = f"{CHECKPOINT_PATH}/qwen_{RESUME_STEP}"
    ppo_ckpt  = f"{CHECKPOINT_PATH}/ppo_{RESUME_STEP}"

    if not os.path.exists(ppo_ckpt + ".zip"):
        raise FileNotFoundError(f"Checkpoint not found: {ppo_ckpt}.zip")
    if not os.path.exists(qwen_ckpt):
        raise FileNotFoundError(f"Qwen checkpoint not found: {qwen_ckpt}")

    print(f"Continue from {RESUME_STEP}. step...")

    model = PPO.load(
        ppo_ckpt,
        env=env,
        device=device,
        custom_objects={
            "policy_kwargs": {
                "qwen_model_path": qwen_ckpt,
            }
        },
    )

else:
    print("Starting from scratch...")

    model = PPO(
        policy=QwenActorCriticPolicy,
        env=env,
        policy_kwargs={
            "qwen_model_path": QWEN_BASE_PATH,
        },
        batch_size=16,
        n_steps=128,
        n_epochs=4,
        learning_rate=1e-5,
        ent_coef=0.01,
        clip_range=0.1,
        verbose=1,
        device=device
    )

try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        reset_num_timesteps=(RESUME_STEP == 0),
        callback=CheckpointCallback(
            save_freq=SAVE_FREQ,
            save_path=CHECKPOINT_PATH,
        ),
        progress_bar=True,
    )

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("Out of memory")
    raise

finally:
    final_step = model.num_timesteps
    model.save(f"{CHECKPOINT_PATH}/ppo_{final_step}")
    model.policy.qwen_model.save_pretrained(f"{CHECKPOINT_PATH}/qwen_{final_step}")
    model.policy.qwen_tokenizer.save_pretrained(f"{CHECKPOINT_PATH}/qwen_{final_step}")
    print(f"{final_step}. step saved")

env.close()