import re
from tqdm import trange
import gymnasium as gym

from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import AutoModelForCausalLMWithValueHead
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llamagym.agent import Agent


# ---------------------------------------------------------------------------
# Alpaca prompt helpers — must match supervised training exactly
# ---------------------------------------------------------------------------

STRICT_PREFIX = (
    "Return exactly one line in this format:\n"
    "Action: <0|1|2|3>\n\n"
)


def build_alpaca_prompt(instruction: str) -> str:
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class LunarLanderAgent(Agent):
    """
    PPO agent for LunarLander-v3.

    Observation space (8 floats):
        [0] x position
        [1] y position
        [2] x velocity
        [3] y velocity
        [4] angle
        [5] angular velocity
        [6] left leg contact  (bool)
        [7] right leg contact (bool)

    Action space (4 discrete):
        0 = do nothing
        1 = fire left orientation engine
        2 = fire main engine
        3 = fire right orientation engine
    """

    def get_system_prompt(self) -> str:
        # Alpaca format has no system turn — return empty string.
        # The full instruction is built inside format_observation.
        return ""

    def format_observation(self, observation) -> str:
        x, y, vx, vy, angle, ang_vel, leg_l, leg_r = observation
        state = (
            f"[x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
            f"angle={angle:.4f}, angular_vel={ang_vel:.4f}, "
            f"left_leg={leg_l:.1f}, right_leg={leg_r:.1f}]"
        )
        instruction = STRICT_PREFIX + f"State: {state}. What action should the lander take?"
        return build_alpaca_prompt(instruction)

    def extract_action(self, response: str) -> int:
        # Primary: explicit "Action: N"
        match = re.search(r"[Aa]ction\s*:\s*([0-3])", response)
        if match:
            return int(match.group(1))

        # Fallback: last standalone digit 0-3
        digits = re.findall(r"\b([0-3])\b", response)
        if digits:
            return int(digits[-1])

        # Default: do nothing
        return 0


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    hyperparams = {
        # ---- model ----
        "model_name": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "hf_model_ep_3500"),
        "env": "LunarLander-v3",

        # ---- LoRA ----
        "lora/r": 16,
        "lora/lora_alpha": 32,
        "lora/lora_dropout": 0.05,
        "lora/bias": "none",
        "lora/task_type": "CAUSAL_LM",

        # ---- loading ----
        # Set True if you have >= 10 GB VRAM, False otherwise
        "load_in_8bit": False,

        # ---- PPO ----
        "batch_size": 8,
        "mini_batch_size": 8,
        "seed": 42,
        "episodes": 2000,

        # ---- generation ----
        # max_new_tokens=16 is enough for "Action: 2" responses
        "generate/max_new_tokens": 16,
        "generate/do_sample": True,
        "generate/top_p": 0.9,
        "generate/top_k": 0,
        "generate/temperature": 0.7,
    }

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"

    # --- LoRA config ---
    lora_config = LoraConfig(
        **{
            k.split("/")[-1]: v
            for k, v in hyperparams.items()
            if k.startswith("lora/")
        }
    )
    

    quantization_config = BitsAndBytesConfig(load_in_8bit=False)
    # --- Load model ---
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        pretrained_model_name_or_path=hyperparams["model_name"],
        peft_config=lora_config,
        quantization_config=quantization_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        hyperparams["model_name"],
    )

    # Ensure pad token exists and is different from eos token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.pretrained_model.resize_token_embeddings(len(tokenizer))
        # Update model config to use the new pad token
        model.config.pad_token_id = tokenizer.pad_token_id

    # --- Build agent ---
    agent = LunarLanderAgent(
        model,
        tokenizer,
        device,
        {k: v for k, v in hyperparams.items() if k.startswith("generate/")},
        {
            "batch_size": hyperparams["batch_size"],
            "mini_batch_size": hyperparams["mini_batch_size"],
        },
    )

    # --- Build environment ---
    env = gym.make(hyperparams["env"])

    # --- Training loop ---
    for episode in trange(hyperparams["episodes"]):
        observation, info = env.reset()
        done = False

        while not done:
            action = agent.act(observation)

            # Guard against None or out-of-range actions
            if action is None or action not in range(4):
                action = 0

            observation, reward, terminated, truncated, info = env.step(action)
            agent.assign_reward(reward)
            done = terminated or truncated

        episode_stats = {
            "episode": episode,
            "total_return": sum(agent.current_episode_rewards),
            "message_ct": len(agent.current_episode_messages),
        }

        train_stats = agent.terminate_episode()
        episode_stats.update(train_stats)
        print(episode_stats)

    env.close()