import re
import logging
import os
import sys
import gymnasium as gym
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, LoraConfig
from trl import AutoModelForCausalLMWithValueHead

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llamagym.agent import Agent

logging.basicConfig(level=logging.WARNING)


def build_alpaca_prompt(instruction):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


class LunarLanderAgent(Agent):
    def get_system_prompt(self):
        return ""

    def format_observation(self, observation):
        x, y, vx, vy, angle, ang_vel, leg_l, leg_r = observation
        state = (
            f"[x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
            f"angle={angle:.4f}, angular_vel={ang_vel:.4f}, "
            f"left_leg={leg_l:.1f}, right_leg={leg_r:.1f}]"
        )

        instruction = (
            f"State: {state}. What action should the lander take?\n"
            f"Return exactly one line in this format:\nAction: <0|1|2|3>"
        )
        return build_alpaca_prompt(instruction)

    def extract_action(self, response):
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


def test_checkpoint(checkpoint_path, base_model_name="Qwen/Qwen2.5-0.5B-Instruct", num_episodes=10, render=False):
    """Test a trained checkpoint"""
    
    hyperparams = {
        "generate/max_new_tokens": 16,
        "generate/do_sample": True,
        "generate/top_p": 0.9,
        "generate/top_k": 0,
        "generate/temperature": 0.7,
    }

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the base model first
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Load the LoRA adapter weights
    print(f"Loading LoRA adapter from: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    
    # Wrap with value head for the agent
    model = AutoModelForCausalLMWithValueHead(model).to(device)

    # Load tokenizer from checkpoint (should be saved there)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.pretrained_model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    # Create agent (no training needed for testing)
    agent = LunarLanderAgent(
        model,
        tokenizer,
        device,
        {k: v for k, v in hyperparams.items() if k.startswith("generate/")},
        {"batch_size": 1, "mini_batch_size": 1},  # Small batch for testing
    )

    # Create environment
    env_name = "LunarLander-v3"
    if render:
        env = gym.make(env_name, render_mode="human")
    else:
        env = gym.make(env_name)

    episode_rewards = []
    
    print(f"\nTesting for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        while not done:
            action = agent.act(observation)

            if action is None or action not in range(4):
                action = 0

            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step_count}")
        
        # Reset agent episode state without training
        agent.terminate_episode(train=False)

    env.close()
    
    # Print statistics
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)
    
    print(f"\n=== Test Results ===")
    print(f"Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Max Reward: {max_reward:.2f}")
    print(f"Min Reward: {min_reward:.2f}")
    print(f"Success Rate: {sum(1 for r in episode_rewards if r >= 200) / num_episodes * 100:.1f}%")
    
    return episode_rewards


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained LunarLander checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, 
                       help="Path to the checkpoint directory")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                       help="Base model name (default: Qwen/Qwen2.5-0.5B-Instruct)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of test episodes (default: 10)")
    parser.add_argument("--render", action="store_true",
                       help="Render the environment during testing")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path '{args.checkpoint}' does not exist!")
        sys.exit(1)
    
    test_checkpoint(args.checkpoint, args.base_model, args.episodes, args.render)