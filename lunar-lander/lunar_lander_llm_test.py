# Run `pip install "gymnasium[box2d]" ollama stable-baselines3` for this example.
from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_llm_env import CustomLunarLanderLLMEnv


MODEL_NAME = "qwen2.5:3b"
CONTINUOUS_MODE = False
MAX_EPISODES = 5
RENDER_MODE = None
LLM_UPDATE_EVERY = 1
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / (
    "lunar_lander_llm_ppo_continuous" if CONTINUOUS_MODE else "lunar_lander_llm_ppo_discrete"
)


def evaluate_model(model_path=MODEL_PATH, continuous_mode=CONTINUOUS_MODE):
    env = CustomLunarLanderLLMEnv(
        model_name=MODEL_NAME,
        render_mode=RENDER_MODE,
        continuous=continuous_mode,
        llm_update_every=LLM_UPDATE_EVERY,
    )
    model = PPO.load(str(model_path), device="cpu")

    print(f"Continuous mode: {continuous_mode}")
    print(f"Action space: {env.action_space}")
    print(f"Sample action: {env.action_space.sample()}")
    print(f"Observation space: {env.observation_space}")
    print(f"Sample observation: {env.observation_space.sample()}")

    episode_rewards = []

    for episode in range(1, MAX_EPISODES + 1):
        observation, info = env.reset()
        print(f"\nEpisode {episode} started")
        print(f"Starting observation: {observation}")

        episode_over = False
        step_count = 0
        total_reward = 0.0

        while not episode_over:
            action, _ = model.predict(observation, deterministic=True)
            step_count += 1

            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            episode_over = terminated or truncated

            print(
                f"Step {step_count} action: {action} | llm_reward: {reward:.4f} | raw: {info.get('llm_text', '')}"
            )

        episode_rewards.append(total_reward)
        print(f"Episode finished! Total LLM reward: {total_reward:.4f}")
        print(f"Last LLM raw output: {info.get('llm_text', '')}")

    average_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
    print(f"\nAverage reward over {MAX_EPISODES} episodes: {average_reward:.4f}")
    env.close()


if __name__ == "__main__":
    evaluate_model()
