# Run `pip install "gymnasium[box2d]" ollama stable-baselines3` for this example.
from pathlib import Path

from stable_baselines3 import PPO

from lunar_lander_llm_env import CustomLunarLanderLLMEnv


MODEL_NAME = "qwen2.5:3b"
CONTINUOUS_MODE = False
TOTAL_TIMESTEPS = 256
LLM_UPDATE_EVERY = 1
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_PATH = MODEL_DIR / (
    "lunar_lander_llm_ppo_continuous" if CONTINUOUS_MODE else "lunar_lander_llm_ppo_discrete"
)


def train_model():
    MODEL_DIR.mkdir(exist_ok=True)

    env = CustomLunarLanderLLMEnv(
        model_name=MODEL_NAME,
        render_mode=None,
        continuous=CONTINUOUS_MODE,
        llm_update_every=LLM_UPDATE_EVERY,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        device="cpu",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(str(MODEL_PATH))
    env.close()

    print(f"Model saved to: {MODEL_PATH}.zip")
    return MODEL_PATH, CONTINUOUS_MODE


if __name__ == "__main__":
    train_model()
