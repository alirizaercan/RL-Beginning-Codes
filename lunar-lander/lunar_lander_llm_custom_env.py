# Run `pip install "gymnasium[box2d]" ollama stable-baselines3` for this example.
from lunar_lander_llm_test import evaluate_model
from lunar_lander_llm_train import train_model


if __name__ == "__main__":
    model_path, continuous_mode = train_model()
    evaluate_model(model_path=model_path, continuous_mode=continuous_mode)
