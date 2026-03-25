from huggingface_hub import login
login(token="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")

from datasets import load_dataset, DatasetDict

dataset = load_dataset("json", data_files="270_reward.json", split="train")

dataset.push_to_hub("Ali2023kosemen/lunar_lander_270_reward")