from huggingface_hub import login
login(token="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")

from datasets import load_dataset, DatasetDict

dataset = load_dataset("json", data_files="ep_3500.json", split="train")

dataset.push_to_hub("Ali2023kosemen/ep_3500")