---
license: apache-2.0
base_model: Qwen/Qwen2.5-0.5B
library_name: transformers
pipeline_tag: text-generation
tags:
  - qwen
  - transformers
  - unsloth
  - trl
  - peft
  - lora
  - lunar-lander
  - reinforcement-learning
  - action-prediction
---

# qwen2.5-0.5b-lunarlander-action-lora

Hugging Face repos:
- LoRA repo: https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-lora
- GGUF repo: https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-gguf

## Model Summary

This repository contains a LoRA fine-tuned version of `Qwen/Qwen2.5-0.5B` for the LunarLander action prediction task.

The model is trained to map a LunarLander state prompt to an action-style answer such as:

```text
Action: 0 (do nothing).
```

## Base Model

- Base model: `Qwen/Qwen2.5-0.5B`
- Fine-tuning method: Supervised Fine-Tuning (SFT)
- PEFT method: LoRA
- Training framework: Unsloth + TRL

## Dataset

- Dataset name: `Ali2023kosemen/lunarlander`
- Dataset format: ShareGPT-style `conversations`
- Example structure:

```json
{
  "conversations": [
    {
      "from": "human",
      "value": "State: [x=..., y=..., vx=..., vy=..., angle=..., angular_vel=..., left_leg=..., right_leg=...]. What action should the lander take?"
    },
    {
      "from": "gpt",
      "value": "Action: 2 (fire main engine)."
    }
  ]
}
```

## Task

The goal of this model is to predict the next LunarLander action from the current environment state.

Current task type:
- input: LunarLander state prompt
- output: action text

This model is not a reward model. It is an action prediction model.

## Repository Split

We are keeping two separate repositories:

1. LoRA adapter repo
- https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-lora

2. GGUF export repo
- https://huggingface.co/alirizaercan/qwen2.5-0.5b-lunarlander-action-gguf

Why separate them?
- LoRA adapters and GGUF files are different artifact types.
- This makes downloading and reuse clearer.
- Ollama / llama.cpp users can go directly to the GGUF repo.

## Training Setup

- Max sequence length: `1024`
- Load in 4bit: `True`
- LoRA rank: `16`
- Gradient checkpointing: `unsloth`
- Optimizer: `adamw_8bit`

## Example Inference

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "alirizaercan/qwen2.5-0.5b-lunarlander-action-lora"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "State: [x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]. What action should the lander take?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Limitations

- This model is trained on the dataset provided above only.
- It predicts action text, not continuous control values.
- It should be evaluated carefully before any downstream control usage.

## License

Fill this section according to:
1. the base model license
2. your dataset license
3. your intended release policy

## Authors

- Model author: `alirizaercan`
- Dataset contributor: `Ali2023kosemen`

## Notes

If needed later, you can still add a third repository for a merged full model.
