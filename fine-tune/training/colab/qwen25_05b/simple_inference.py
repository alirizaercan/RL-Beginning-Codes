from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

STRICT_PREFIX = (
    "Return exactly one line in this format:\n"
    "Action: <0|1|2|3>\n\n"
)

def build_prompt(state: str) -> str:
    instruction = STRICT_PREFIX + f"State: {state}. What action should the lander take?"
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )

def load_model(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model

def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

m_tokenizer, m_model = load_model(Path("x"))
prompt_ex = build_prompt("State: [x=-0.2301, y=0.6570, vx=-0.1006, vy=-0.4914, angle=-0.3743, angular_vel=0.0626, left_leg=0.0000, right_leg=0.0000]. What action should the lander take?")
text_den = generate_answer(m_tokenizer, m_model, prompt_ex, max_new_tokens=512)
print(text_den)