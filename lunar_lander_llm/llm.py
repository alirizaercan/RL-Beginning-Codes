import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "./models/hf_model_ep_3500"

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _model is not None:
        return
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    _model.eval()


def obs_to_prompt(obs):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs
    instruction = (
        "Return exactly one line in this format:\n"
        "Action: <0|1|2|3>\n\n"
        f"State: [x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
        f"angle={angle:.4f}, angular_vel={ang_vel:.4f}, "
        f"left_leg={leg1:.4f}, right_leg={leg2:.4f}]. "
        f"What action should the lander take?"
    )
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def get_llm_action(obs):
    _load_model()
    prompt = obs_to_prompt(obs)

    inputs = _tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(_model.device)
    if inputs["input_ids"].shape[1] == 0:
        raise ValueError("Tokenizer produced empty input — check prompt or tokenizer config.")
    with torch.inference_mode():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=_tokenizer.pad_token_id,
            eos_token_id=_tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    raw = _tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    print(f"[LLM] {raw}")

    match = re.search(r"action\s*:\s*([0-3])", raw.lower())
    if not match:
        match = re.search(r"\b([0-3])\b", raw)
    if not match:
        raise ValueError(f"LLM geçerli bir aksiyon döndürmedi: {raw!r}")

    action = int(match.group(1))
    return action
