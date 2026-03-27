# Qwen 2.5 0.5B LunarLander Inference

Bu repo, LunarLander durumu verildiğinde bir sonraki aksiyonu tahmin eden fine-tuned modeli içerir.

Model çıktısı tek satır olmalıdır:

```text
Action: <0|1|2|3>
```

## Aksiyon Anlamları

- `Action: 0` -> hiçbir şey yapma
- `Action: 1` -> sol yönlendirme motoru
- `Action: 2` -> ana motor
- `Action: 3` -> sağ yönlendirme motoru

## Kurulum

Python 3.10+ önerilir.

```bash
pip install -U transformers torch
```

## Modeli Yükleme

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "alirizaercan/qwen25_05b_base_full_ft_lunarlander_a4000_inference"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
```

## Örnek Kullanım

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "alirizaercan/qwen25_05b_base_full_ft_lunarlander_a4000_inference"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Return exactly one line in this format:
Action: <0|1|2|3>

State: [x=0.0005, y=1.4126, vx=0.0492, vy=0.0739, angle=-0.0006, angular_vel=-0.0112, left_leg=0.0000, right_leg=0.0000]. What action should the lander take?

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_new_tokens=8,
    do_sample=False,
)

text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(text)
```

## Beklenen Çıktı

Örnek çıktı:

```text
Action: 2
```

## Notlar

- Prompt formatını değiştirmeden kullanmanız önerilir.
- Model, en iyi sonucu `Action: <id>` formatı beklendiğinde verir.
- `do_sample=False` kullanmak daha kararlı çıktı verir.
