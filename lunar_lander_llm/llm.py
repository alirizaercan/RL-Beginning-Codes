import re
from openai import OpenAI


_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "alirizaercan1/lunarlander-action-v3:latest"

def obs_to_prompt(obs):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs
    return (
        f"State: [x={x:.4f}, y={y:.4f}, vx={vx:.4f}, vy={vy:.4f}, "
        f"angle={angle:.4f}, angular_vel={ang_vel:.4f}, "
        f"left_leg={leg1:.4f}, right_leg={leg2:.4f}]. "
        f"What action should the lander take?"
    )

def get_llm_action(obs):
    prompt = obs_to_prompt(obs)
 
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
 
    match = re.search(r"\b([0-3])\b", raw)
    if not match:
        raise ValueError(f"LLM geçerli bir aksiyon döndürmedi: {raw!r}")
 
    action = int(match.group(1))
    print(f"[LLM action] {action}")
    return action