import re
from openai import OpenAI


_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
MODEL = "qwen2.5:3b"

ACTION_NAMES = {
    0: "do nothing",
    1: "fire left orientation engine",
    2: "fire main engine",
    3: "fire right orientation engine",
}

OFFICIAL_REWARD_RULES = """
Each step:
  - Reward INCREASES the closer/slower the lander is to the landing pad
  - Reward DECREASES the further/faster it is from the pad
  - Reward DECREASES the more the lander is tilted from horizontal
  - +10 for each leg in contact with the ground
  - -0.03 for per frame a side engine fires
  - -0.30 for per frame the main engine fires
  - GIVE -100 for crashing
  - GIVE +100 for landing safely
Goal: cumulative episode reward >= 200
"""

OBS_SPACE_DESCRIPTION = """
  x position        range -2.5..+2.5   
  y position        range -2.5..+2.5   
  x velocity        range -10..+10     
  y velocity        range -10..+10     
  angle             range -2π..+2π     
  angular velocity  range -10..+10     
  left  leg contact 0 or 1            
  right leg contact 0 or 1             
"""

ACTION_SPACE_DESCRIPTION = """
  0 = do nothing
  1 = fire left orientation engine
  2 = fire main engine 
  3 = fire right orientation engine
"""

def build_reward_prompt(obs, action, terminated, extra_rules: str = ""):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs
    print(f"x:{x}, y:{y}, vx:{vx}, vy:{vy}, angle:{angle}, ang_vel:{ang_vel}, leg1:{leg1}, leg2:{leg2}, action:{action}, terminated:{terminated}")
    action_str = (
        f"[{action[0]:.3f}, {action[1]:.3f}] (continuous)"
        if hasattr(action, "__len__")
        else f"[{action}] {ACTION_NAMES[action]}"
    )
    extra = f"\nExtra rules:\n{extra_rules.strip()}" if extra_rules.strip() else ""

    return f"""You are a reward function for a Lunar Lander RL environment.

=== Observation space (all values normalised) ===
{OBS_SPACE_DESCRIPTION.strip()}

=== Action space (discrete) ===
{ACTION_SPACE_DESCRIPTION.strip()}    

=== State after this step ===
x position      : {x:.4f}
y position      : {y:.4f}   
x velocity      : {vx:.4f}
y velocity      : {vy:.4f}
angle           : {angle:.4f} rad  ({float(angle)*57.296:.2f} deg)
angular velocity: {ang_vel:.4f}
left leg contact: {bool(leg1)}
right leg contact:{bool(leg2)}
action taken    : {action_str}
episode ended   : {terminated}

=== Official reward rules ===
{OFFICIAL_REWARD_RULES.strip()}
{extra}

=== Your task ===
Apply the rules to the state above and output a SINGLE float number.
No explanation, no units, no labels. Just the number.
Valid examples:  -0.3   10.0   -100   0.0
"""

def get_llm_reward(prompt):
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    print(f"[LLM reward response] = {raw}")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", raw)
    if not match:
        raise ValueError(f"LLM returned non-numeric output: {raw!r}")
    return float(match.group())