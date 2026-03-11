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

OFFICIAL_TERMINATION_RULES = """
The episode MUST end (terminated=true) if ANY of the following are true:
  - The lander has crashed (hull touched the ground, game_over=true)
  - The lander drifted too far horizontally (|x position| >= 1.0)
  - The lander has come to a complete stop on the ground (both legs in contact AND nearly zero velocity)
The episode continues (terminated=false) in all other cases.
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

def build_reward_prompt(obs, action, game_over, extra_rules: str = ""):
    x, y, vx, vy, angle, ang_vel, leg1, leg2 = obs
    #print(f"x:{x}, y:{y}, vx:{vx}, vy:{vy}, angle:{angle}, ang_vel:{ang_vel}, leg1:{leg1}, leg2:{leg2}, action:{action}, game_over:{game_over}")
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
game_over (hull hit ground): {game_over}

=== Official reward rules ===
{OFFICIAL_REWARD_RULES.strip()}
{extra}

=== Official termination rules ===
{OFFICIAL_TERMINATION_RULES.strip()}
{extra}

=== Your task ===
Apply the rules above and output EXACTLY two values on two separate lines:
Line 1: a single float for the reward (e.g. -0.3  or  10.0  or  -100)
Line 2: true or false for whether the episode should terminate

No explanation, no labels, no extra text. Example output:
-4.5
false
"""

def get_llm_decision(prompt):
    response = _client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16,
        temperature=0.0,
    )
    raw = response.choices[0].message.content.strip()
    print(f"[LLM reward response] = {raw}")

    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if not lines:
        raise ValueError(f"LLM returned empty output: {raw!r}")
    reward_match = re.search(r"[-+]?\d+(?:\.\d+)?", lines[0])
    if not reward_match:
        raise ValueError(f"LLM returned non-numeric reward: {raw!r}")
    reward = float(reward_match.group())

    if len(lines) < 2:
        raise ValueError(f"LLM did not return a terminated line: {raw!r}")
    second = lines[1].lower()
    if "true" in second:
        terminated = True
    elif "false" in second:
        terminated = False
    else:
        raise ValueError(f"LLM returned non-boolean terminated value: {raw!r}")

    return reward, terminated