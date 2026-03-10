import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility


def heuristic(env, s):
    angle_targ = s[0] * 0.5 + s[2] * 1.0
    angle_targ = min(0.4, max(-0.4, angle_targ))
    hover_targ = 0.55 * np.abs(s[0])

    angle_todo = (angle_targ - s[4]) * 0.5 - s[5] * 1.0
    hover_todo = (hover_targ - s[1]) * 0.5 - s[3] * 0.5

    if s[6] or s[7]:
        angle_todo = 0
        hover_todo = -s[3] * 0.5

    if env.unwrapped.continuous:
        action = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        return np.clip(action, -1, +1)

    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
        return 2
    if angle_todo < -0.05:
        return 3
    if angle_todo > +0.05:
        return 1
    return 0


def demo_heuristic_lander(env, seed=None, render=False):
    total_reward = 0
    steps = 0
    state, info = env.reset(seed=seed)
    while True:
        action = heuristic(env, state)
        state, reward, terminated, truncated, info = step_api_compatibility(env.step(action), True)
        total_reward += reward

        if render:
            still_open = env.render()
            if still_open is False:
                break

        if steps % 20 == 0 or terminated or truncated:
            print("observations:", " ".join([f"{x:+0.2f}" for x in state]))
            print(f"step {steps} total_reward {total_reward:+0.2f}")
        steps += 1
        if terminated or truncated:
            break

    if render:
        env.close()
    return total_reward
