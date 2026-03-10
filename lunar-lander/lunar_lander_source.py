import gymnasium as gym
from heuristic import demo_heuristic_lander
from lunar_lander_env import LunarLander, LunarLanderContinuous


if __name__ == "__main__":
    env = gym.make("LunarLander-v3", render_mode="rgb_array")
    demo_heuristic_lander(env, render=True)
