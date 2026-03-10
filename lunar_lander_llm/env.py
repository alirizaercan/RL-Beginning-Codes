import math
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import EzPickle

import Box2D

from physics import (
    FPS, SCALE, VIEWPORT_W, VIEWPORT_H, LEG_DOWN,
    ContactDetector,
    build_world, destroy_world, apply_wind, apply_engines,
    clean_particles,
)
from rendering import render_frame, close_display
from llm_reward import build_reward_prompt, get_llm_reward


class LunarLanderEnv(gym.Env, EzPickle):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous:  bool  = False,
        gravity:     float = -10.0,
        enable_wind: bool  = False,
        wind_power:       float = 15.0,
        turbulence_power: float = 1.5,
        extra_rules: str = "",
    ):
        EzPickle.__init__(
            self, render_mode, continuous, gravity,
            enable_wind, wind_power, turbulence_power, extra_rules,
        )

        assert -12.0 < gravity < 0.0, (
            f"gravity must be in (-12, 0), got {gravity}"
        )
        if not (0.0 <= wind_power <= 20.0):
            gym.logger.warn(f"wind_power={wind_power} outside recommended 0..20")
        if not (0.0 <= turbulence_power <= 2.0):
            gym.logger.warn(f"turbulence_power={turbulence_power} outside recommended 0..2")

        self.gravity          = gravity
        self.enable_wind      = enable_wind
        self.wind_power       = wind_power
        self.turbulence_power = turbulence_power
        self.continuous       = continuous
        self.extra_rules      = extra_rules
        self.render_mode      = render_mode

        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon   = None
        self.lander = None
        self.legs   = []
        self.particles = []

        self.screen = None
        self.clock  = None
        self.isopen = True

        self.prev_shaping = None
        self.game_over    = False

        # these are bounds for position
        # x coordinate
        # y coordinate
        # velocity bounds is 5x rated speed
        low = np.array(
            [-2.5, -2.5, -10.0, -10.0, -2 * math.pi, -10.0, 0.0, 0.0],
            dtype=np.float32,
        )
        high = np.array(
            [+2.5, +2.5, +10.0, +10.0, +2 * math.pi, +10.0, 1.0, 1.0],
            dtype=np.float32,
        )
        
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.continuous:
           # Action is two floats [main engine, left-right engines].
            # Main engine: -1..0 off, 0..+1 throttle from 50% to 100% power. Engine can't work with less than 50% power.
            # Left-right:  -1.0..-0.5 fire left engine, +0.5..+1.0 fire right engine, -0.5..0.5 off
            self.action_space = spaces.Box(-1, +1, shape=(2,), dtype=np.float32)
        else:
            # Nop, fire left engine, main engine, right engine
            self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity

        state = np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)

        return state

    def _get_info(self) -> dict:
        if self.lander is None:
            return {}
        obs = self._get_obs()
        return {
            "x":              float(obs[0]),
            "y":              float(obs[1]),
            "vx":             float(obs[2]),
            "vy":             float(obs[3]),
            "angle_deg":      float(np.degrees(obs[4])),
            "angular_vel":    float(obs[5]),
            "legs_on_ground": int(obs[6]) + int(obs[7]),
            "game_over":      self.game_over,
        }

    def reset(
        self,
        seed:    Optional[int]  = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)      
        destroy_world(self) 

        self.world = Box2D.b2World(gravity=(0, self.gravity))
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over    = False
        self.prev_shaping = None

        build_world(self)                  
        if self.render_mode == "human":
            render_frame(self)

        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        assert self.lander is not None, "call reset() before step()"

        if self.continuous:
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            assert self.action_space.contains(action), f"Invalid action: {action}"

        if self.enable_wind:
            apply_wind(self)

        m_power, s_power = apply_engines(self, action)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        obs = self._get_obs()

        shaping = (
            -100 * np.sqrt(obs[0]**2 + obs[1]**2)   
            - 100 * np.sqrt(obs[2]**2 + obs[3]**2)  
            - 100 * abs(obs[4])                      
            + 10  * obs[6]                           
            + 10  * obs[7]                           
        ) # And ten points for legs contact, the idea is if you
        original_reward = 0.0
        # lose contact again after landing, you get negative reward
        if self.prev_shaping is not None:
            original_reward = shaping - self.prev_shaping   # delta, not absolute
        self.prev_shaping = shaping

        original_reward -= m_power * 0.30   # less fuel spent is better, about -30 for heuristic landing
        original_reward -= s_power * 0.03   

        terminated = False
        if self.game_over or abs(obs[0]) >= 1.0:
            terminated    = True
            original_reward = -100.0
        if not self.lander.awake:
            terminated    = True
            original_reward = +100.0

        prompt     = build_reward_prompt(obs, action, terminated, self.extra_rules)
        llm_reward = get_llm_reward(prompt)

        if self.render_mode == "human":
            render_frame(self)

        info = {**self._get_info(), "original_reward": original_reward}
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return obs, llm_reward, terminated, False, info

    def render(self):
        return render_frame(self)

    def close(self):
        close_display(self)
        super().close()