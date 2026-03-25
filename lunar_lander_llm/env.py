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
from llm_reward import get_llm_action        


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

        assert -12.0 < gravity < 0.0
        self.gravity          = gravity
        self.enable_wind      = enable_wind
        self.wind_power       = wind_power
        self.turbulence_power = turbulence_power
        self.continuous       = continuous
        self.extra_rules      = extra_rules
        self.render_mode      = render_mode

        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon      = None
        self.lander    = None
        self.legs      = []
        self.particles = []

        self.screen = None
        self.clock  = None
        self.isopen = True

        self.prev_shaping = None
        self.game_over    = False

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
            self.action_space = spaces.Box(-1, +1, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        pos = self.lander.position
        vel = self.lander.linearVelocity
        return np.array([
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
        ], dtype=np.float32)

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

    def reset(self, seed=None, options=None):
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

    def step(self, action=None):          
        assert self.lander is not None, "call reset() before step()"

        obs_now = self._get_obs()

        if action is None:
            action = get_llm_action(obs_now) 

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
            -100 * np.sqrt(obs[0] ** 2 + obs[1] ** 2)
            - 100 * np.sqrt(obs[2] ** 2 + obs[3] ** 2)
            - 100 * abs(obs[4])
            + 10  * obs[6]
            + 10  * obs[7]
        )
        reward = 0.0
        if self.prev_shaping is not None:
            reward = shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        terminated = False
        if self.game_over or abs(obs[0]) >= 1.0:
            terminated    = True
            reward = -100.0
        if not self.lander.awake:
            terminated    = True
            reward = +100.0

        if self.render_mode == "human":
            render_frame(self)

        info = {**self._get_info(), "action": action}
        return obs, reward, terminated, False, info

    def render(self):
        return render_frame(self)

    def close(self):
        close_display(self)
        super().close()