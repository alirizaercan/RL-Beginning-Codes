import numpy as np

import gymnasium as gym
from gymnasium import error, spaces
from gymnasium.utils import EzPickle

from box2d_imports import Box2D
from constants import FPS, STATE_HIGH, STATE_LOW
from contact_detector import ContactDetector
from physics import (
    apply_engines,
    apply_wind,
    compute_step_result,
    create_lander,
    create_legs,
    create_terrain,
    destroy_world,
    reset_world,
)
from render_utils import render_env


class LunarLander(gym.Env, EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: str | None = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
    ):
        EzPickle.__init__(
            self, render_mode, continuous, gravity, enable_wind, wind_power, turbulence_power
        )

        assert -12.0 < gravity < 0.0, (
            f"gravity (current value: {gravity}) must be between -12 and 0"
        )
        self.gravity = gravity
        self.wind_power = wind_power
        self.turbulence_power = turbulence_power
        self.enable_wind = enable_wind

        if not 0.0 <= wind_power <= 20.0:
            gym.logger.warn(
                f"wind_power value is recommended to be between 0.0 and 20.0, (current value: {wind_power})"
            )
        if not 0.0 <= turbulence_power <= 2.0:
            gym.logger.warn(
                f"turbulence_power value is recommended to be between 0.0 and 2.0, (current value: {turbulence_power})"
            )

        self.render_mode = render_mode
        self.continuous = continuous
        self.screen = None
        self.clock = None
        self.isopen = True
        self.world = Box2D.b2World(gravity=(0, gravity))
        self.moon = None
        self.lander = None
        self.particles = []
        self.prev_reward = None

        # LunarLander state is an 8D observation vector:
        # [0] x_position
        # [1] y_position
        # [2] x_velocity
        # [3] y_velocity
        # [4] angle
        # [5] angular_velocity
        # [6] left_leg_contact
        # [7] right_leg_contact
        self.observation_space = spaces.Box(
            np.array(STATE_LOW, dtype=np.float32),
            np.array(STATE_HIGH, dtype=np.float32),
        )

        # continuous=False  -> Discrete(4) action space
        #   0 = do nothing
        #   1 = fire left engine
        #   2 = fire main engine
        #   3 = fire right engine
        #
        # continuous=True -> Box(-1, 1, (2,)) action space
        #   action[0] = main engine throttle
        #   action[1] = side engine throttle / direction
        self.action_space = (
            spaces.Box(-1, +1, (2,), dtype=np.float32)
            if continuous
            else spaces.Discrete(4)
        )

    def _destroy(self):
        destroy_world(self)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._destroy()

        reset_world(self)
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref
        self.game_over = False
        self.prev_shaping = None

        initial_x, initial_y = create_terrain(self)
        create_lander(self, initial_x, initial_y)

        if self.enable_wind:
            self.wind_idx = self.np_random.integers(-9999, 9999)
            self.torque_idx = self.np_random.integers(-9999, 9999)

        create_legs(self, initial_x, initial_y)

        if self.render_mode == "human":
            self.render()
        return self.step(np.array([0, 0]) if self.continuous else 0)[0], {}

    def step(self, action):
        assert self.lander is not None, "You forgot to call reset()"

        apply_wind(self)
        if self.continuous:
            # Continuous mode:
            # action is a 2D float vector and each value is clipped into [-1, 1].
            action = np.clip(action, -1, +1).astype(np.float64)
        else:
            # Discrete mode:
            # action must be one of {0, 1, 2, 3}.
            assert self.action_space.contains(action), f"{action!r} ({type(action)}) invalid "

        m_power, s_power = apply_engines(self, action)
        state, reward, terminated = compute_step_result(self, m_power, s_power)

        if self.render_mode == "human":
            self.render()
        return state, reward, terminated, False, {}

    def render(self):
        return render_env(self)

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


class LunarLanderContinuous:
    def __init__(self):
        raise error.Error(
            "Error initializing LunarLanderContinuous Environment.\n"
            "Currently, we do not support initializing this mode of environment by calling the class directly.\n"
            "To use this environment, instead create it by specifying the continuous keyword in gym.make, i.e.\n"
            'gym.make("LunarLander-v3", continuous=True)'
        )
