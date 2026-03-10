import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled

from box2d_imports import circleShape
from constants import FPS, SCALE, VIEWPORT_H, VIEWPORT_W
from physics import clean_particles


def render_env(env):
    if env.render_mode is None:
        assert env.spec is not None
        gym.logger.warn(
            "You are calling render method without specifying any render mode. "
            f'You can specify the render_mode at initialization, e.g. gym.make("{env.spec.id}", render_mode="rgb_array")'
        )
        return

    try:
        import pygame
        from pygame import gfxdraw
    except ImportError as e:
        raise DependencyNotInstalled(
            'pygame is not installed, run `pip install "gymnasium[box2d]"`'
        ) from e

    if env.screen is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    if env.clock is None:
        env.clock = pygame.time.Clock()

    env.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
    pygame.transform.scale(env.surf, (SCALE, SCALE))
    pygame.draw.rect(env.surf, (255, 255, 255), env.surf.get_rect())

    for obj in env.particles:
        obj.ttl -= 0.15
        obj.color1 = (
            int(max(0.2, 0.15 + obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
            int(max(0.2, 0.5 * obj.ttl) * 255),
        )
        obj.color2 = obj.color1

    clean_particles(env, False)

    for poly in env.sky_polys:
        scaled = [(coord[0] * SCALE, coord[1] * SCALE) for coord in poly]
        pygame.draw.polygon(env.surf, (0, 0, 0), scaled)
        gfxdraw.aapolygon(env.surf, scaled, (0, 0, 0))

    for obj in env.particles + env.drawlist:
        for fixture in obj.fixtures:
            trans = fixture.body.transform
            if type(fixture.shape) is circleShape:
                center = trans * fixture.shape.pos * SCALE
                pygame.draw.circle(env.surf, obj.color1, center, fixture.shape.radius * SCALE)
                pygame.draw.circle(env.surf, obj.color2, center, fixture.shape.radius * SCALE)
            else:
                path = [trans * vertex * SCALE for vertex in fixture.shape.vertices]
                pygame.draw.polygon(env.surf, color=obj.color1, points=path)
                gfxdraw.aapolygon(env.surf, path, obj.color1)
                pygame.draw.aalines(env.surf, color=obj.color2, points=path, closed=True)

            for x in [env.helipad_x1, env.helipad_x2]:
                x = x * SCALE
                flagy1 = env.helipad_y * SCALE
                flagy2 = flagy1 + 50
                pygame.draw.line(env.surf, (255, 255, 255), (x, flagy1), (x, flagy2), 1)
                pygame.draw.polygon(
                    env.surf,
                    (204, 204, 0),
                    [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                )
                gfxdraw.aapolygon(
                    env.surf,
                    [(x, flagy2), (x, flagy2 - 10), (x + 25, flagy2 - 5)],
                    (204, 204, 0),
                )

    env.surf = pygame.transform.flip(env.surf, False, True)
    if env.render_mode == "human":
        env.screen.blit(env.surf, (0, 0))
        pygame.event.pump()
        env.clock.tick(FPS)
        pygame.display.flip()
    elif env.render_mode == "rgb_array":
        return np.transpose(np.array(pygame.surfarray.pixels3d(env.surf)), axes=(1, 0, 2))
