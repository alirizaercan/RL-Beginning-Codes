import numpy as np
from physics import SCALE, VIEWPORT_W, VIEWPORT_H, FPS

try:
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise ImportError(
        'pygame is not installed, run `pip install "gymnasium[box2d]"`'
    ) from e

def render_frame(env):
    if env.screen is None and env.render_mode == "human":
        pygame.init()
        pygame.display.init()
        env.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
    if env.clock is None:
        env.clock = pygame.time.Clock()

    surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
    pygame.draw.rect(surf, (255, 255, 255), surf.get_rect())

    for obj in env.particles:
        obj.ttl -= 0.15
        obj.color1 = (
            int(max(0.2, 0.15 + obj.ttl) * 255),
            int(max(0.2, 0.5  * obj.ttl) * 255),
            int(max(0.2, 0.5  * obj.ttl) * 255),
        )
        obj.color2 = obj.color1

    for p in env.sky_polys:
        scaled = [(c[0] * SCALE, c[1] * SCALE) for c in p]
        pygame.draw.polygon(surf, (0, 0, 0), scaled)
        gfxdraw.aapolygon(surf, scaled, (0, 0, 0))

    for obj in env.particles + env.drawlist:
        for f in obj.fixtures:
            trans = f.body.transform
            if type(f.shape) is __import__("Box2D").b2CircleShape:
                pygame.draw.circle(
                    surf,
                    color=obj.color1,
                    center=trans * f.shape.pos * SCALE,
                    radius=f.shape.radius * SCALE,
                )
            else:
                path = [trans * v * SCALE for v in f.shape.vertices]
                pygame.draw.polygon(surf, color=obj.color1, points=path)
                gfxdraw.aapolygon(surf, path, obj.color1)
                pygame.draw.aalines(surf, color=obj.color2, points=path, closed=True)

            for hx in [env.helipad_x1, env.helipad_x2]:
                hx  *= SCALE
                fy1  = env.helipad_y * SCALE
                fy2  = fy1 + 50
                pygame.draw.line(surf, (255, 255, 255), (hx, fy1), (hx, fy2), 1)
                pygame.draw.polygon(
                    surf, (204, 204, 0),
                    [(hx, fy2), (hx, fy2 - 10), (hx + 25, fy2 - 5)],
                )

    surf = pygame.transform.flip(surf, False, True)

    if env.render_mode == "human":
        env.screen.blit(surf, (0, 0))
        pygame.event.pump()
        env.clock.tick(FPS)
        pygame.display.flip()
        return None
    else:  # rgb_array
        return np.transpose(np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2))

def close_display(env):
    if env.screen is not None:
        pygame.display.quit()
        pygame.quit()
        env.screen = None