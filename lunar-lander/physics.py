import math

import numpy as np

from box2d_imports import Box2D, circleShape, edgeShape, fixtureDef, polygonShape, revoluteJointDef
from constants import (
    FPS,
    INITIAL_RANDOM,
    LANDER_POLY,
    LEG_AWAY,
    LEG_DOWN,
    LEG_H,
    LEG_SPRING_TORQUE,
    LEG_W,
    MAIN_ENGINE_POWER,
    MAIN_ENGINE_Y_LOCATION,
    SCALE,
    SIDE_ENGINE_AWAY,
    SIDE_ENGINE_HEIGHT,
    SIDE_ENGINE_POWER,
    VIEWPORT_H,
    VIEWPORT_W,
)


def destroy_world(env):
    if not env.moon:
        return
    env.world.contactListener = None
    clean_particles(env, True)
    env.world.DestroyBody(env.moon)
    env.moon = None
    env.world.DestroyBody(env.lander)
    env.lander = None
    env.world.DestroyBody(env.legs[0])
    env.world.DestroyBody(env.legs[1])


def reset_world(env):
    env.world = Box2D.b2World(gravity=(0, env.gravity))


def create_terrain(env):
    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE

    chunks = 11
    height = env.np_random.uniform(0, H / 2, size=(chunks + 1,))
    chunk_x = [W / (chunks - 1) * i for i in range(chunks)]
    env.helipad_x1 = chunk_x[chunks // 2 - 1]
    env.helipad_x2 = chunk_x[chunks // 2 + 1]
    env.helipad_y = H / 4
    height[chunks // 2 - 2] = env.helipad_y
    height[chunks // 2 - 1] = env.helipad_y
    height[chunks // 2 + 0] = env.helipad_y
    height[chunks // 2 + 1] = env.helipad_y
    height[chunks // 2 + 2] = env.helipad_y
    smooth_y = [
        0.33 * (height[i - 1] + height[i] + height[i + 1])
        for i in range(chunks)
    ]

    env.moon = env.world.CreateStaticBody(shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
    env.sky_polys = []
    for i in range(chunks - 1):
        p1 = (chunk_x[i], smooth_y[i])
        p2 = (chunk_x[i + 1], smooth_y[i + 1])
        env.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
        env.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

    env.moon.color1 = (0.0, 0.0, 0.0)
    env.moon.color2 = (0.0, 0.0, 0.0)

    return VIEWPORT_W / SCALE / 2, VIEWPORT_H / SCALE


def create_lander(env, initial_x, initial_y):
    env.lander = env.world.CreateDynamicBody(
        position=(initial_x, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,
            restitution=0.0,
        ),
    )
    env.lander.color1 = (128, 102, 230)
    env.lander.color2 = (77, 77, 128)
    env.lander.ApplyForceToCenter(
        (
            env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        ),
        True,
    )


def create_legs(env, initial_x, initial_y):
    env.legs = []
    for i in [-1, +1]:
        leg = env.world.CreateDynamicBody(
            position=(initial_x - i * LEG_AWAY / SCALE, initial_y),
            angle=(i * 0.05),
            fixtures=fixtureDef(
                shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                density=1.0,
                restitution=0.0,
                categoryBits=0x0020,
                maskBits=0x001,
            ),
        )
        leg.ground_contact = False
        leg.color1 = (128, 102, 230)
        leg.color2 = (77, 77, 128)
        joint = revoluteJointDef(
            bodyA=env.lander,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=LEG_SPRING_TORQUE,
            motorSpeed=+0.3 * i,
        )
        if i == -1:
            joint.lowerAngle = +0.9 - 0.5
            joint.upperAngle = +0.9
        else:
            joint.lowerAngle = -0.9
            joint.upperAngle = -0.9 + 0.5
        leg.joint = env.world.CreateJoint(joint)
        env.legs.append(leg)
    env.drawlist = [env.lander] + env.legs


def create_particle(env, mass, x, y, ttl):
    particle = env.world.CreateDynamicBody(
        position=(x, y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
            density=mass,
            friction=0.1,
            categoryBits=0x0100,
            maskBits=0x001,
            restitution=0.3,
        ),
    )
    particle.ttl = ttl
    env.particles.append(particle)
    clean_particles(env, False)
    return particle


def clean_particles(env, all_particle):
    while env.particles and (all_particle or env.particles[0].ttl < 0):
        env.world.DestroyBody(env.particles.pop(0))


def apply_wind(env):
    if not env.enable_wind or env.legs[0].ground_contact or env.legs[1].ground_contact:
        return
    wind_mag = (
        math.tanh(
            math.sin(0.02 * env.wind_idx) + math.sin(math.pi * 0.01 * env.wind_idx)
        )
        * env.wind_power
    )
    env.wind_idx += 1
    env.lander.ApplyForceToCenter((wind_mag, 0.0), True)

    torque_mag = (
        math.tanh(
            math.sin(0.02 * env.torque_idx) + math.sin(math.pi * 0.01 * env.torque_idx)
        )
        * env.turbulence_power
    )
    env.torque_idx += 1
    env.lander.ApplyTorque(torque_mag, True)


def apply_engines(env, action):
    tip = (math.sin(env.lander.angle), math.cos(env.lander.angle))
    side = (-tip[1], tip[0])
    dispersion = [env.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if (env.continuous and action[0] > 0.0) or (not env.continuous and action == 2):
        if env.continuous:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5
        else:
            m_power = 1.0

        ox = tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
        oy = -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
        impulse_pos = (env.lander.position[0] + ox, env.lander.position[1] + oy)
        if env.render_mode is not None:
            particle = create_particle(env, 3.5, impulse_pos[0], impulse_pos[1], m_power)
            particle.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos,
                True,
            )
        env.lander.ApplyLinearImpulse(
            (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
            impulse_pos,
            True,
        )

    s_power = 0.0
    if (env.continuous and np.abs(action[1]) > 0.5) or (not env.continuous and action in [1, 3]):
        if env.continuous:
            direction = np.sign(action[1])
            s_power = np.clip(np.abs(action[1]), 0.5, 1.0)
        else:
            direction = action - 2
            s_power = 1.0

        ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE)
        impulse_pos = (
            env.lander.position[0] + ox - tip[0] * 17 / SCALE,
            env.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )
        if env.render_mode is not None:
            particle = create_particle(env, 0.7, impulse_pos[0], impulse_pos[1], s_power)
            particle.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos,
                True,
            )
        env.lander.ApplyLinearImpulse(
            (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
            impulse_pos,
            True,
        )

    return m_power, s_power


def compute_step_result(env, m_power, s_power):
    env.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

    pos = env.lander.position
    vel = env.lander.linearVelocity
    state = [
        (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
        (pos.y - (env.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
        vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
        vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
        env.lander.angle,
        20.0 * env.lander.angularVelocity / FPS,
        1.0 if env.legs[0].ground_contact else 0.0,
        1.0 if env.legs[1].ground_contact else 0.0,
    ]

    shaping = (
        -100 * np.sqrt(state[0] * state[0] + state[1] * state[1])
        - 100 * np.sqrt(state[2] * state[2] + state[3] * state[3])
        - 100 * abs(state[4])
        + 10 * state[6]
        + 10 * state[7]
    )
    reward = 0 if env.prev_shaping is None else shaping - env.prev_shaping
    env.prev_shaping = shaping
    reward -= m_power * 0.30
    reward -= s_power * 0.03

    terminated = False
    if env.game_over or abs(state[0]) >= 1.0:
        terminated = True
        reward = -100
    if not env.lander.awake:
        terminated = True
        reward = +100

    return np.array(state, dtype=np.float32), reward, terminated
