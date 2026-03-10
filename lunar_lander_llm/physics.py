import math
import numpy as np
import Box2D
from Box2D.b2 import (
    circleShape,
    contactListener,
    edgeShape,
    fixtureDef,
    polygonShape,
    revoluteJointDef,
)


FPS   = 50
SCALE = 30.0  # affects how fast-paced the game is, forces should be adjusted as well

MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6

INITIAL_RANDOM = 1000.0  # Set 1500 to make game harder

LANDER_POLY = [(-14, +17), (-17, 0), (-17, -10), (+17, -10), (+17, 0), (+14, +17)]
LEG_AWAY         = 20   
LEG_DOWN         = 18   
LEG_W, LEG_H     = 2, 8
LEG_SPRING_TORQUE = 40

SIDE_ENGINE_HEIGHT    = 14
SIDE_ENGINE_AWAY      = 12
MAIN_ENGINE_Y_LOCATION = 4   # The Y location of the main engine on the body of the Lander.

VIEWPORT_W = 600
VIEWPORT_H = 400


class ContactDetector(contactListener):
    def __init__(self, env):
        contactListener.__init__(self)
        self.env = env

    def BeginContact(self, contact):
        # Hull contact = crash
        if (
            self.env.lander == contact.fixtureA.body
            or self.env.lander == contact.fixtureB.body
        ):
            self.env.game_over = True
        # Leg contact = safe touchdown sensor
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False

def build_world(env):
    W = VIEWPORT_W / SCALE
    H = VIEWPORT_H / SCALE

    # Create Terrain
    CHUNKS = 11
    height    = env.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
    chunk_x   = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]
    env.helipad_x1 = chunk_x[CHUNKS // 2 - 1]
    env.helipad_x2 = chunk_x[CHUNKS // 2 + 1]
    env.helipad_y  = H / 4

    for k in range(-2, 3):
        height[CHUNKS // 2 + k] = env.helipad_y

    smooth_y = [
        0.33 * (height[i - 1] + height[i] + height[i + 1])
        for i in range(CHUNKS)
    ]

    env.moon = env.world.CreateStaticBody(
        shapes=edgeShape(vertices=[(0, 0), (W, 0)])
    )
    env.sky_polys = []
    for i in range(CHUNKS - 1):
        p1 = (chunk_x[i],     smooth_y[i])
        p2 = (chunk_x[i + 1], smooth_y[i + 1])
        env.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
        env.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])

    env.moon.color1 = (0.0, 0.0, 0.0)
    env.moon.color2 = (0.0, 0.0, 0.0)

    # Create Lander body
    initial_x = VIEWPORT_W / SCALE / 2
    initial_y = VIEWPORT_H / SCALE
    env.lander = env.world.CreateDynamicBody(
        position=(initial_x, initial_y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=polygonShape(
                vertices=[(x / SCALE, y / SCALE) for x, y in LANDER_POLY]
            ),
            density=5.0,
            friction=0.1,
            categoryBits=0x0010,
            maskBits=0x001,      # collide only with ground
            restitution=0.0,
        ),
    )
    env.lander.color1 = (128, 102, 230)
    env.lander.color2 = (77,  77,  128)

    # Apply the initial random impulse to the lander
    env.lander.ApplyForceToCenter(
        (
            env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
            env.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
        ),
        True,
    )

    if env.enable_wind:  # Initialize wind pattern based on index
        env.wind_idx   = env.np_random.integers(-9999, 9999)
        env.torque_idx = env.np_random.integers(-9999, 9999)

    # Create Lander Legs
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
        leg.color2 = (77,  77,  128)

        rjd = revoluteJointDef(
            bodyA=env.lander,
            bodyB=leg,
            localAnchorA=(0, 0),
            localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
            enableMotor=True,
            enableLimit=True,
            maxMotorTorque=LEG_SPRING_TORQUE,
            motorSpeed=+0.3 * i,  # low enough not to jump back into the sky
        )
        if i == -1:
            rjd.lowerAngle = +0.9 - 0.5  # The most esoteric numbers here, angled legs have freedom to travel within
            rjd.upperAngle = +0.9
        else:
            rjd.lowerAngle = -0.9
            rjd.upperAngle = -0.9 + 0.5

        leg.joint = env.world.CreateJoint(rjd)
        env.legs.append(leg)

    env.drawlist = [env.lander] + env.legs

def apply_wind(env):
    if env.legs[0].ground_contact or env.legs[1].ground_contact:
        return 
    
    # the function used for wind is tanh(sin(2 k x) + sin(pi k x)),
    # which is proven to never be periodic, k = 0.01
    wind_mag = (
        math.tanh(
            math.sin(0.02 * env.wind_idx)
            + math.sin(math.pi * 0.01 * env.wind_idx)
        )
        * env.wind_power
    )
    env.wind_idx += 1
    env.lander.ApplyForceToCenter((wind_mag, 0.0), True)
    
    # the function used for torque is tanh(sin(2 k x) + sin(pi k x)),
    # which is proven to never be periodic, k = 0.01
    torque_mag = (
        math.tanh(
            math.sin(0.02 * env.torque_idx)
            + math.sin(math.pi * 0.01 * env.torque_idx)
        )
        * env.turbulence_power
    )
    env.torque_idx += 1
    env.lander.ApplyTorque(torque_mag, True)

def apply_engines(env, action):
    # Tip is the (X and Y) components of the rotation of the lander.
    tip        = (math.sin(env.lander.angle), math.cos(env.lander.angle))

    # Side is the (-Y and X) components of the rotation of the lander.
    side       = (-tip[1], tip[0])

    # Generate two random numbers between -1/SCALE and 1/SCALE.
    dispersion = [env.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

    m_power = 0.0
    if (env.continuous and action[0] > 0.0) or (not env.continuous and action == 2):
        # Main engine
        if env.continuous:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5   # 0.5 .. 1.0
        else:
            m_power = 1.0

        # 4 is move a bit downwards, +-2 for randomness
        # The components of the impulse to be applied by the main engine.    
        ox = (
            tip[0] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            + side[0] * dispersion[1]
        )
        oy = (
            -tip[1] * (MAIN_ENGINE_Y_LOCATION / SCALE + 2 * dispersion[0])
            - side[1] * dispersion[1]
        )

        impulse_pos = (env.lander.position[0] + ox, env.lander.position[1] + oy)
        if env.render_mode is not None:
            # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
            p = create_particle(env, 3.5, impulse_pos[0], impulse_pos[1], m_power)  # 3.5 is here to make particle speed adequate
            p.ApplyLinearImpulse(
                (ox * MAIN_ENGINE_POWER * m_power, oy * MAIN_ENGINE_POWER * m_power),
                impulse_pos, True,
            )
        env.lander.ApplyLinearImpulse(
            (-ox * MAIN_ENGINE_POWER * m_power, -oy * MAIN_ENGINE_POWER * m_power),
            impulse_pos, True,
        )

    
    s_power = 0.0
    if (env.continuous and np.abs(action[1]) > 0.5) or (
        not env.continuous and action in [1, 3]
    ):
        # Orientation/Side engines
        if env.continuous:
            direction = np.sign(action[1])
            s_power   = np.clip(np.abs(action[1]), 0.5, 1.0)
        else:
            direction = action - 2   # action=1 → left (−1), action=3 → right (+1)
            s_power   = 1.0
        
        # The components of the impulse to be applied by the side engines.
        ox = tip[0] * dispersion[0] + side[0] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )
        oy = -tip[1] * dispersion[0] - side[1] * (
            3 * dispersion[1] + direction * SIDE_ENGINE_AWAY / SCALE
        )

        # The constant 17 is a constant, that is presumably meant to be SIDE_ENGINE_HEIGHT.
        # However, SIDE_ENGINE_HEIGHT is defined as 14
        # This causes the position of the thrust on the body of the lander to change, depending on the orientation of the lander.
        # This in turn results in an orientation dependent torque being applied to the lander.
        impulse_pos = (
            env.lander.position[0] + ox - tip[0] * 17 / SCALE,
            env.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE,
        )

        if env.render_mode is not None:
            # particles are just a decoration, with no impact on the physics, so don't add them when not rendering
            p = create_particle(env, 0.7, impulse_pos[0], impulse_pos[1], s_power)
            p.ApplyLinearImpulse(
                (ox * SIDE_ENGINE_POWER * s_power, oy * SIDE_ENGINE_POWER * s_power),
                impulse_pos, True,
            )
        env.lander.ApplyLinearImpulse(
            (-ox * SIDE_ENGINE_POWER * s_power, -oy * SIDE_ENGINE_POWER * s_power),
            impulse_pos, True,
        )

    return m_power, s_power

def create_particle(env, mass, x, y, ttl):
    p = env.world.CreateDynamicBody(
        position=(x, y),
        angle=0.0,
        fixtures=fixtureDef(
            shape=circleShape(radius=2 / SCALE, pos=(0, 0)),
            density=mass,
            friction=0.1,
            categoryBits=0x0100,
            maskBits=0x001,  # collide only with ground
            restitution=0.3,
        ),
    )
    p.ttl = ttl
    env.particles.append(p)
    clean_particles(env, all_particle=False)
    return p

def clean_particles(env, all_particle):
    while env.particles and (all_particle or env.particles[0].ttl < 0):
        env.world.DestroyBody(env.particles.pop(0))

def destroy_world(env):
    if not env.moon:
        return
    env.world.contactListener = None
    clean_particles(env, all_particle=True)
    env.world.DestroyBody(env.moon)
    env.moon = None
    env.world.DestroyBody(env.lander)
    env.lander = None
    env.world.DestroyBody(env.legs[0])
    env.world.DestroyBody(env.legs[1])