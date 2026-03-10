from gymnasium.error import DependencyNotInstalled

try:
    import Box2D
    from Box2D.b2 import (
        circleShape,
        contactListener,
        edgeShape,
        fixtureDef,
        polygonShape,
        revoluteJointDef,
    )
except ImportError as e:
    raise DependencyNotInstalled(
        'Box2D is not installed, you can install it by run `pip install swig` followed by `pip install "gymnasium[box2d]"`'
    ) from e

__all__ = [
    "Box2D",
    "circleShape",
    "contactListener",
    "edgeShape",
    "fixtureDef",
    "polygonShape",
    "revoluteJointDef",
]
