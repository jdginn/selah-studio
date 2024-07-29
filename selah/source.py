from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import typing

import numpy.typing as npt

from selah.wall import Wall
from selah.exceptions import SelahException


class ShotException(SelahException):
    """Indicates an exception while processing a shot"""

    def __init__(self, shot: "Shot"):
        self.shot = shot


class ReflectionException(SelahException):
    """Indicates an exception while processing a reflection"""


@dataclass
class Source:
    """
    Represents a ray that arrives at a target zone

    Target zone is typically a reflection-free zone
    """

    pos: npt.NDArray
    gain: float
    total_dist: float

    def color(self, default: str) -> str:
        if hasattr(self, "_color"):
            return self._color
        self._color = default
        return default


@dataclass_json
@dataclass
class ShotSpecification:
    # source: str
    pitch: float = 0
    yaw: float = 0


@dataclass
class Shot(Source):
    """
    Represents the origin of a ray of sound, including its direction, intensity,
    and any other initial information required to predict its behavior.

    Degrees in angles.
    Intensity in dB.
    """

    dir: npt.NDArray
    source: typing.Any
    spec: ShotSpecification = field(default_factory=ShotSpecification)


@dataclass
class Reflection(Source):
    """
    Represents a discrete sound reflection off of a surface.
    """

    parent: "Source"
    wall: Wall

    def shot(self) -> Shot:
        while True:
            if isinstance(self.parent, Shot):
                return self.parent
            if not isinstance(self.parent, "Reflection"):
                raise ReflectionException("Reflection does not originate from a shot")

    #
    # def total_dist(self) -> float:
    #     total_dist: float = 0
    #     while True:
    #         if isinstance(self.parent, Shot):
    #             total_dist += float(np.linalg.norm(self.pos - self.parent.pos))
    #             return total_dist
    #         if isinstance(self.parent, "Reflection"):
    #             total_dist += float(np.linalg.norm(self.pos - self.parent.pos))
    #         raise ReflectionException("Invalid parent type for reflection")
