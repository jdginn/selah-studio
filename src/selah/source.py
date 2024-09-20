from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import typing

import numpy as np
import numpy.typing as npt

from selah.wall import Wall
from selah.exceptions import SelahException


class SourceException(ABC, SelahException):
    """
    Exception for issues tracing sources

    Supports dumping debug data to JSON
    """

    @property
    @abstractmethod
    def source(self) -> "Source":
        pass

    @abstractmethod
    def to_json(self) -> str:
        """
        Returns debug information formatted as a json string
        """
        pass


class ShotException(SourceException):
    """Indicates an exception while processing a shot"""

    def __init__(self, shot: "Shot", message: str):
        self.shot = shot
        self.message = message

    @property
    def source(self) -> "Shot":
        return self.shot

    def to_json(self) -> str:
        return self.shot.spec.to_json()


class ReflectionException(SourceException):
    """Indicates an exception while processing a reflection"""

    def __init__(self, reflection: "Reflection", message: str):
        self.reflection = reflection
        self.message = message

    @property
    def source(self) -> "Reflection":
        return self.reflection

    def to_json(self) -> str:
        return self.reflection.to_json()


@dataclass
class Source:
    """
    Represents a ray that arrives at a target zone

    Target zone is typically a reflection-free zone
    """

    pos: npt.NDArray
    gain: float

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


@dataclass_json
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
                raise ReflectionException(
                    self, "Reflection does not originate from a shot"
                )

    @property
    def total_dist(self) -> float:
        total_dist: float = 0
        while True:
            if isinstance(self.parent, Shot):
                segment_length = float(np.linalg.norm(self.pos - self.parent.pos))
                total_dist = total_dist + float(
                    np.linalg.norm(self.pos - self.parent.pos)
                )
                return total_dist
            if isinstance(self.parent, Reflection):
                segment_length = float(np.linalg.norm(self.pos - self.parent.pos))
                total_dist = self.parent.total_dist + segment_length
                return total_dist
            raise ReflectionException(
                self, f"Invalid parent type for reflection: {type(self.parent)}"
            )
