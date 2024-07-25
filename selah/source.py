from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import math
import typing

import numpy as np
import numpy.typing as npt
import trimesh

from . import geometry

kh420_horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100}
kh420_vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30}

kh310_horiz_disp: dict[float, float] = {0: 0, 30: 0, 50: -3, 70: -6, 80: -9, 90: -20}
kh310_vert_disp: dict[float, float] = {0: 0, 30: -3, 60: -6, 90: -9, 100: -30}

@dataclass_json
@dataclass
class ShotSpecification:
    # source: str
    pitch: float = 0
    yaw: float = 0

@dataclass
class Shot:
    """
    Represents the origin of a ray of sound, including its direction, intensity,
    and any other initial information required to predict its behavior.

    Degrees in angles.
    Intensity in dB.
    """

    dir: npt.NDArray
    gain: float
    source: typing.Any = None
    spec: ShotSpecification = field(default_factory=ShotSpecification)

class Source:
    """Dispersions in degrees"""

    # Takes arguments mapping degrees to gain in dB
    def __init__(
        self,
        horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100},
        vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30},
        x_dim: float = 0.520,
        y_dim: float = 0.256,
        z_dim: float = 0.380,
        y_offset: float = 0.128,
        z_offset: float = 0.128,
        x_margin: float = 0.05,
        y_margin: float = 0.05,
        z_margin: float = 0.05,
    ):
        """
        Source represents a directional sound source.

        horiz_disp and vert_disp map dispersions angles in degrees to gain at that
        angle relative to the main acoustic axis in decibels.
        """
        self._h_x = np.array(list(horiz_disp.keys()), np.float32)
        self._h_y = np.array(list(horiz_disp.values()), np.float32)
        self._v_x = np.array(list(vert_disp.keys()), np.float32)
        self._v_y = np.array(list(vert_disp.values()), np.float32)

        self._x_dim = x_dim
        self._y_dim = y_dim
        self._z_dim = z_dim
        self._y_offset = y_offset
        self._z_offset = z_offset

    def get_shot_from_angles(self, source_pos: npt.NDArray, listening_pos: npt.NDArray, pitch: float=0, yaw: float=0) -> Shot:
        """
        Returns a shot fired from this speaker at the specified pitch and yaw offset from the direct path to the listening_pos

        Angles in degrees.
        """
        shot_spec = ShotSpecification(pitch, yaw)
        normal = geometry.dir_from_points(source_pos, listening_pos)
        pitch_rads = pitch / 180 * np.pi
        pitch_matrix = np.array(
            [
                [math.cos(pitch_rads), 0, -math.sin(pitch_rads)],
                [0, 1, 0],
                [math.sin(pitch_rads), 0, math.cos(pitch_rads)],
            ]
        )
        yaw_rads = pitch / 180 * np.pi
        yaw_matrix = np.array(
            [
                [math.cos(yaw_rads), math.sin(yaw_rads), 0],
                [-math.sin(pitch_rads), math.cos(pitch_rads), 0],
                [0, 0, 1],
            ]
        )
        new_dir = yaw_matrix.dot(pitch_matrix).dot(normal)
        new_dir = new_dir / np.linalg.norm(new_dir)
        return Shot(
            new_dir,
            self.gain(pitch, yaw),
            self,
            shot_spec,
        )

    def get_shots(self, source_pos: npt.NDArray, listening_pos: npt.NDArray, num_rays: int=1000) -> typing.List[Shot]:
        """Returns num_rays shots shot from this speaker"""
        # TODO: this should probably be an iterator rather than return a list
        shots: typing.List[Shot] = [Shot(geometry.dir_from_points(source_pos, listening_pos), 0, self)]
        SIMULATION_DISPERSION_RANGE=180
        h_steps = int(math.floor(math.sqrt(num_rays)))
        h_step_size = SIMULATION_DISPERSION_RANGE/ (h_steps - 1)
        v_steps = num_rays// h_steps
        v_step_size = SIMULATION_DISPERSION_RANGE/ (v_steps - 1)
        for v in range(v_steps):
            pitch = -SIMULATION_DISPERSION_RANGE / 2 + v_step_size * v
            for h in range(h_steps):
                yaw = -SIMULATION_DISPERSION_RANGE / 2 + h_step_size * h
                shots.append(self.get_shot_from_angles(source_pos, listening_pos, pitch, yaw))
        return shots

    def gain(self, vert_angle: float, horiz_angle: float) -> float:
        """
        Returns the gain of the source at the given angle in decibels.

        Angles in degrees.
        """
        val = np.interp(abs(vert_angle), self._v_x, self._v_y) + np.interp(
            abs(horiz_angle), self._h_x, self._h_y
        )
        if not isinstance(val, float):
            raise RuntimeError
        return val

    def test_intersection(
        self, placement: npt.NDArray, norm: npt.NDArray, test_point: npt.NDArray
    ) -> bool:
        """work in progress"""
        box = trimesh.primitives.Box(np.array([self._x_dim, self._y_dim, self._z_dim]))
        translation = trimesh.transformations.translation_matrix(
            placement - np.array([0, self._y_offset, self._z_offset])
        )
        rotation = geometry.rotation_matrix(np.array([0, 0, 0]), norm)
        return (
            box.apply_transform(translation)
            .apply_transform(rotation)
            .contains(test_point)[0]
        )
