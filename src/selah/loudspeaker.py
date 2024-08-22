import math
import typing

import numpy as np
import numpy.typing as npt
import trimesh
import trimesh.visual as tv
import trimesh.proximity as prox

from . import geometry
from . import sound

from selah.source import Shot, ShotSpecification

kh420_horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100}
kh420_vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30}

kh310_horiz_disp: dict[float, float] = {0: 0, 30: 0, 50: -3, 70: -6, 80: -9, 90: -20}
kh310_vert_disp: dict[float, float] = {0: 0, 30: -3, 60: -6, 90: -9, 100: -30}


class Loudspeaker:
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
        Louspeaker represents a directional louspeaker.

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

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Returns a mesh representing this loudspeaker.

        Mesh always places the front, bottom, left corner at the origin.
        """
        extents = np.array([self._x_dim, self._y_dim, self._z_dim])
        mesh = trimesh.primitives.Box(np.array(extents))
        mesh.apply_translation(extents / 2)
        mesh.visual.vertex_colors = tv.random_color()  # pyright: ignore
        return mesh

    def get_shot_from_angles(
        self,
        source_pos: npt.NDArray,
        listening_pos: npt.NDArray,
        pitch: float = 0,
        yaw: float = 0,
    ) -> Shot:
        """
        Returns a shot fired from this speaker at the specified pitch and yaw offset from the direct path to the listening_pos

        Angles in degrees.
        """
        shot_spec = ShotSpecification(pitch, yaw)
        normal = geometry.dir_from_points(source_pos, listening_pos)
        pitch_rads = pitch / 180 * math.pi
        pitch_matrix = np.array(
            [
                [math.cos(pitch_rads), 0, -math.sin(pitch_rads)],
                [0, 1, 0],
                [math.sin(pitch_rads), 0, math.cos(pitch_rads)],
            ]
        )
        yaw_rads = yaw / 180 * math.pi
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
            source_pos,
            sound.from_db(self.gain(pitch, yaw)),
            0,
            new_dir,
            self,
            shot_spec,
        )

    def get_shots(
        self, source_pos: npt.NDArray, listening_pos: npt.NDArray, num_rays: int = 1000
    ) -> typing.List[Shot]:
        """Returns shots to be shot from this speaker"""
        # TODO: this should probably be an iterator rather than return a list
        shots: typing.List[Shot] = [
            Shot(
                source_pos,
                1.0,
                0,
                geometry.dir_from_points(source_pos, listening_pos),
                self,
            )
        ]
        SIMULATION_DISPERSION_RANGE = 180
        h_steps = int(math.floor(math.sqrt(num_rays)))
        h_step_size = SIMULATION_DISPERSION_RANGE / (h_steps - 1)
        v_steps = num_rays // h_steps
        v_step_size = SIMULATION_DISPERSION_RANGE / (v_steps - 1)
        for v in range(v_steps):
            pitch = -SIMULATION_DISPERSION_RANGE / 2 + v_step_size * v
            for h in range(h_steps):
                yaw = -SIMULATION_DISPERSION_RANGE / 2 + h_step_size * h
                shots.append(
                    self.get_shot_from_angles(source_pos, listening_pos, pitch, yaw)
                )
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
        self,
        test_mesh: trimesh.Trimesh,
        placement: typing.Union[npt.NDArray, list[float]] = np.array([0, 0, 0]),
        norm: typing.Union[None, npt.NDArray, list[float]] = None,
    ) -> bool:
        """
        Returns True if this loudspeaker intersects another mesh.

        Parameters
        ------------
        test_mesh:  mesh of object to check for intersection
        placement:  3D position of the acoustic axis of this loudspeaker
        norm:       vector describing the direction of this loudspeaker is pointed
        """
        # TODO:
        if isinstance(placement, list):
            placement = np.array(placement)
        if isinstance(norm, list):
            norm = np.array(placement)
        mesh = self.mesh.copy()
        if norm is not None:
            angle = trimesh.transformations.angle_between_vectors(
                np.array([1, 0, 0]), norm
            )
            axis = np.cross(np.array([1, 0, 0]), norm)
            rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
            mesh.apply_transform(rotation_matrix)
        # Position is measured from the lower, front, left corner
        # TODO: is corner position adjustment really doing what we need?
        corner_position = placement - np.array([0, self._y_offset, self._z_offset])
        contained_points = test_mesh.contains(mesh.vertices + corner_position)

        # NOTE: it would seem like we could take a shortcut here and return False
        # if no points are contaiend. However, that will not correctly handle the
        # case where all our vertices intersect the test mesh.

        intersection = False
        # Check whether each vertex intersects the mesh
        pq = prox.ProximityQuery(test_mesh)
        points_on_surface, distance_to_surface, _ = pq.on_surface(
            mesh.vertices + corner_position
        )
        for i, dist in enumerate(distance_to_surface):
            if dist == 0:
                intersection = True
                print(
                    f"Intersection at point [{points_on_surface[i][0]}, {points_on_surface[i][1]}, {points_on_surface[i][2]}]"
                )
        # If some of our vertices are inside and some are outside, we need to consider whether the edge between them intersects a face
        if any(contained_points) and not all(contained_points):
            print("Some edges straddle")
            # Find edges between vertex pairs where one is inside and one is outside
            for index, contained in enumerate(contained_points):
                if not contained:
                    for neighbor in self.mesh.vertex_neighbors[index]:
                        if contained_points[neighbor]:
                            intersection = True
                            return True
        return intersection
