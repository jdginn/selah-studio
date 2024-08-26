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


class LoudspeakerSpec:
    # Takes arguments mapping degrees to gain in dB
    def __init__(
        self,
        x_dim: float = 0.520,
        y_dim: float = 0.256,
        z_dim: float = 0.380,
        y_offset: float = 0.128,
        z_offset: float = 0.128,
        horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100},
        vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30},
        x_margin: float = 0.05,
        y_margin: float = 0.05,
        z_margin: float = 0.05,
    ):
        """
        LouspeakerSpec represents a certain kind of directional louspeaker.

        Speaker dimensions assume the drivers are on the plane of X=0.

        Parameters
        ----------
        horiz_disp : dict[float, float]
            maps dispersion angle in degrees to gain in dB relative to acoustic axis
            (implicitly assuemes 0deg : 0dB)
        vert_disp : dict[float, float]
            maps dispersion angle in degrees to gain in dB relative to acoustic axis
            (implicitly assuemes 0deg : 0dB)
        x_dim : float
            dimension of speaker on x axis
        y_dim : float
            dimension of speaker on y axis
        z_dim : float
            dimension of speaker on z axis
        y_offset: float
            offset of the acoustic axis from [0, 0, 0] (i.e. the front bottom left corner)
        z_offset: float
            offset of the acoustic axis from [0, 0, 0] (i.e. the front bottom left corner)
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


class Loudspeaker:
    # Takes arguments mapping degrees to gain in dB
    def __init__(
        self,
        spec: LoudspeakerSpec,
        position: typing.Union[npt.NDArray, list[float]] = np.array([0, 0, 0]),
        normal: typing.Union[npt.NDArray, list[float]] = np.array([-1, 0, 0]),
    ):
        """
        Loudspeaker represents a specific louspeaker at a specific location in space.

        Parameters
        ----------
        spec : LoudspeakerSpec
           Defines the specifics of the kind of speaker
        position : (3,1) float
            Location of the acoustic axis in 3-dimensional space
        normal: (3,1) float
            Normal vector from acoustic axis
        """
        self.spec = spec
        if isinstance(position, list):
            position = np.array(position)
        self.position = position
        if isinstance(normal, list):
            normal = np.array(normal)
        self.normal = normal

    @property
    def mesh(self) -> trimesh.Trimesh:
        """
        Returns a mesh representing this loudspeaker.

        Mesh always places the front, bottom, left corner at the origin.
        """
        if not hasattr(self, "_mesh"):
            extents = np.array([self.spec._x_dim, self.spec._y_dim, self.spec._z_dim])
            mesh = trimesh.primitives.Box(np.array(extents))
            mesh.apply_translation(
                extents / 2 - [0, self.spec._y_offset, self.spec._z_offset]
            )

            if not np.allclose(self.normal, np.array([-1, 0, 0])):
                angle = trimesh.transformations.angle_between_vectors(
                    np.array([-1, 0, 0]), self.normal
                )
                axis = np.cross(np.array([-1, 0, 0]), self.normal)
                rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
                mesh.apply_transform(rotation_matrix)
            mesh.apply_translation(self.position)

            mesh.visual.vertex_colors = tv.random_color()  # pyright: ignore
            self._mesh = mesh
        return self._mesh

    def get_shot_from_angles(
        self,
        listening_pos: npt.NDArray,
        pitch: float = 0,
        yaw: float = 0,
    ) -> Shot:
        """
        Returns a shot fired from this speaker at the specified pitch and yaw offset from the direct path to the listening_pos

        Angles in degrees.
        """
        shot_spec = ShotSpecification(pitch, yaw)
        normal = geometry.dir_from_points(self.position, listening_pos)
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
            self.position,
            sound.from_db(self.spec.gain(pitch, yaw)),
            0,
            new_dir,
            self,
            shot_spec,
        )

    def get_shots(
        self, listening_pos: npt.NDArray, num_rays: int = 1000
    ) -> typing.List[Shot]:
        """Returns shots to be shot from this speaker"""
        # TODO: this should probably be an iterator rather than return a list
        shots: typing.List[Shot] = [
            Shot(
                self.position,
                1.0,
                0,
                geometry.dir_from_points(self.position, listening_pos),
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
                shots.append(self.get_shot_from_angles(listening_pos, pitch, yaw))
        return shots

    def test_intersection(
        self,
        test_mesh: trimesh.Trimesh,
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
        contained_points = test_mesh.contains(self.mesh.vertices)

        # NOTE: it would seem like we could take a shortcut here and return False
        # if no points are contaiend. However, that will not correctly handle the
        # case where all our vertices intersect the test mesh.

        intersection = False
        # Check whether each vertex intersects the mesh
        pq = prox.ProximityQuery(test_mesh)
        points_on_surface, distance_to_surface, _ = pq.on_surface(self.mesh.vertices)
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
