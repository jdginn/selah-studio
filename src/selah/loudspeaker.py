import math
import typing
from enum import Enum

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
        x_dim: float,
        y_dim: float,
        z_dim: float,
        y_offset: float,
        z_offset: float,
        horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100},
        vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30},
        x_margin: float = 0.05,
        y_margin: float = 0.05,
        z_margin: float = 0.05,
    ):
        """
        LouspeakerSpec represents a the left speaker for a certain kind of directional louspeaker.
        The right speaker will be produced by flipping the y-plane around the z-axis.

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


class Side(Enum):
    LEFT = 0
    RIGHT = 1


class Loudspeaker:
    # Reference normal against which all new normals are referenced
    ref_vec = np.array([-1, 0, 0])

    def __init__(
        self,
        spec: LoudspeakerSpec,
        position: typing.Union[npt.NDArray, list[float]] = np.array([0, 0, 0]),
        normal: typing.Union[npt.NDArray, list[float]] = np.array([-1, 0, 0]),
        side: Side = Side.LEFT,
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
        self._side = side
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

        Mesh is already rotated and translated to the correct position.
        """
        if not hasattr(self, "_mesh"):
            extents = np.array([self.spec._x_dim, self.spec._y_dim, self.spec._z_dim])
            mesh = trimesh.primitives.Box(np.array(extents))

            match self._side:
                case Side.LEFT:
                    mesh.apply_translation(
                        # normalize so bottom left corner is at [0, 0, 0]
                        np.array([-extents[0], extents[1], extents[2]]) / 2
                        # normalize so acoustic center is at [0, 0, 0]
                        - [0, self.spec._y_offset, self.spec._z_offset]
                    )
                case Side.RIGHT:
                    mesh.apply_translation(
                        # normalize so bottom left corner is at [0, 0, 0]
                        np.array([-extents[0], extents[1], extents[2]]) / 2
                        # normalize so acoustic center is at [0, 0, 0]
                        - [
                            0,
                            (self.spec._y_dim - self.spec._y_offset),
                            self.spec._z_offset,
                        ]
                    )
            ref_vec = -self.ref_vec
            angle = trimesh.transformations.angle_between_vectors(
                ref_vec, self.normal, True
            )
            axis = np.cross(ref_vec, self.normal)
            if np.allclose(axis, [0, 0, 0]):
                axis = np.array([0, 0, 1])
            mesh.apply_transform(
                trimesh.transformations.rotation_matrix(
                    angle, axis, np.array([0, 0, 0])
                )
            )

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
        # If some of our vertices are inside and some are outside, we need to consider whether the edge between them intersects a face
        contained_points = test_mesh.contains(self.mesh.vertices)
        if any(contained_points) and not all(contained_points):
            scene = trimesh.Scene()
            scene.add_geometry(test_mesh)
            scene.add_geometry(self.mesh)
            # Find edges between vertex pairs where one is inside and one is outside
            for index, contained in enumerate(contained_points):
                if not contained:
                    for neighbor in self.mesh.vertex_neighbors[index]:
                        if contained_points[neighbor]:
                            scene.add_geometry(
                                trimesh.PointCloud(
                                    [
                                        self.mesh.vertices[index],
                                    ]
                                )
                            )
                            intersection = True
                            return True
            scene.show()
        return intersection
