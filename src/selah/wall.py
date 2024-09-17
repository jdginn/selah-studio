import typing
from enum import Enum

import trimesh
import trimesh.visual
import trimesh.intersections
import numpy as np
import numpy.typing as npt

from .material import Material


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


class Wall:
    def __init__(
        self,
        name: str,
        mesh: trimesh.Trimesh,
        material: Material = Material(([1000], [0.05])),
        **kwargs,
    ):
        """Represents a wall whose shape is defined by a mesh."""
        # TODO: don't use Axis enum; instead define directions using mesh normals
        self.name = name
        self.mesh = mesh.process(True, True, True)
        self.vertices = mesh.vertices
        self.material = material
        if hasattr(kwargs, "color"):
            self.mesh.visual = typing.cast(
                trimesh.visual.ColorVisuals, kwargs.get("color")
            )

    def pos(self, height: float) -> tuple[Axis, float]:
        """Returns the position of the wall along its respective axis"""
        # For now, assume that this wall falls squarely on either the x or y axis
        # This won't work for any kind of diagonal wall but should be good enough for our needs
        #
        # TODO: fix this once we start using normals instead of Axis enum
        v = self.vertices[0]
        x, y, z = v[0], v[1], v[2]
        validX, validY, validZ = True, True, True
        for v in self.vertices:
            if x != v[0]:
                validX = False
            if y != v[1]:
                validY = False
            if z != v[2]:
                validZ = False
            if validX is False and validY is False and validZ is False:
                raise RuntimeError
        if validX:
            if validY or validZ:
                raise RuntimeError
            return Axis.X, x
        if validY:
            if validX or validZ:
                raise RuntimeError
            return Axis.Y, y
        if validX or validY:
            raise RuntimeError
        return Axis.Z, z

    def center_pos(self) -> npt.NDArray:
        """Returns the position of the center of this wall."""
        # TODO: use mesh.centroid?
        min_x, max_x, min_y, max_y, min_z, max_z = 0, 0, 0, 0, 0, 0
        for v in self.vertices:
            min_x = min(min_x, v[0])
            max_x = max(max_x, v[0])
            min_y = min(min_y, v[1])
            max_y = max(max_y, v[1])
            min_z = min(min_z, v[2])
            max_z = max(min_z, v[2])
        return np.array(
            [
                min_x + (max_x - min_x) / 2.0,
                min_y + (max_y - min_y) / 2.0,
                min_z + (max_z - min_z) / 2.0,
            ],
            dtype="float32",
        )

    def width(self, axis: Axis) -> float:
        """Returns the width of this wall. Width is perpendicular to the axis."""
        min_x, max_x, min_y, max_y, min_z, max_z = 0, 0, 0, 0, 0, 0
        for v in self.vertices:
            min_x = min(min_x, v[0])
            max_x = max(max_x, v[0])
            min_y = min(min_y, v[1])
            max_y = max(max_y, v[1])
            min_z = min(min_z, v[2])
            max_z = max(min_z, v[2])
        match axis:
            case Axis.X:
                return max_x - min_x
            case Axis.Y:
                return max_y - min_y
            case Axis.Z:
                return max_z - min_z


def build_wall_from_point(
    name: str,
    mesh: trimesh.Trimesh,
    point: npt.NDArray,
    normal: npt.NDArray,
    material: Material,
) -> Wall:
    """
    Returns a new wall on the plane defined by one point and normal. Wall is bounded by its intersection with the passed mesh.
    """

    def get_matching_index(
        list: typing.List[npt.NDArray], test: npt.NDArray
    ) -> typing.Union[int, None]:
        for i, elem in enumerate(list):
            if np.allclose(test, elem):
                return i
        return None

    mp = trimesh.intersections.mesh_plane(
        mesh,
        normal,
        point,
    )
    vertices: typing.List[npt.NDArray] = [point]
    faces: typing.List[npt.NDArray] = []
    for li, line in enumerate(mp):
        v0 = line[0]
        v1 = line[1]
        v0_idx = get_matching_index(vertices, v0)
        if v0_idx is None:
            vertices.append(v0)
            v0_idx = len(vertices) - 1
        else:
            v0 = vertices[v0_idx]
        v1_idx = get_matching_index(vertices, v1)
        if v1_idx is None:
            vertices.append(v1)
            v1_idx = len(vertices) - 1
        else:
            v1 = vertices[v1_idx]
        faces.append(np.array([0, v0_idx, v1_idx]))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    mesh.process(True, True, True)
    mesh.fill_holes()
    return Wall(name, mesh, material)
