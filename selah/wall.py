import typing
from enum import Enum

import trimesh
import numpy as np
import numpy.typing as npt

from .material import Material


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


class Wall:

    def __init__(
        self, name: str, mesh: trimesh.Trimesh, material: Material = Material(0.05)
    ):
        self.name = name
        self.mesh = mesh
        self.vertices = mesh.vertices
        self.material = material

    def pos(self, height: float) -> tuple[Axis, float]:
        # For now, assume that this wall falls squarely on either the x or y axis
        # This won't work for any kind of diagonal wall but should be good enough for our needs
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
    mp = trimesh.intersections.mesh_plane(
        mesh,
        normal,
        point,
    )
    vertices: typing.List[npt.NDArray] = [point]
    faces: typing.List[npt.NDArray] = []
    for line in mp:
        vertices.append(line[0])
        vertices.append(line[1])
        if len(vertices) > 3:
            faces.append(np.array([0, len(vertices) - 3, len(vertices) - 2]))
        faces.append(np.array([0, len(vertices) - 2, len(vertices) - 1]))
    return Wall(name, trimesh.Trimesh(vertices=vertices, faces=faces), material)
