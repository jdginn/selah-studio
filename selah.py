import argparse
from math import dist
from dataclasses import dataclass
import zipfile
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import pyroomacoustics.libroom as libroom

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pprint
import pyfqmr
import typing
import math
from enum import Enum

import IPython

namespace = {"schema": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

# material_map = {
#     # "Default": pra.materials_data["brick_wall_rough"],
#     # "Arch": pra.materials_data["brick_wall_rough"],
#     "Default": pra.materials_data["brickwork"],
#     "Arch": pra.materials_data["brickwork"],
# }


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


class Material:
    # name: string
    # absorption: ?
    # scattering: ?
    # diffusion: ?
    def __init__(self):
        pass


class Wall:
    #   name: string
    #   material: string
    #   vertices: numpy.ndarray
    #       array of vertices of shape (n_vertices, 3)
    #   faces: numpy.ndarray
    #       array of faces of shape (n_faces, 3)
    def __init__(self, tree: ET.Element):
        self.name = tree.get("name")
        self._pra_walls = typing.List[pra.Wall]
        mesh = tree.find("schema:mesh", namespace)
        if mesh is None:
            raise RuntimeError
        meshv = mesh.find("schema:vertices", namespace)
        if meshv is None:
            raise RuntimeError
        vertices = [v for v in meshv.iter()]
        self.vertices = np.ndarray(shape=(len(vertices) - 1, 3))
        for i, v in enumerate(vertices[1:]):
            self.vertices[i] = [v.get("x"), v.get("y"), v.get("z")]
        mesht = mesh.find("schema:triangles", namespace)
        if mesht is None:
            raise RuntimeError
        triangles = [t for t in mesht.iter()]
        self.triangles = np.ndarray(shape=(len(triangles) - 1, 3))
        for i, t in enumerate(triangles[1:]):
            self.triangles[i] = [t.get("v1"), t.get("v2"), t.get("v3")]

    def simplify(self):
        # TODO: make this take the simplifier as a closure
        simplifier = pyfqmr.Simplify()
        simplifier.setMesh(self.vertices, self.triangles)
        simplifier.simplify_mesh(
            target_count=60, aggressiveness=100, preserve_border=True, verbose=10
        )
        self.vertices, self.triangles, _ = simplifier.getMesh()

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


class ListeningTriangle:

    def __init__(
        self, wall: Wall, height: float, dist_from_wall: float, dist_from_center: float
    ) -> None:
        self._wall = wall
        self.height = height
        self.dist_from_wall = dist_from_wall
        self.dist_from_center = dist_from_center

        self._axis, self._wall_pos = self._wall.pos(self.height)
        # TODO: need to know which direction from the wall is interior vs exterior

    def positions(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                if self.dist_from_center > self._wall.width(Axis.Y) / 2.0:
                    raise RuntimeError("attempting to locate speaker outside of room")
                return (
                    np.array(
                        [
                            self._wall_pos + self.dist_from_wall,
                            p[1] - self.dist_from_center,
                            self.height,
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            self._wall_pos + self.dist_from_wall,
                            p[1] + self.dist_from_center,
                            self.height,
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            self._wall_pos
                            + self.dist_from_wall
                            + (self.dist_from_center * math.sqrt(3)),
                            p[1],
                            self.height,
                        ],
                        dtype="float32",
                    ),
                )
            case Axis.Y:
                axis = Axis.X
            case Axis.Z:
                raise RuntimeError
        pass


@dataclass
class Hit:
    pos: np.ndarray
    wall: Wall
    parent: typing.Union[Wall, np.ndarray]


class Room:

    # TODO: probably needs to be fleshed out better
    def __init__(self, walls: typing.List[Wall]):
        self.walls = walls
        reduc = 1000.0
        mat = pra.Material(energy_absorption=0.1, scattering=0.2)
        _pra_walls = []
        for w in walls:
            w.simplify()
            for tri in w.triangles:
                corner = np.array(
                    [
                        w.vertices[int(tri[0])],
                        w.vertices[int(tri[1])],
                        w.vertices[int(tri[2])],
                    ],
                    dtype=np.float32,
                )
                _pra_walls.append(
                    pra.Wall(
                        corner.T / reduc,
                        mat.energy_absorption["coeffs"],
                        mat.scattering["coeffs"],
                    )
                )
        self.pra_room = pra.Room(
            _pra_walls,
            fs=44100,
            max_order=3,
            ray_tracing=True,
            air_absorption=False,
        )

    # Stub to provide type awareness
    #
    # Also, most of what we use pra_room for is accessing the internal engine, since we wrap
    # most of the higher-level function in our own class here.
    @property
    def engine(self) -> libroom.Room:
        return self.pra_room.room_engine

    def get_wall(self, name: str | int) -> Wall:
        for w in self.walls:
            if w.name == name:
                return w
        raise RuntimeError

    def trace(self, lt: ListeningTriangle) -> typing.List[Hit]:
        max_dist = self.engine.get_max_distance()
        l_speaker, r_speaker, crit = lt.positions()
        hits: typing.List[Hit] = [Hit(l_speaker / 1000, None, None)]
        next_hit, next_wall_index, hit_dist = self.engine.next_wall_hit(
            l_speaker / 1000, crit / 1000 + max_dist, False
        )
        print(f"Reflection location: {next_hit}")
        w: libroom.Wall = self.engine.get_wall(next_wall_index)
        print(f"Wall index: {next_wall_index}")
        hits.append(Hit(next_hit, w, l_speaker))
        p2 = np.empty([3, 1], dtype="float32")
        w.reflect(l_speaker, p2)
        print(f"Reflected direction: {p2}")
        p2 += max_dist
        order = 15
        for i in range(order):
            next_hit, next_wall_index, hit_dist = self.engine.next_wall_hit(
                l_speaker / 1000, crit / 1000 + self.engine.get_max_distance(), False
            )
            # IPython.embed()
            # wall: libroom.Wall = self.engine.get_wall(next_wall_index)
            # hits.append(Hit(next_hit, self.engine.get_wall(next_wall_index), l_speaker))
            print("Next hit:")
            pprint.pprint(next_hit)
            hits.append(Hit(next_hit, None, hits[i]))
        return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process room from 3mf file")
    parser.add_argument("--file", type=str, required=True, help="Path to 3mf file")
    parser.add_argument(
        "-simplify",
        action="store_const",
        const=True,
        help="Set to simplify meshes for faster compute",
    )
    args = parser.parse_args()

    z = zipfile.ZipFile(args.file)
    f = z.extract("3D/3dmodel.model")
    tree = ET.parse(f)
    res = tree.getroot().find("schema:resources", namespace)
    if res is None:
        raise RuntimeError
    objects = [Wall(o) for o in res.findall("schema:object", namespace)]
    room = Room(objects)
    print("Front")
    axis, pos = room.get_wall("Front").pos(0)
    pprint.pprint((axis, pos))
    print("center:")
    pprint.pprint(room.get_wall("Front").center_pos())
    print("width:")
    pprint.pprint(room.get_wall("Front").width(Axis.Y))
    lt = ListeningTriangle(room.get_wall("Front"), 1800, 300, 650)
    # room.pra_room.add_source(l_speaker / 1000)
    # room.pra_room.add_microphone(critical / 1000)
    hits = room.trace(lt)
    # IPython.embed()

    # # compute the rir
    # # room.pra_room.image_source_model()
    # room.pra_room.ray_tracing()
    # room.pra_room.compute_rir()
    # # for s in room.pra_room.sources:
    # #     s.set_ordering("order")
    # #     for i in s.images:
    # #         print("image:")
    # #         print(i)
    # room.pra_room.plot_rir()
    # plt.xlim(0, 60)
    # plt.savefig("imgs/stl_rir_plot.png")
    # plt.show()

    # show the room
    room.pra_room.plot(img_order=0, mic_marker_size=0, figsize=(10, 10))
    for hit in hits:
        pprint.pprint(hit.pos)
        plt.scatter(hit.pos[0], hit.pos[1], hit.pos[2], c=30)
    plt.ylim(0, 6)
    plt.xlim(0, 6)
    plt.savefig("imgs/stl_room.png")
    plt.show()
