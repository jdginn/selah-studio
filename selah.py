import argparse
import math
import typing
import zipfile
from dataclasses import dataclass
from enum import Enum

import IPython
import trimesh
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyroomacoustics as pra
import pyroomacoustics.libroom as libroom

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
    def __init__(self, name: str, mesh: trimesh.Trimesh):
        self.name = name
        self.mesh = mesh
        self.vertices = mesh.vertices

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


class Source:
    """Dispersions in degrees"""

    def __init__(self, horiz_disp: float = 60, vert_disp: float = 20) -> None:
        self.horiz_disp = horiz_disp
        self.vert_disp = vert_disp


def dir_from_points(p1: npt.NDArray, p2: npt.NDArray) -> npt.NDArray:
    unscaled = p2 - p1
    return unscaled / np.linalg.norm(unscaled)


class ListeningTriangle:

    def __init__(
        self,
        wall: Wall,
        height: float,
        dist_from_wall: float,
        dist_from_center: float,
        source: Source,
    ) -> None:
        self._wall = wall
        self.height = height
        self.dist_from_wall = dist_from_wall
        self.dist_from_center = dist_from_center
        self.source = source

        self._axis, self._wall_pos = self._wall.pos(self.height)
        match self._axis:
            case Axis.X:
                if self.dist_from_center > self._wall.width(Axis.Y) / 2.0:
                    raise RuntimeError("attempting to locate speaker outside of room")

        # TODO: need to know which direction from the wall is interior vs exterior

    def l_source(self) -> npt.NDArray:
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                return np.array(
                    [
                        self._wall_pos + self.dist_from_wall,
                        p[1] - self.dist_from_center,
                        self.height,
                    ],
                    dtype="float32",
                )
            case Axis.Y:
                raise RuntimeError
            case Axis.Z:
                raise RuntimeError

    def r_source(self) -> npt.NDArray:
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                return np.array(
                    [
                        self._wall_pos + self.dist_from_wall,
                        p[1] + self.dist_from_center,
                        self.height,
                    ],
                    dtype="float32",
                )
            case Axis.Y:
                raise RuntimeError
            case Axis.Z:
                raise RuntimeError

    def listening_pos(self) -> npt.NDArray:
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                return np.array(
                    [
                        self._wall_pos
                        + self.dist_from_wall
                        + (self.dist_from_center * math.sqrt(3)),
                        p[1],
                        self.height,
                    ],
                    dtype="float32",
                )
            case Axis.Y:
                raise RuntimeError
            case Axis.Z:
                raise RuntimeError

    def additional_walls(self, mesh: trimesh.Trimesh) -> typing.List[Wall]:
        def build_wall_from_point(name: str, point: npt.NDArray) -> Wall:
            normal = dir_from_points(point, self.listening_pos())
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
            l_wall = Wall(
                "Left Speaker Wall", trimesh.Trimesh(vertices=vertices, faces=faces)
            )
            return l_wall

        return [
            build_wall_from_point("left speaker wall", self.l_source()),
            build_wall_from_point("right speaker wall", self.r_source()),
        ]

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
                raise RuntimeError
            case Axis.Z:
                raise RuntimeError
        pass


@dataclass
class Hit:
    pos: np.ndarray
    wall: Wall
    parent: np.ndarray


@dataclass
class Shot:
    dir: npt.NDArray

    def __hash__(self) -> int:
        return f"x:{self.dir[0]}y:{self.dir[1]}z:{self.dir[2]}".__hash__()


class Room:

    # TODO: probably needs to be fleshed out better
    def __init__(self, walls: typing.List[Wall]):
        self.walls = walls

        mat = pra.Material(energy_absorption=0.1, scattering=0.2)
        _pra_walls = []
        for w in walls:
            # w.simplify()
            for tri in w.mesh.triangles:
                pw = pra.Wall(
                    tri.T,
                    mat.energy_absorption["coeffs"],
                    mat.scattering["coeffs"],
                )
                pw.name = w.name
                _pra_walls.append(pw)
        self.pra_room = pra.Room(
            _pra_walls,
            fs=44100,
            max_order=3,
            ray_tracing=True,
            air_absorption=False,
        )

    def listening_triangle(
        self,
        wall_name: str,
        height: float,
        dist_from_wall: float,
        dist_from_center: float,
        source: Source,
    ) -> None:
        self._lt = ListeningTriangle(
            self.get_wall(wall_name), height, dist_from_wall, dist_from_center, source
        )
        self.walls = self.walls + self._lt.additional_walls(self.mesh)

    # Stub to provide type awareness
    #
    # Also, most of what we use pra_room for is accessing the internal engine, since we wrap
    # most of the higher-level function in our own class here.
    @property
    def engine(self) -> libroom.Room:
        return self.pra_room.room_engine

    @property
    def mesh(self) -> trimesh.Trimesh:
        m = trimesh.util.concatenate([x.mesh for x in self.walls])
        if not isinstance(m, trimesh.Trimesh):
            raise RuntimeError
        m.fix_normals(True)
        return m

    def get_wall(self, name: str | int) -> Wall:
        for w in self.walls:
            if w.name == name:
                return w
        raise RuntimeError

    def trace(
        self,
        **kwargs,
    ) -> typing.List[typing.List[Hit]]:
        order = kwargs.get("order", 10)
        max_time = kwargs.get("max_time", 0.1)
        min_gain = kwargs.get("min_gain", -20)
        rfz_radius = kwargs.get("rfz_radius", 0.3)
        num_samples = kwargs.get("num_samples", 10)
        vert_disp: float = kwargs.get("vert_disp", 40) / 360 * 2 * math.pi
        horiz_disp: float = kwargs.get("vert_disp", 60) / 360 * 2 * math.pi
        # TODO: kwarg source selection

        speed_of_sound = 336
        max_dist = self.engine.get_max_distance()

        # TODO: factor in absorptive losses

        l_speaker, r_speaker, listen_pos = self._lt.positions()
        source = l_speaker
        source_normal = dir_from_points(source, listen_pos)

        shots: typing.List[Shot] = [Shot(source_normal)]
        i = 1
        for _ in range(num_samples):
            print(f"i: {i}")
            print(f"horiz_step: {horiz_disp / num_samples}")
            print(f"vert_step: {vert_disp/ num_samples}")
            # TODO: this is sampling is simply a horizontal stepping. It should probably
            # be pseudorandom.
            unscaled = np.array(
                # TODO: for now this only works for sources on X-axis wall!
                [
                    0,
                    -horiz_disp / num_samples * i,
                    # -horiz_disp + horiz_disp / num_samples * i,
                    vert_disp / num_samples * i,
                    # -vert_disp + vert_disp / num_samples * i,
                ]
            )
            print(f"Unscaled {unscaled}")
            # norm = np.linalg.norm(unscaled)
            # scaled = unscaled / norm
            # print(f"Scaled {scaled}")
            # print(f"Normal {source_normal}")
            print(f"Adjusted {source_normal + unscaled}")
            adjusted = source_normal + unscaled
            rescaled = adjusted / np.linalg.norm(adjusted)
            print(f"Rescaled {rescaled}")
            shots.append(Shot(rescaled))
            i = i + 1

        hits: typing.List[typing.List[Hit]] = []
        mesh = self.mesh
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        for j, shot in enumerate(shots):
            hits.append([])
            source = l_speaker
            total_dist = 0

            dir = shot.dir
            for i in range(order):
                norm: npt.NDArray = np.empty(3)
                new_dir: npt.NDArray = np.empty(3)
                new_source: npt.NDArray = np.empty(3)
                # print(f"source: {source}, dir: {dir}")
                idx_tri, idx_ray, loc = intersector.intersects_id(
                    [source],
                    [dir],
                    return_locations=True,
                    multiple_hits=True,
                )
                if len(loc) == 0:
                    raise RuntimeError
                found = False

                def min_norm(e):
                    return np.linalg.norm(source - e[0])

                for this_loc, tri_idx in sorted(
                    zip(loc, idx_tri), key=min_norm, reverse=False
                ):
                    # Check here for multiple hits
                    if np.linalg.norm(source - this_loc) > 0.001:
                        print(f"Trying {this_loc}")
                        new_source = this_loc
                        norm = mesh.face_normals[tri_idx]
                        found = True
                        break
                if not found:
                    raise RuntimeError
                new_dir = dir - norm * 2 * dir.dot(norm)
                # print(f"source: {source}, loc: {new_source}, norm:{norm}")
                print(f"source: {source}, locs: {loc}, norm:{norm}")
                # print(
                #     f"source: {source}, incident: {dir}, hit: {new_source}, reflection: {new_dir}"
                # )
                hits[j].append(Hit(new_source, None, source))
                dir = new_dir
                source = new_source

        return hits

    def draw(self, fig, ax):
        plt.scatter(
            self._lt.l_source()[0], self._lt.l_source()[1], marker="x", linewidth=8
        )
        plt.scatter(
            self._lt.listening_pos()[0],
            self._lt.listening_pos()[1],
            marker="h",
            linewidth=12,
        )
        plt.draw()

        # self.mesh.show()
        # TODO: fix magic number
        sec = self.mesh.section((0, 0, 1), (0, 0, 0.5))
        if sec is None:
            raise RuntimeError
        outline = sec.to_planar()[0]
        # IPython.embed()
        outline.apply_translation((-outline.bounds[0][0], -outline.bounds[0][1]))
        outline.plot_entities()


def animate_hits(fig, hits: typing.List[Hit]):

    def plot_single_hit(frame):
        h = hits[frame]
        point = ax.scatter(h.pos[0], h.pos[1])
        line = ax.plot([h.pos[0], h.parent[0]], [h.pos[1], h.parent[0]], marker="o")
        plt.waitforbuttonpress()
        return (point, line)

    fig.cla()
    room.draw(fig, ax)
    anim = animation.FuncAnimation(
        fig=fig, func=plot_single_hit, frames=len(hits), interval=600
    )
    plt.show()


def manually_advance_hits(fig, hits: typing.List[typing.List[Hit]]):

    plt.clf()
    room.draw(fig, ax)
    colors = ["b", "g", "r", "y", "c", "m", "y", "k"]
    for i, hh in enumerate(hits):
        for h in hh:
            plt.waitforbuttonpress()
            plt.scatter(h.pos[0], h.pos[1])
            plt.plot(
                [h.pos[0], h.parent[0]],
                [h.pos[1], h.parent[1]],
                marker="o",
                color=colors[i],
            )
            plt.draw()


def plot_hits(fig, shots: dict[Shot, typing.List[Hit]]):
    room.draw(fig, ax)
    colors = ["b", "g", "r", "y", "c", "m", "y", "k"]
    for (shot, hits), c in zip(shots.items(), colors):
        for h in hits:
            plt.scatter(h.pos[0], h.pos[1], c=c)
            plt.plot([h.pos[0], h.parent[0]], [h.pos[1], h.parent[1]], marker="o", c=c)
    plt.draw()


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

    # IPython.embed()

    scene = trimesh.load(args.file).scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    room = Room([Wall(name, mesh) for (name, mesh) in scene.geometry.items()])
    room.listening_triangle("Front", 0.8, 0.3, 0.65, Source())
    # hits = room.trace(kwargs={"vert_disp": 0})
    hits = room.trace(num_samples=0, order=15)

    fig, ax = plt.subplots()
    # room.draw(fig, ax)
    # plt.show()
    # animate_hits(fig, hits)
    # for shot, ray in hits.items():
    #     manually_advance_hits(fig, ray)
    manually_advance_hits(fig, hits)
    # plot_hits(fig, hits)
    # plt.show()

    # room.pra_room.add_source(l_speaker)
    # room.pra_room.add_microphone(critical)

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
    # room.pra_room.plot(img_order=0, mic_marker_size=0, figsize=(10, 10))
    # for hit in hits:
    #     plt.scatter(hit.pos[0], hit.pos[1], hit.pos[2], c=30)
    # plt.ylim(0, 6)
    # plt.xlim(0, 6)
    # plt.savefig("imgs/stl_room.png")
    # plt.show()
