import argparse
import math
import typing
from dataclasses import dataclass
from enum import Enum

import IPython
import trimesh
import matplotlib.patches as patches
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

SPEED_OF_SOUND = 343.0


def db(gain: float) -> float:
    return 10 * math.log10(gain)


def lineseg_dist(p: npt.NDArray, a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def dist(p: npt.NDArray, q: npt.NDArray, rs: npt.NDArray) -> float:
    x = p - q
    return np.linalg.norm(
        np.outer(np.dot(rs - q, x) / np.dot(x, x), x) + q - rs, axis=1
    )


class Axis(Enum):
    X = 1
    Y = 2
    Z = 3


material_abs = {
    "brick": 0.04,
    "gypsum": 0.05,
    "diffuser": 0.99,
    "wood": 0.1,
    "absorber": 0.95,
}

wall_materials = {
    "Default": "brick",
    "Ceiling": "brick",
    "Floor": "wood",
    "Street": "brick",
    "Doorway": "brick",
    "Front": "gypsum",
    "Back": "brick",
    "Door Absorber": "absorber",
    "Street Absorber": "absorber",
    "Door Fin": "absorber",
    "Street Fin": "absorber",
    "Back Diffuser": "diffuser",
    "Ceiling Diffuser": "diffuser",
    "Spooky Curtain": "absorber",
}


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
        self.abs = material_abs[
            wall_materials.get(self.name, wall_materials["Default"])
        ]
        # if self.name not in wall_materials:
        #     self.abs = material_abs[wall_materials["Default"]]
        # else:
        #     self.abs = material_abs[wall_materials[self.name]]

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
        rfz_radius: float,
        **kwargs,
    ) -> None:
        self._wall = wall
        self.height = height
        self.dist_from_wall = dist_from_wall
        self.dist_from_center = dist_from_center
        self.source = source
        self.rfz_radius = rfz_radius

        if kwargs.get("listen_pos") is not None:
            self._listen_pos = kwargs["listen_pos"]

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
        if hasattr(self, "_listen_pos"):
            return p + [self._listen_pos, 0, 0]
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
class Reflection:
    pos: np.ndarray
    wall: typing.Union[Wall, None]
    parent: np.ndarray
    intensity: float
    total_dist: float
    # For visualization purposes
    _color: str = ""

    def color(self, default: str) -> str:
        if self._color != "":
            return self._color
        self._color = default
        return default


@dataclass
class Shot:
    dir: npt.NDArray

    def __hash__(self) -> int:
        return f"x:{self.dir[0]}y:{self.dir[1]}z:{self.dir[2]}".__hash__()


class Arrival:
    pos: np.ndarray
    parent: Reflection
    reflection_list: typing.List[Reflection]
    # For visualization purposes
    _color: str

    def __init__(self, pos: npt.NDArray, reflections: typing.List[Reflection]):
        self.pos = pos
        self.reflection_list = reflections
        self.parent = reflections[-1]
        self.intensity = self.parent.intensity
        self.total_dist = self.parent.total_dist + np.linalg.norm(pos - self.parent.pos)
        self._color = ""

    def color(self, default: str) -> str:
        if self._color != "":
            return self._color
        self._color = default
        return default


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
        rfz_radius: float,
        **kwargs,
    ) -> None:
        self._lt = ListeningTriangle(
            self.get_wall(wall_name),
            height,
            dist_from_wall,
            dist_from_center,
            source,
            rfz_radius,
            **kwargs,
        )
        self.walls = self.walls + self._lt.additional_walls(self.mesh)

    def direct_distances(self) -> tuple[float, float]:
        return (
            float(np.linalg.norm(self._lt.l_source() - self._lt.listening_pos)),
            float(np.linalg.norm(self._lt.r_source() - self._lt.listening_pos)),
        )

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

    def faces_to_wall(self, idx: int) -> Wall:
        if not hasattr(self, "_faces_to_wall"):
            self._faces_to_wall: typing.List[Wall] = []
            for w in self.walls:
                for _ in w.mesh.faces:
                    self._faces_to_wall.append(w)
        return self._faces_to_wall[idx]

    def get_wall(self, name: str | int) -> Wall:
        for w in self.walls:
            if w.name == name:
                return w
        raise RuntimeError

    def trace(
        self,
        orig_source: npt.NDArray,
        listen_pos: npt.NDArray,
        **kwargs,
    ) -> typing.Tuple[typing.List[typing.List[Reflection]], typing.List[Arrival]]:
        order = kwargs.get("order", 10)
        max_time = kwargs.get("max_time", 0.1)
        min_gain = kwargs.get("min_gain", -20)
        num_samples = kwargs.get("num_samples", 10)
        vert_disp: float = kwargs.get("vert_disp", 50) / 360 * 2 * math.pi
        horiz_disp: float = kwargs.get("horiz_disp", 60) / 360 * 2 * math.pi
        # TODO: kwarg source selection
        self._max_time = max_time
        self._min_gain = min_gain
        source = orig_source

        source_normal = dir_from_points(source, listen_pos)
        print(f"norm: {source_normal}")
        direct_dist = np.linalg.norm(source - listen_pos)

        shots: typing.List[Shot] = [Shot(source_normal)]
        h_steps = int(math.floor(math.sqrt(num_samples)))
        h_step_size = 2 * horiz_disp / h_steps
        v_steps = num_samples // h_steps
        v_step_size = 2 * vert_disp / v_steps
        for v in range(v_steps):
            # TODO: this is sampling is simply a horizontal stepping. It should probably
            # be pseudorandom.
            for h in range(h_steps):
                unscaled = np.array(
                    # TODO: for now this only works for sources on X-axis wall!
                    [
                        0,
                        -horiz_disp / 2 + h_step_size * h,
                        -vert_disp / 2 + v_step_size * v,
                    ]
                )
                adjusted = source_normal + unscaled
                rescaled = adjusted / np.linalg.norm(adjusted)
                shots.append(Shot(rescaled))

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(-1, 1)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(-1, 1)
        ax1.plot([0, source_normal[0] * 2], [0, source_normal[1] * 2])
        ax2.plot([0, source_normal[0] * 2], [0, source_normal[2] * 2])
        for shot in shots:
            ax1.plot([0, shot.dir[0]], [0, shot.dir[1]])
        for shot in shots:
            ax2.plot([0, shot.dir[0]], [0, shot.dir[2]])
        plt.show(block=True)

        hits: typing.List[typing.List[Reflection]] = []
        arrivals: typing.List[Arrival] = []
        mesh = self.mesh
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        for j, shot in enumerate(shots):
            source = orig_source
            temp_hits: typing.List[Reflection] = []
            total_dist: float = -direct_dist
            reflected_to_rfz = False
            intensity = 1
            wall: typing.Union[Wall, None] = None

            dir = shot.dir
            for i in range(order):
                norm: npt.NDArray = np.empty(3)
                new_dir: npt.NDArray = np.empty(3)
                new_source: npt.NDArray = np.empty(3)

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
                        new_source = this_loc
                        norm = mesh.face_normals[tri_idx]
                        wall = self.faces_to_wall(tri_idx)
                        intensity = intensity * (1 - wall.abs)
                        found = True
                        break
                if not found:
                    break
                    # raise RuntimeError

                temp_hits.append(
                    Reflection(new_source, wall, source, intensity, total_dist)
                )

                # Check whether this reflection passes within the RFZ
                # dist_from_crit = float(
                #     np.linalg.norm(np.cross(new_source - source, listen_pos - source))
                #     / np.linalg.norm(new_source - source)
                # )
                # dist_from_crit = float(
                #     np.linalg.norm(
                #         # np.cross(new_source - source, listen_pos - source)
                #         np.cross(new_source - source, source - listen_pos)
                #         / np.linalg.norm(new_source - source)
                #     )
                # )
                # dist_from_crit = lineseg_dist(new_source, source, listen_pos)
                dist_from_crit = dist(new_source, source, listen_pos)
                total_dist = total_dist + float(np.linalg.norm(new_source - source))
                # Only check out to some number of ms
                if total_dist / SPEED_OF_SOUND > max_time:
                    break
                # Only check out to some minimum gain
                if db(intensity) < min_gain:
                    break
                if dist_from_crit < self._lt.rfz_radius and i > 1:
                    # We only care about rays that reflect to the RFZ
                    reflected_to_rfz = True
                    arrivals.append(Arrival(listen_pos, temp_hits.copy()))
                    if not isinstance(wall, Wall):
                        raise RuntimeError
                    print(
                        f"Reflection from {wall.name}: {db(intensity):.1f}db {source}:{shot.dir}->{new_source} at {total_dist + np.linalg.norm(listen_pos - new_source)/ SPEED_OF_SOUND * 1000:.2f}ms"
                    )

                dir = dir - norm * 2 * dir.dot(norm)
                source = new_source
            if reflected_to_rfz:
                hits.append(temp_hits)

            arrivals.sort(key=lambda a: a.total_dist)

        return (hits, arrivals)

    def draw_from_above(self):
        plt.scatter(
            self._lt.l_source()[0], self._lt.l_source()[1], marker="x", linewidth=8
        )
        plt.scatter(
            self._lt.r_source()[0], self._lt.r_source()[1], marker="x", linewidth=8
        )
        circle = plt.Circle(
            (self._lt.listening_pos()[0], self._lt.listening_pos()[1]),
            self._lt.rfz_radius,
            fill=False,
            color="dimgrey",
        )
        plt.gca().add_patch(circle)
        # plt.scatter(
        #     self._lt.listening_pos()[0],
        #     self._lt.listening_pos()[1],
        #     marker="h",
        #     linewidth=12,
        # )

        plt.draw()

        sec = self.mesh.section((0, 0, 1), (0, 0, 0.5))
        if sec is None:
            raise RuntimeError
        outline = sec.to_planar()[0]
        outline.apply_translation((-outline.bounds[0][0], -outline.bounds[0][1]))
        outline.plot_entities()

    def draw_from_side(self):
        plt.scatter(
            self._lt.l_source()[0], self._lt.l_source()[2], marker="x", linewidth=8
        )
        plt.scatter(
            self._lt.r_source()[0], self._lt.r_source()[2], marker="x", linewidth=8
        )
        circle = plt.Circle(
            (self._lt.listening_pos()[0], self._lt.listening_pos()[2]),
            self._lt.rfz_radius,
            fill=False,
            color="dimgrey",
        )
        plt.gca().add_patch(circle)
        # plt.scatter(
        #     self._lt.listening_pos()[0],
        #     self._lt.listening_pos()[2],
        #     marker="h",
        #     linewidth=12,
        # )

        plt.draw()

        sec = self.mesh.section((0, 1, 0), (0, 3, 0))
        if sec is None:
            raise RuntimeError
        outline: trimesh.path.Path2D = sec.to_planar()[0]
        outline.apply_transform(((0, -1, 0), (-1, 0, 0), (0, 0, 1)))
        # Rotate outline by 90deg
        outline.apply_translation((-outline.bounds[0][0], -outline.bounds[0][1]))
        outline.plot_entities()

    def plot_arrivals(
        self,
        fig,
        arrivals: typing.List[Arrival],
        manually_advance=False,
    ):

        colors = ["b", "g", "r", "y", "c", "m", "y", "k"]
        ax1 = fig.add_subplot(2, 2, 1)
        self.draw_from_above()
        ax2 = fig.add_subplot(2, 2, 2)
        self.draw_from_side()
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.set_xlabel("time (ms)")
        ax3.set_xlim(0, self._max_time * 1000)
        ax3.set_ylabel("intensity (dB)")
        ax3.set_ylim(self._min_gain, 0)
        for i, a in enumerate(arrivals):
            color = colors[i % len(colors)]
            ax3.bar(
                a.total_dist / SPEED_OF_SOUND * 1000,
                bottom=db(a.intensity),
                height=self._min_gain,
                color=a.color(color),
                picker=True,
            )
            for h in a.reflection_list:
                if manually_advance:
                    plt.waitforbuttonpress()
                ax1.scatter(h.pos[0], h.pos[1])
                ax1.plot(
                    [h.pos[0], h.parent[0]],
                    [h.pos[1], h.parent[1]],
                    marker="o",
                    color=h.color(color),
                    linewidth=4 * h.intensity,
                )
                ax2.scatter(h.pos[0], h.pos[2])
                ax2.plot(
                    [h.pos[0], h.parent[0]],
                    [h.pos[2], h.parent[2]],
                    marker="o",
                    color=h.color(color),
                    linewidth=4 * h.intensity,
                )
                plt.draw()

    def plot_arrivals_interactive(
        self,
        fig,
        arrivals: typing.List[Arrival],
        manually_advance=False,
    ):
        self._curr_arrival = -1
        orig_arrivals = arrivals

        def on_pick(event):
            EPS = 1
            if isinstance(event.artist, patches.Rectangle):
                rect = event.artist
                for i, arrival in enumerate(arrivals):
                    if (
                        abs((arrival.total_dist / SPEED_OF_SOUND * 1000) - rect.get_x())
                        < EPS
                    ):
                        self._curr_arrival = i
                        self.plot_arrivals(
                            fig,
                            [arrival],
                            False,
                        )

        def on_press(event):
            match event.key:
                case "x":
                    self.plot_arrivals(fig, orig_arrivals, manually_advance)
                case "backspace":
                    self.plot_arrivals(fig, orig_arrivals, manually_advance)
                case "right":
                    self._curr_arrival += 1
                    self.plot_arrivals(
                        fig,
                        [orig_arrivals[self._curr_arrival % len(orig_arrivals)]],
                        manually_advance,
                    )
                case "left":
                    self._curr_arrival -= 1
                    self.plot_arrivals(
                        fig,
                        [orig_arrivals[self._curr_arrival % len(orig_arrivals)]],
                        manually_advance,
                    )

        fig.canvas.mpl_connect("pick_event", on_pick)
        fig.canvas.mpl_connect("key_press_event", on_press)
        self.plot_arrivals(fig, arrivals, manually_advance)


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

    scene = trimesh.load(args.file)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    scene = scene.scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    room = Room([Wall(name, mesh) for (name, mesh) in scene.geometry.items()])
    room.listening_triangle(
        wall_name="Front",
        height=0.8,
        dist_from_wall=0.4,
        dist_from_center=1.6,
        source=Source(),
        listen_pos=1.9,
        rfz_radius=0.25,
    )
    l_speaker, r_speaker, _ = room._lt.positions()
    (_, l_arrivals) = room.trace(
        l_speaker,
        room._lt.listening_pos(),
        num_samples=1_00,
        max_time=50 / 1000,
        min_gain=-20,
        order=50,
        horiz_disp=60,
        vert_disp=50,
    )
    (_, r_arrivals) = room.trace(
        r_speaker,
        room._lt.listening_pos(),
        num_samples=1_00,
        max_time=50 / 1000,
        min_gain=-20,
        order=50,
        horiz_disp=60,
        vert_disp=50,
    )
    arrivals = l_arrivals + r_arrivals
    arrivals.sort(key=lambda a: a.total_dist)

    plt.ion()
    fig = plt.figure()
    room.plot_arrivals_interactive(fig, arrivals, False)
    plt.show(block=True)
