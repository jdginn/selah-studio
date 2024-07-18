import argparse
import math
import typing
from dataclasses import dataclass
import dataclasses
import collections
from enum import Enum

import pygad
import pprint
import trimesh
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyroomacoustics as pra
import pyroomacoustics.libroom as libroom

SPEED_OF_SOUND = 343.0


class SelahStudioException(RuntimeError):
    pass


class ObscuresWindow(SelahStudioException):
    """Indicates an attempt to create a wall that obscures the window"""


class ListeningPositionError(SelahStudioException):
    """Indicates the listening position has been placed outside the valid area"""


def db(gain: float) -> float:
    return 10 * math.log10(gain)


def from_db(gain: float) -> float:
    return math.pow(10, gain / 10)


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

def rotation_matrix(A: npt.NDArray,B: npt.NDArray) -> npt.NDArray:
# a and b are in the form of numpy array

   ax = A[0]
   ay = A[1]
   az = A[2]

   bx = B[0]
   by = B[1]
   bz = B[2]

   au = A/(np.sqrt(ax*ax + ay*ay + az*az))
   bu = B/(np.sqrt(bx*bx + by*by + bz*bz))

   R=np.array([[bu[0]*au[0], bu[0]*au[1], bu[0]*au[2]], [bu[1]*au[0], bu[1]*au[1], bu[1]*au[2]], [bu[2]*au[0], bu[2]*au[1], bu[2]*au[2]] ])


   return(R)


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
    "weak_scatter": 0.25,
}

wall_materials = {
    "Default": "brick",
    "Ceiling": "brick",
    "Floor": "wood",
    "Street": "brick",
    "Doorway": "brick",
    "Front": "weak_scatter",
    "Back": "brick",
    "Door Absorber": "absorber",
    "Street Absorber": "absorber",
    "Back Diffuser": "diffuser",
    "Back Diffuser Top": "wood",
    "Ceiling Diffuser": "absorber",
    "Cutout Top": "absorber",
    "Street Absorber": "absorber",
    "Street Absorber Top": "wood",
    "Door Absorber": "absorber",
    "Door Absorber Top": "wood",
    "Door Front Absorber": "absorber",
    "Door Front Absorber Top": "wood",
    "Door Back Absorber": "absorber",
    "Door Back Absorber Top": "wood",
    "Door Door": "absorber",
    "Floor Absorber": "absorber",
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


def build_wall_from_point(
    name: str, mesh: trimesh.Trimesh, point: npt.NDArray, normal: npt.NDArray
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
    return Wall(name, trimesh.Trimesh(vertices=vertices, faces=faces))


def test_intersection(
    mesh: trimesh.Trimesh, point: npt.NDArray, normal: npt.NDArray
) -> bool:
    mp = trimesh.intersections.mesh_plane(mesh, normal, point)
    return len(mp) > 0


kh420_horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100}
kh420_vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30}

kh310_horiz_disp: dict[float, float] = {0: 0, 30: 0, 50: -3, 70: -6, 80: -9, 90: -20}
kh310_vert_disp: dict[float, float] = {0: 0, 30: -3, 60: -6, 90: -9, 100: -30}


class Source:
    """Dispersions in degrees"""

    # Takes arguments mapping degrees to gain in dB
    def __init__(
        self,
        horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100},
        vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30},
        x_dim: float,
        y_dim: float,
        z_dim: float,
        y_offset: float,
        z_offset: float,
        x_margin: float,
        y_margin: float,
        z_margin: float,
    ):
        self._h_x = np.array(list(horiz_disp.keys()), np.float32)
        self._h_y = np.array(list(horiz_disp.values()), np.float32)
        self._v_x = np.array(list(vert_disp.keys()), np.float32)
        self._v_y = np.array(list(vert_disp.values()), np.float32)

        self._x_dim = x_dim
        self._y_dim = y_dim
        self._z_dim = z_dim
        self._y_offset= y_offset
        self._z_offset= x_offset

    def intensity(self, vert_pos: float, horiz_pos: float) -> float:
        val = np.interp(abs(vert_pos), self._v_x, self._v_y) + np.interp(
            abs(horiz_pos), self._h_x, self._h_y
        )
        if not isinstance(val, float):
            raise RuntimeError
        return val

    def test_intersection(self, placement: npt.NDArray, norm: npt.NDArray, test_point: npt.NDArray) -> bool:
        box = trimesh.primitives.Box(np.array([self._x_dim, self._y_dim, self._z_dim]))
        translation = trimesh.transformations.translation_matrix(placement - np.array([0, self._y_offset, self._z_offset]))
        rotation = rotation_matrix(np.array([0, 0, 0]), norm)
        return box.apply_transform(translation).apply_transform(rotation).contains(test_point)[0]

    def test_intersection_plane(self, placement: npt.NDArray, norm: npt.NDArray, test_point: npt.NDArray) -> bool:
        translated_test_point = test_point
        return self.mesh.contains(translated_test_point)[0]

    def test_intersection_with_margin(self, placement: npt.NDArray, norm: npt.NDArray, test_point: npt.NDArray) -> bool:
        translated_test_point = test_point
        return self.mesh.contains(translated_test_point)[0]


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
        self.speaker_height = kwargs.get("speaker_height", height)
        self.dist_from_wall = dist_from_wall
        self.dist_from_center = dist_from_center
        self.source = source
        self.rfz_radius = rfz_radius
        self._deviation = kwargs.get("deviation", 0)

        if kwargs.get("listen_pos") is not None:
            self._listen_pos = kwargs["listen_pos"]

        self._axis, self._wall_pos = self._wall.pos(self.speaker_height)
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
                        self.speaker_height,
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
                        self.speaker_height,
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
            return np.array([p[0] + self._listen_pos, p[1], self.height])
        match self._axis:
            case Axis.X:
                return np.array(
                    [
                        self._wall_pos
                        + self.dist_from_wall
                        + (self.dist_from_center * math.sqrt(3))
                        + self._deviation
                        - 0.38,  # magic number from Rod Gervais
                        p[1],
                        self.height,
                    ],
                    dtype="float32",
                )
            case Axis.Y:
                raise RuntimeError
            case Axis.Z:
                raise RuntimeError

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
                            self.speaker_height,
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            self._wall_pos + self.dist_from_wall,
                            p[1] + self.dist_from_center,
                            self.speaker_height,
                        ],
                        dtype="float32",
                    ),
                    np.array(
                        [
                            self._wall_pos
                            + self.dist_from_wall
                            + (self.dist_from_center * math.sqrt(3)),
                            +self._deviation - 0.38,  # magic number from Rod Gervais
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
    intensity: float


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

        # mat = pra.Material(energy_absorption=0.1, scattering=0.2)
        # _pra_walls = []
        # for w in walls:
        #     # w.simplify()
        #     for tri in w.mesh.triangles:
        #         pw = pra.Wall(
        #             tri.T,
        #             mat.energy_absorption["coeffs"],
        #             mat.scattering["coeffs"],
        #         )
        #         pw.name = w.name
        #         _pra_walls.append(pw)
        # self.pra_room = pra.Room(
        #     _pra_walls,
        #     fs=44100,
        #     max_order=3,
        #     ray_tracing=True,
        #     air_absorption=False,
        # )

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
        l_source = self._lt.l_source()
        r_source = self._lt.r_source()
        listen_pos = self._lt.listening_pos()
        for w in self.walls:
            if w.name == "Window":
                if test_intersection(
                    w.mesh, l_source, dir_from_points(l_source, listen_pos)
                ):
                    raise ObscuresWindow("Left wall obscures window")
        for w in self.walls:
            if w.name == "Window":
                if test_intersection(
                    w.mesh, r_source, dir_from_points(r_source, listen_pos)
                ):
                    raise ObscuresWindow("Right wall obscures window")
        # for w in self.walls:
        #     if w.name == "Door Door":
        #         if test_intersection(
        #             w.mesh, r_source, dir_from_points(r_source, listen_pos)
        #         ):
        #             raise ObscuresWindow("Right wall intersects door")
        self.walls.append(
            build_wall_from_point(
                "left speaker wall",
                self.mesh,
                l_source,
                dir_from_points(l_source, listen_pos),
            )
        )
        self.walls.append(
            build_wall_from_point(
                "right speaker wall",
                self.mesh,
                r_source,
                dir_from_points(r_source, listen_pos),
            )
        )

    def ceiling_diffuser(
        self, height: float, length: float, width: float, position: float
    ) -> None:
        floor = self.get_wall("Floor")
        centroid = floor.mesh.centroid + np.array([0, 0, height])
        l = np.array([length, 0, 0])
        w = np.array([0, width, 0])
        vertices = [
            centroid - l / 2 - w / 2,
            centroid - l / 2 + w / 2,
            centroid + l / 2 - w / 2,
            centroid + l / 2 + w / 2,
        ]
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        self.walls.append(
            Wall("Ceiling Diffuser", trimesh.Trimesh(vertices=vertices, faces=faces))
        )
        pass

    def corner_wall(
        self,
        name: str,
        wall_names: typing.Tuple[str, str],
        x_pos: float = 0.25,
        y_pos: float = 0.25,
        height: float = 0,
        inclination: float = 0,
        **kwargs,
    ) -> Wall:
        # TODO: support using walls to define this
        x_wall, y_wall = wall_names
        xw = self.get_wall(x_wall)
        yw = self.get_wall(y_wall)
        shared_vertices = []
        for i, v in enumerate(xw.mesh.vertices):
            for j, vv in enumerate(yw.mesh.vertices):
                if np.all(np.array(v) == np.array(vv)):
                    shared_vertices.append((i, j, v))
        x_faces: npt.NDArray
        y_faces: npt.NDArray
        shared_vertex_at_zero: npt.NDArray
        for i, j, v in shared_vertices:
            if v[2] == 0:
                shared_vertex_at_zero = v
                x_faces = xw.mesh.faces[xw.mesh.vertex_faces[i]]
                y_faces = yw.mesh.faces[yw.mesh.vertex_faces[j]]
                break
        xdir = npt.NDArray
        ydir = npt.NDArray
        for f in x_faces:
            for i in f:
                v = xw.vertices[i]
                if not np.all(v == shared_vertex_at_zero) and v[2] == 0:
                    xdir = dir_from_points(shared_vertex_at_zero, v)
                    break
        for f in y_faces:
            for i in f:
                v = yw.vertices[i]
                if not np.all(v == shared_vertex_at_zero) and v[2] == 0:
                    ydir = dir_from_points(shared_vertex_at_zero, v)
                    break
        # print(f"Corner {name}")
        # print(f"Shared point: {shared_vertex_at_zero}")
        xpoint = shared_vertex_at_zero + x_pos * xdir
        # print(f"xpoint: {xpoint}")
        ypoint = shared_vertex_at_zero + y_pos * ydir
        # print(f"ypoint: {ypoint}")
        midpoint = xpoint + (ypoint - xpoint) / 2 + np.array([0, 0, height])
        # print(f"midpoint: {midpoint}")
        i_rad = inclination * np.pi / 180
        pitch = np.array(
            [
                [math.cos(i_rad), 0, -math.sin(i_rad)],
                [0, 1, 0],
                [math.sin(i_rad), 0, math.cos(i_rad)],
            ]
        )
        # norm = np.array([(midpoint - xpoint)[1], -(midpoint - xpoint)[0], 0]).dot(pitch)
        line_dir = dir_from_points(xpoint, midpoint)
        norm = np.array([line_dir[1], -line_dir[0], 0]).dot(pitch)
        # print(f"norm: {norm}")
        w = build_wall_from_point(name, self.mesh, midpoint, norm)
        self.walls.append(w)
        return w

    def direct_distances(self) -> tuple[float, float]:
        return (
            float(np.linalg.norm(self._lt.l_source() - self._lt.listening_pos)),
            float(np.linalg.norm(self._lt.r_source() - self._lt.listening_pos)),
        )

    # Stub to provide type awareness
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

    # TODO: terminate reflections for each trace with the nearest point to listen pos, instead of the upcoming next reflection
    # TODO monte carlo simulation:
    # 1. automatically sweep features to search for optimal:
    #       [X] speaker distance from center
    #       [X] speaker distance from front wall
    #       [X] speaker height
    #       [X] listener distance from front wall
    #       [X] listener height
    #       [X] ceiling diffuser height
    #       [X] ceiling diffuser width
    #       [X] ceiling diffuser length
    #       [X] ceiling diffuser position (x axis)
    #       [ ] rear corner positions
    #       [ ] rear corner inclination
    # 2. limitations:
    #       [ ] speaker collision with front wall
    #       [ ] speaker obscures window
    #       [X] limited range for listener height
    #       [X] limited range for listener position on x axis (not too close to front or rear wall)
    #       [X] limited deviation from equilateral listening triangle
    # 3. reward function:
    #       [X] maximize ITD (time until first reflection)
    #       [ ] minimize intensity of first X reflections
    #       [ ] minimize deviation from equilateral listening triangle
    def trace(
        self,
        source: Source,
        orig_source_pos: npt.NDArray,
        listen_pos: npt.NDArray,
        **kwargs,
    ) -> typing.Tuple[typing.List[typing.List[Reflection]], typing.List[Arrival]]:
        order = kwargs.get("order", 10)
        max_time = kwargs.get("max_time", 0.1)
        min_gain = kwargs.get("min_gain", -20)
        num_samples = int(kwargs.get("num_samples", 10))
        vert_disp: float = kwargs.get("vert_disp", 180)
        horiz_disp: float = kwargs.get("horiz_disp", 180)
        # TODO: kwarg source selection
        self._max_time = max_time
        self._min_gain = min_gain
        source_pos = orig_source_pos

        source_normal = dir_from_points(source_pos, listen_pos)
        direct_dist = np.linalg.norm(source_pos - listen_pos)

        shots: typing.List[Shot] = [Shot(source_normal, source.intensity(0, 0))]
        # These are in degrees
        h_steps = int(math.floor(math.sqrt(num_samples)))
        h_step_size = horiz_disp / (h_steps - 1)
        v_steps = num_samples // h_steps
        v_step_size = vert_disp / (v_steps - 1)
        for v in range(v_steps):
            theta_v_deg = -vert_disp / 2 + v_step_size * v
            theta_v = (theta_v_deg) / 180 * math.pi
            pitch = np.array(
                [
                    [math.cos(theta_v), 0, -math.sin(theta_v)],
                    [0, 1, 0],
                    [math.sin(theta_v), 0, math.cos(theta_v)],
                ]
            )
            for h in range(h_steps):
                theta_h_deg = -horiz_disp / 2 + h_step_size * h
                theta_h = (theta_h_deg) / 180 * math.pi
                yaw = np.array(
                    [
                        [math.cos(theta_h), math.sin(theta_h), 0],
                        [-math.sin(theta_h), math.cos(theta_h), 0],
                        [0, 0, 1],
                    ]
                )

                new_dir = yaw.dot(pitch).dot(source_normal)
                new_dir = new_dir / np.linalg.norm(new_dir)
                shots.append(
                    Shot(
                        new_dir / np.linalg.norm(new_dir),
                        source.intensity(theta_v_deg, theta_h_deg),
                    )
                )

        hits: typing.List[typing.List[Reflection]] = []
        arrivals: typing.List[Arrival] = []
        mesh = self.mesh
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        for j, shot in enumerate(shots):
            source_pos = orig_source_pos
            temp_hits: typing.List[Reflection] = []
            total_dist: float = -direct_dist
            reflected_to_rfz = False
            intensity = from_db(shot.intensity)
            wall: typing.Union[Wall, None] = None

            dir = shot.dir
            for i in range(order):
                norm: npt.NDArray = np.empty(3)
                new_dir: npt.NDArray = np.empty(3)
                new_source: npt.NDArray = np.empty(3)

                idx_tri, idx_ray, loc = intersector.intersects_id(
                    [source_pos],
                    [dir],
                    return_locations=True,
                    multiple_hits=True,
                )
                if len(loc) == 0:
                    raise RuntimeError
                found = False

                def min_norm(e):
                    return np.linalg.norm(source_pos - e[0])

                for this_loc, tri_idx in sorted(
                    zip(loc, idx_tri), key=min_norm, reverse=False
                ):
                    # Check here for multiple hits
                    if np.linalg.norm(source_pos - this_loc) > 0.001:
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
                    Reflection(new_source, wall, source_pos, intensity, total_dist)
                )

                # Check whether this reflection passes within the RFZ
                dist_from_crit = dist(new_source, source_pos, listen_pos)
                total_dist = total_dist + float(np.linalg.norm(new_source - source_pos))
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

                dir = dir - norm * 2 * dir.dot(norm)
                source_pos = new_source
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
        plt.draw()

        # sec = self.mesh.section((0, 0, 1), (0, 0, self._lt.speaker_height))
        sec = self.mesh.section((0, 0, 1), (0, 0, 0))
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

    def write(self, filename: str):
        scene = trimesh.Scene(self.mesh)
        scene.export(filename, "stl")


def get_arrivals(solution) -> tuple[Room, typing.List[Arrival]]:
    params = training_parameters(*solution)

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
        height=params.height,
        speaker_height=params.speaker_height,
        dist_from_wall=params.dist_from_wall,
        dist_from_center=params.dist_from_center,
        deviation=params.deviation_from_equilateral,
        source=Source(
            vert_disp={0: 0, 25: -5, 60: -6, 80: -12, 90: -100},
            horiz_disp={0: 0, 30: -3, 50: -6, 60: -9, 90: -100},
        ),
        rfz_radius=0.3,
    )
    listen_pos = room._lt.listening_pos()
    if listen_pos[0] <= params.min_listen_pos:
        raise ListeningPositionError("Too close to front wall")
    if listen_pos[0] >= params.max_listen_pos:
        raise ListeningPositionError("Too close to back wall")
    room.ceiling_diffuser(
        params.ceiling_diffuser_height,
        params.ceiling_diffuser_length,
        params.ceiling_diffuser_width,
        params.ceiling_diffuser_position,
    )
    (_, l_arrivals) = room.trace(
        room._lt.source,
        room._lt.l_source(),
        room._lt.listening_pos(),
        num_samples=params.num_samples,
        max_time=params.max_time,
        min_gain=params.min_gain,
        order=10,
    )
    (_, r_arrivals) = room.trace(
        room._lt.source,
        room._lt.r_source(),
        room._lt.listening_pos(),
        num_samples=params.num_samples,
        max_time=params.max_time,
        min_gain=params.min_gain,
        order=10,
    )
    arrivals = l_arrivals + r_arrivals
    return room, arrivals


def fitness_func(ga_instance, solution, solution_idx) -> float:
    params = training_parameters(*solution)
    try:
        _, arrivals = get_arrivals(solution)
    except SelahStudioException as ex:
        print(f"Invalid solution: {ex}")
        return 0
    arrivals.sort(key=lambda a: a.total_dist)
    if len(arrivals) == 0:
        print(f"HOLY GRAIL: {params.max_time * 1000}")
        return params.max_time * 1000
    ITD = float(arrivals[0].total_dist / SPEED_OF_SOUND * 1000)
    print(f"ITD: {ITD:.1f}")
    return ITD


@dataclass
class training_parameters:
    height: typing.Union[float, dict[str, float]] = 1.4
    speaker_height: typing.Union[float, dict[str, float]] = 1.4
    dist_from_wall: typing.Union[float, dict[str, float]] = 0.3
    dist_from_center: typing.Union[float, dict[str, float]] = 0.9
    deviation_from_equilateral: typing.Union[float, dict[str, float]] = 0.0
    max_listen_pos: typing.Union[float, dict[str, float]] = 2.4
    min_listen_pos: typing.Union[float, dict[str, float]] = 1.3
    ceiling_diffuser_height: typing.Union[float, dict[str, float]] = 2.3
    ceiling_diffuser_length: typing.Union[float, dict[str, float]] = 1.0
    ceiling_diffuser_width: typing.Union[float, dict[str, float]] = 1.0
    ceiling_diffuser_position: typing.Union[float, dict[str, float]] = 1.5
    num_samples: int = 20
    max_time: float = 80 / 1000
    min_gain: float = -15

    def aslist(self):
        retlist = []
        for name, val in self.__dict__.items():
            if isinstance(val, dict):
                retlist.append(val)
            else:
                retlist.append([val])
        return retlist


if __name__ == "__main__":

    gene_space = training_parameters(
        speaker_height={"low": 1.4, "high": 1.9},
        dist_from_center={"low": 0.95, "high": 1.3},
        dist_from_wall={"low": 0.2, "high": 0.5},
        deviation_from_equilateral={"low": -0.4, "high": 0.4},
        ceiling_diffuser_height={"low": 2.5, "high": 2.75},
        ceiling_diffuser_width={"low": 1.0, "high": 2.75},
        ceiling_diffuser_length={"low": 1.0, "high": 2.75},
        ceiling_diffuser_position={"low": 0.0, "high": 2.5},
        max_listen_pos=2.2,  # QRD diffuser up to 1000Hz design freq
        min_listen_pos=1.3,
        num_samples=3_000,
        max_time=60 / 1000,
        min_gain=-15,
    )
    ga_instance = pygad.GA(
        num_generations=8,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=24,
        num_genes=len(gene_space.aslist()),
        gene_space=gene_space.aslist(),
        mutation_probability=0.4,
        parent_selection_type="tournament",
        K_tournament=4,
        crossover_type="two_points",
        crossover_probability=0.7,
        keep_elitism=8,
        parallel_processing=["process", 24],
    )
    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    pprint.pprint(f"Parameters of the best solution : {training_parameters(*solution)}")
    pprint.pprint(f"Fitness value of the best solution = {solution_fitness}")

    room, arrivals = get_arrivals(solution)
    plt.ion()
    fig = plt.figure()
    room.plot_arrivals_interactive(fig, arrivals, False)
    plt.show(block=True)
    room.mesh.show()
