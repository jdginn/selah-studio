import math
import typing
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt
import trimesh

from . import geometry
from .exceptions import SelahException
from .material import MaterialManager
from .sound import SPEED_OF_SOUND, db, from_db
from .source import Source
from .wall import Axis, Wall, build_wall_from_point


class InvalidMeshException(SelahException):
    """Indicates an invalid mesh"""


class ObscuresWindow(SelahException):
    """Indicates an attempt to create a wall that obscures the window"""


class ListeningPositionError(SelahException):
    """Indicates the listening position has been placed outside the valid area"""


class ListeningTriangle:
    """
    Represents a listening triangle composed of two sound sources and a listener.

    Useful for computing positions and ensuring that symmetry is respected.
    """

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
        """Returns the position of the left stereo source"""
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
        """Returns the position of the right stereo source"""
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

    # Value from Rod Gervais' book Home Recording Studio: Build It Like The Pros
    LISTENER_DIST_INTO_TRIANGLE = 0.38

    def listening_pos(self) -> npt.NDArray:
        """Returns the position of the listener's head"""
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


@dataclass
class Reflection:
    """
    Represents a discrete sound reflection off of a surface.
    """

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
    """
    Represents the origin of a ray of sound, including its direction, intensity,
    and any other initial information required to predict its behavior.
    """

    dir: npt.NDArray
    intensity: float


class Arrival:
    """
    Represents a series of reflections that arrives at a given position. Allows tracing the full
    path of reflections that were required to reach the position.
    """

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
    """
    Models a room for acoustic purposes.

    Rooms are constructed with walls and their shape is defined as a geometrical mesh.
    """

    def __init__(
        self, walls: typing.List[Wall], mm: MaterialManager = MaterialManager()
    ):
        self.walls = walls
        self._mm = mm
        for wall in self.walls:
            wall.material = self._mm.get_wall(wall.name)

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
        """
        Installs sources and listening position in accordance with a listening triangle.

        Also adds walls to the room appropriate for flush-mounting speakers.
        """
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
                if geometry.test_intersection(
                    w.mesh, l_source, geometry.dir_from_points(l_source, listen_pos)
                ):
                    raise ObscuresWindow("Left wall obscures window")
        for w in self.walls:
            if w.name == "Window":
                if geometry.test_intersection(
                    w.mesh, r_source, geometry.dir_from_points(r_source, listen_pos)
                ):
                    raise ObscuresWindow("Right wall obscures window")
        self.walls.append(
            build_wall_from_point(
                "left speaker wall",
                self.mesh,
                l_source,
                geometry.dir_from_points(l_source, listen_pos),
                self._mm.get_wall("left speaker wall"),
            )
        )
        self.walls.append(
            build_wall_from_point(
                "right speaker wall",
                self.mesh,
                r_source,
                geometry.dir_from_points(r_source, listen_pos),
                self._mm.get_wall("right speaker wall"),
            )
        )

    def ceiling_absorber(
        self, height: float, length: float, width: float, position: float
    ) -> None:
        """Adds an acoustic absorber to the room suspended from the ceiling"""
        floor = self.get_wall("Floor")
        centroid = floor.mesh.centroid + np.array([0, 0, height])
        larr = np.array([length, 0, 0])
        warr = np.array([0, width, 0])
        vertices = [
            centroid - larr / 2 - warr / 2,
            centroid - larr / 2 + warr / 2,
            centroid + larr / 2 - warr / 2,
            centroid + larr / 2 + warr / 2,
        ]
        faces = np.array([[0, 1, 2], [1, 2, 3]])
        self.walls.append(
            Wall(
                "Ceiling Diffuser",
                trimesh.Trimesh(vertices=vertices, faces=faces),
                self._mm.get_wall("Ceiling Diffuser"),
            )
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
        """Adds a wall straddling a corner of the room at the specified location and angle."""
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
                    xdir = geometry.dir_from_points(shared_vertex_at_zero, v)
                    break
        for f in y_faces:
            for i in f:
                v = yw.vertices[i]
                if not np.all(v == shared_vertex_at_zero) and v[2] == 0:
                    ydir = geometry.dir_from_points(shared_vertex_at_zero, v)
                    break
        xpoint = shared_vertex_at_zero + x_pos * xdir
        ypoint = shared_vertex_at_zero + y_pos * ydir
        midpoint = xpoint + (ypoint - xpoint) / 2 + np.array([0, 0, height])
        i_rad = inclination * np.pi / 180
        pitch = np.array(
            [
                [math.cos(i_rad), 0, -math.sin(i_rad)],
                [0, 1, 0],
                [math.sin(i_rad), 0, math.cos(i_rad)],
            ]
        )
        # norm = np.array([(midpoint - xpoint)[1], -(midpoint - xpoint)[0], 0]).dot(pitch)
        line_dir = geometry.dir_from_points(xpoint, midpoint)
        norm = np.array([line_dir[1], -line_dir[0], 0]).dot(pitch)
        w = build_wall_from_point(name, self.mesh, midpoint, norm)
        self.walls.append(w)
        return w

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Returns a mesh representing the entirety of the shape of this room."""
        m = trimesh.util.concatenate([x.mesh for x in self.walls])
        if not isinstance(m, trimesh.Trimesh):
            raise RuntimeError
        m.fix_normals(True)
        return m

    def faces_to_wall(self, idx: int) -> Wall:
        """Maps a given face to the wall to which it belongs."""
        if not hasattr(self, "_faces_to_wall"):
            self._faces_to_wall: typing.List[Wall] = []
            for w in self.walls:
                for _ in w.mesh.faces:
                    self._faces_to_wall.append(w)
        return self._faces_to_wall[idx]

    def get_wall(self, name: str | int) -> Wall:
        """Returns a wall from a room by name."""
        for w in self.walls:
            if w.name == name:
                return w
        raise SelahException(f"Could not find requested wall {name}")

    def trace(
        self,
        source: Source,
        orig_source_pos: npt.NDArray,
        listen_pos: npt.NDArray,
        **kwargs,
    ) -> typing.List[Arrival]:
        """
        Uses ray tracing to determine time of arrival and intensity of each reflection
        that arrives at the listening position.
        """
        order = kwargs.get("order", 10)
        max_time = kwargs.get("max_time", 0.1)
        min_gain = kwargs.get("min_gain", -20)
        num_samples = int(kwargs.get("num_samples", 10))
        vert_disp: float = kwargs.get("vert_disp", 180)
        horiz_disp: float = kwargs.get("horiz_disp", 180)
        self._max_time = max_time
        self._min_gain = min_gain
        source_pos = orig_source_pos

        source_normal = geometry.dir_from_points(source_pos, listen_pos)
        direct_dist = np.linalg.norm(source_pos - listen_pos)

        shots: typing.List[Shot] = [Shot(source_normal, source.gain(0, 0))]
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
                        source.gain(theta_v_deg, theta_h_deg),
                    )
                )

        hits: typing.List[typing.List[Reflection]] = []
        arrivals: typing.List[Arrival] = []
        mesh = self.mesh
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        for _, shot in enumerate(shots):
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

                idx_tri, _, loc = intersector.intersects_id(
                    [source_pos],
                    [dir],
                    return_locations=True,
                    multiple_hits=True,
                )
                if len(loc) == 0:
                    raise SelahException("Reflected ray never terminates")
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
                        intensity = intensity * (1 - wall.material.absorption())
                        found = True
                        break
                if not found:
                    raise SelahException("Ray only reflects back to self")

                temp_hits.append(
                    Reflection(new_source, wall, source_pos, intensity, total_dist)
                )

                # Check whether this reflection passes within the RFZ
                dist_from_crit = geometry.lineseg_dist(
                    new_source, source_pos, listen_pos
                )
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
        """
        Plots a 2-dimensional representation of the room as viewed from above.
        """
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
        """
        Plots a 2-dimensional representation of the room as viewed from the side.
        """
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
        """
        Plots all arrivals to the listening position along with the paths they took to get there.
        """

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
        """
        Interactive view for arrivals allowing each reflection to be viewed individually
        """
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
