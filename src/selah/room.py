import math
import typing

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt
import trimesh
from trimesh.ray import ray_triangle

from . import geometry
from .exceptions import SelahException
from .material import MaterialManager
from .loudspeaker import Loudspeaker, LoudspeakerSpec
from .sound import SPEED_OF_SOUND, db, from_db
from .source import Source, Shot, Reflection, ReflectionException, ShotException
from .wall import Axis, Wall, build_wall_from_point


class InvalidMeshException(SelahException):
    """Indicates an invalid mesh"""


class ObscuresWindow(SelahException):
    """Indicates an attempt to create a wall that obscures the window"""


class CollisionException(SelahException):
    """Indicates the speaker would collide with a wall"""

    def __init__(self, items: list[typing.Tuple[trimesh.Trimesh, npt.NDArray]]):
        self.scene = trimesh.Scene()
        for mesh, location in items:
            self.scene.add_geometry(
                mesh, transform=trimesh.transformations.translation_matrix(location)
            )


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
        source: LoudspeakerSpec,
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
        source: LoudspeakerSpec,
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
        listen_pos = self._lt.listening_pos()
        l_source = Loudspeaker(
            source,
            self._lt.l_source(),
            geometry.dir_from_points(self._lt.l_source(), listen_pos),
        )
        r_source = Loudspeaker(
            source,
            self._lt.r_source(),
            geometry.dir_from_points(self._lt.r_source(), listen_pos),
        )

        if l_source.test_intersection(self.mesh):
            raise CollisionException(
                [(self.mesh, np.array([0, 0, 0])), (l_source.mesh, l_source.position)]
            )
        if r_source.test_intersection(self.mesh):
            raise CollisionException(
                [(self.mesh, np.array([0, 0, 0])), (r_source.mesh, r_source.position)]
            )

        for w in self.walls:
            if w.name == "Window":
                if geometry.test_intersection(
                    w.mesh, l_source.position, l_source.normal
                ):
                    raise ObscuresWindow("Left wall obscures window")
        for w in self.walls:
            if w.name == "Window":
                if geometry.test_intersection(
                    w.mesh, r_source.position, r_source.normal
                ):
                    raise ObscuresWindow("Right wall obscures window")
        self.walls.append(
            build_wall_from_point(
                "left speaker wall",
                self.mesh,
                l_source.position,
                l_source.normal,
                self._mm.get_wall("left speaker wall"),
            )
        )
        self.walls.append(
            build_wall_from_point(
                "right speaker wall",
                self.mesh,
                r_source.position,
                r_source.normal,
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

    # def corner_wall(
    #     self,
    #     name: str,
    #     wall_names: typing.Tuple[str, str],
    #     x_pos: float = 0.25,
    #     y_pos: float = 0.25,
    #     height: float = 0,
    #     inclination: float = 0,
    #     **kwargs,
    # ) -> Wall:
    #     """Adds a wall straddling a corner of the room at the specified location and angle."""
    #     # TODO: support using walls to define this
    #     x_wall, y_wall = wall_names
    #     xw = self.get_wall(x_wall)
    #     yw = self.get_wall(y_wall)
    #     shared_vertices = []
    #     for i, v in enumerate(xw.mesh.vertices):
    #         for j, vv in enumerate(yw.mesh.vertices):
    #             if np.all(np.array(v) == np.array(vv)):
    #                 shared_vertices.append((i, j, v))
    #     x_faces: npt.NDArray
    #     y_faces: npt.NDArray
    #     shared_vertex_at_zero: npt.NDArray
    #     for i, j, v in shared_vertices:
    #         if v[2] == 0:
    #             shared_vertex_at_zero = v
    #             x_faces = xw.mesh.faces[xw.mesh.vertex_faces[i]]
    #             y_faces = yw.mesh.faces[yw.mesh.vertex_faces[j]]
    #             break
    #     xdir = npt.NDArray
    #     ydir = npt.NDArray
    #     for f in x_faces:
    #         for i in f:
    #             v = xw.vertices[i]
    #             if not np.all(v == shared_vertex_at_zero) and v[2] == 0:
    #                 xdir = geometry.dir_from_points(shared_vertex_at_zero, v)
    #                 break
    #     for f in y_faces:
    #         for i in f:
    #             v = yw.vertices[i]
    #             if not np.all(v == shared_vertex_at_zero) and v[2] == 0:
    #                 ydir = geometry.dir_from_points(shared_vertex_at_zero, v)
    #                 break
    #     xpoint = shared_vertex_at_zero + x_pos * xdir
    #     ypoint = shared_vertex_at_zero + y_pos * ydir
    #     midpoint = xpoint + (ypoint - xpoint) / 2 + np.array([0, 0, height])
    #     i_rad = inclination * np.pi / 180
    #     pitch = np.array(
    #         [
    #             [math.cos(i_rad), 0, -math.sin(i_rad)],
    #             [0, 1, 0],
    #             [math.sin(i_rad), 0, math.cos(i_rad)],
    #         ]
    #     )
    #     line_dir = geometry.dir_from_points(xpoint, midpoint)
    #     norm = np.array([line_dir[1], -line_dir[0], 0]).dot(pitch)
    #     w = build_wall_from_point(
    #         name, self.mesh, midpoint, norm, self._mm.get_wall("back_corners")
    #     )
    #     self.walls.append(w)
    #     return w

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Returns a mesh representing the entirety of the shape of this room."""
        m = trimesh.util.concatenate([x.mesh for x in self.walls])
        if not isinstance(m, trimesh.Trimesh):
            raise SelahException("Failed to create mesh")
        m.fix_normals(False)
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

    def trace_shot(
        self,
        mesh: trimesh.Trimesh,
        shot: Shot,
        orig_source_pos: npt.NDArray,
        listen_pos: npt.NDArray,
        rfz_radius: float,
        order: int = 10,
        max_time: float = 60,
        min_gain: float = -20,
        ignore_walls: typing.List[str] = [],
    ) -> typing.Tuple[Source, bool]:
        source_pos = orig_source_pos
        last_source: Source = shot
        direct_dist = np.linalg.norm(source_pos - listen_pos)
        total_dist: float = -float(direct_dist)
        intensity = from_db(shot.gain)
        wall: Wall

        # First, check whether this ray intersects the rfz. If so, return.
        # If not, check subsequent reflections of this ray.

        intersector = ray_triangle.RayMeshIntersector(mesh)
        dir = shot.dir
        for i in range(order):
            norm: npt.NDArray = np.empty(3)
            new_pos: npt.NDArray = np.empty(3)

            idx_tri, _, loc = intersector.intersects_id(  # type:ignore
                # This method has multiple return signatures. Since return_locations=True, we know we are accepting the right signature here.
                [source_pos],
                [dir],
                return_locations=True,
                multiple_hits=True,
            )

            def min_norm(e):
                return np.linalg.norm(source_pos - e[0])

            match len(loc):
                case 0:
                    if isinstance(last_source, Reflection):
                        print("Never terminates")
                        # raise ReflectionException(
                        #     last_source, "Reflected ray never terminates"
                        # )
                    if isinstance(last_source, Shot):
                        print("Never terminates")
                        # raise ShotException(
                        #     last_source, "Reflected ray never terminates"
                        # )
                case 1:
                    if np.linalg.norm(source_pos - loc[0]) > 0:
                        new_pos = loc[0]
                        if mesh.face_normals is None:
                            raise SelahException(
                                "code bug: face_normals should never return None"
                            )
                        norm = mesh.face_normals[idx_tri[0]]
                        dir = dir - norm * 2 * dir.dot(norm)
                        wall = self.faces_to_wall(idx_tri[0])
                        intensity = intensity * (1 - wall.material.absorption())
                        last_source = Reflection(
                            new_pos, intensity, total_dist, last_source, wall
                        )
                case _:
                    found = False
                    for this_loc, tri_idx in sorted(
                        zip(loc, idx_tri), key=min_norm, reverse=False
                    ):
                        if np.linalg.norm(source_pos - this_loc) < 1e-6:
                            continue
                        new_pos = this_loc
                        if mesh.face_normals is None:
                            raise SelahException(
                                "code bug: face_normals should never return None"
                            )
                        norm = mesh.face_normals[tri_idx]
                        dir = dir - norm * 2 * dir.dot(norm)
                        wall = self.faces_to_wall(tri_idx)
                        intensity = intensity * (1 - wall.material.absorption())
                        last_source = Reflection(
                            new_pos, intensity, total_dist, last_source, wall
                        )
                        found = True
                        break
                    if not found:
                        if isinstance(last_source, Reflection):
                            raise ReflectionException(
                                last_source, "Malformed reflection"
                            )
                        raise SelahException("Malformed reflection with wrong type")

            # Check whether this reflection passes within the RFZ
            dist_from_crit = geometry.lineseg_dist(new_pos, source_pos, listen_pos)
            total_dist = total_dist + float(np.linalg.norm(new_pos - source_pos))

            source_pos = new_pos
            # Only check out to some number of ms
            if total_dist / SPEED_OF_SOUND > max_time:
                break
            # Only check out to some minimum gain
            if db(intensity) < min_gain:
                break
            if isinstance(last_source, Reflection):
                prev_source = last_source.parent
                if isinstance(prev_source, Reflection):
                    if prev_source.wall.name in ignore_walls:
                        continue
            if dist_from_crit < rfz_radius and i > 0:
                # We only care about rays that reflect to the RFZ
                return last_source, True

        return last_source, False

    def trace_arrivals(
        self,
        source: Loudspeaker,
        source_pos: npt.NDArray,
        listen_pos: npt.NDArray,
        **kwargs,
    ) -> typing.List[Source]:
        """
        Uses ray tracing to determine time of arrival and intensity of each reflection
        that arrives at the listening position.
        """

        order = kwargs.get("order", 10)
        max_time = kwargs.get("max_time", 0.1)
        min_gain = kwargs.get("min_gain", -20)
        num_samples = int(kwargs.get("num_samples", 10))
        ignore_walls = kwargs.get("ignore_walls", [])
        self._max_time = max_time
        self._min_gain = min_gain

        shots = source.get_shots(listen_pos, num_samples)
        mesh = self.mesh

        arrivals: typing.List[Source] = []
        for shot in shots:
            arrival, intersects_rfz = self.trace_shot(
                mesh,
                shot,
                source_pos,
                listen_pos,
                self._lt.rfz_radius,
                order,
                max_time,
                min_gain,
                ignore_walls,
            )
            if intersects_rfz:
                arrivals.append(arrival)

        arrivals.sort(key=lambda a: a.total_dist)
        return arrivals

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
        circle = patches.Circle(
            (self._lt.listening_pos()[0], self._lt.listening_pos()[1]),
            self._lt.rfz_radius,
            fill=False,
            color="dimgrey",
        )
        plt.gca().add_patch(circle)
        plt.draw()

        # sec = self.mesh.section((0, 0, 1), (0, 0, self._lt.speaker_height))
        sec = self.mesh.section((0, 0, 1), (0, 0, 0))
        if not isinstance(sec, trimesh.path.Path3D):
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
        circle = patches.Circle(
            (self._lt.listening_pos()[0], self._lt.listening_pos()[2]),
            self._lt.rfz_radius,
            fill=False,
            color="dimgrey",
        )
        plt.gca().add_patch(circle)
        plt.draw()

        sec = self.mesh.section((0, 1, 0), (0, 3, 0))
        if not isinstance(sec, trimesh.path.Path3D):
            raise RuntimeError
        outline = sec.to_planar()[0]
        outline.apply_transform(((0, -1, 0), (-1, 0, 0), (0, 0, 1)))
        # Rotate outline by 90deg
        outline.apply_translation((-outline.bounds[0][0], -outline.bounds[0][1]))
        outline.plot_entities()

    def plot_arrivals(
        self,
        fig,
        arrivals: typing.List[Source],
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
                bottom=db(a.gain),
                height=self._min_gain,
                color=a.color(color),
                picker=True,
            )
            # TODO: this needs to walk back the linked list
            h = a
            while True:
                if isinstance(h, Shot):
                    break
                if isinstance(h, Reflection):
                    if manually_advance:
                        plt.waitforbuttonpress()
                    ax1.scatter(h.pos[0], h.pos[1])
                    ax1.plot(
                        [h.pos[0], h.parent.pos[0]],
                        [h.pos[1], h.parent.pos[1]],
                        marker="o",
                        color=h.color(color),
                        linewidth=4 * h.gain,
                    )
                    ax2.scatter(h.pos[0], h.pos[2])
                    ax2.plot(
                        [h.pos[0], h.parent.pos[0]],
                        [h.pos[2], h.parent.pos[2]],
                        marker="o",
                        color=h.color(color),
                        linewidth=4 * h.gain,
                    )
                    h = h.parent
            plt.draw()

    def plot_arrivals_interactive(
        self,
        fig,
        arrivals: typing.List[Source],
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
