import math
import typing
import pdb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import numpy.typing as npt
import trimesh
from trimesh.ray import ray_triangle
import trimesh.visual as tv
import trimesh.path.entities as entities
from trimesh.path import Path2D

from . import geometry
from .exceptions import SelahException
from .material import MaterialManager, Material
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

    def __init__(self, items: list[trimesh.Trimesh]):
        self.scene = trimesh.Scene()
        for mesh in items:
            self.scene.add_geometry(mesh)


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
        source_spec: LoudspeakerSpec,
        rfz_radius: float,
        **kwargs,
    ) -> None:
        self._wall = wall
        self.height = height
        self.speaker_height = kwargs.get("speaker_height", height)
        self.dist_from_wall = dist_from_wall
        self.dist_from_center = dist_from_center
        self.source_spec = source_spec
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

    @property
    def y_center(self) -> float:
        return self._wall.center_pos()[1]

    def l_source(self) -> Loudspeaker:
        """Returns the position of the left stereo source"""
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                speaker_pos = np.array(
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
        return Loudspeaker(
            self.source_spec,
            speaker_pos,
            geometry.dir_from_points(speaker_pos, self.listening_pos()),
        )

    def r_source(self) -> Loudspeaker:
        """Returns the position of the right stereo source"""
        p = self._wall.center_pos()
        match self._axis:
            case Axis.X:
                speaker_pos = np.array(
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
        return Loudspeaker(
            self.source_spec,
            speaker_pos,
            geometry.dir_from_points(speaker_pos, self.listening_pos()),
        )

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

    @property
    def listening_dist(self) -> float:
        return float(np.linalg.norm(self.l_source().position - self.listening_pos()))


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
        l_source = self._lt.l_source()
        r_source = self._lt.r_source()

        if l_source.test_intersection(self.mesh):
            raise CollisionException([self.mesh, l_source.mesh])
        if r_source.test_intersection(self.mesh):
            raise CollisionException([self.mesh, r_source.mesh])

        windows = ["Window A", "Window B"]
        combined_window = trimesh.util.concatenate(
            [x.mesh for x in self.walls if x.name in windows]
        )
        if isinstance(combined_window, list):
            raise SelahException("Could not find window")
        combined_window = typing.cast(trimesh.Trimesh, combined_window)
        window2d = typing.cast(
            Path2D, combined_window.projected([-1, 0, 0], origin=[0, 0, 0])
        )
        source_x = max(l_source.position[0], r_source.position[0])
        window_box = typing.cast(trimesh.Trimesh, window2d.extrude(-2 * source_x))
        window_box.visual = tv.ColorVisuals(window_box, tv.random_color())
        window_box.apply_transform(
            trimesh.transformations.rotation_matrix(
                -90 / 180 * np.pi, np.array([0, 1, 0]), np.array([0, 0, 0])
            )
        )
        window_box.apply_translation([-source_x, 0, 0])
        l_wall = build_wall_from_point(
            "left speaker wall",
            self.mesh,
            l_source.position,
            l_source.normal,
            self._mm.get_wall("left speaker wall"),
        )
        v1, v2 = (l_wall.vertices[10], l_wall.vertices[20])
        print(
            f"l_wall defined by: [{l_source.position[0]}, {l_source.position[1]}, {l_source.position[2]}] [{v1[0]}, {v1[1]}, {v1[2]}] [{v2[0]}, {v2[1]}, {v2[2]}]"
        )
        r_wall = build_wall_from_point(
            "right speaker wall",
            self.mesh,
            r_source.position,
            r_source.normal,
            self._mm.get_wall("right speaker wall"),
        )
        # Hack
        mesh = r_wall.mesh.slice_plane(
            plane_origin=[0, self.get_wall("Hall A").vertices[0][1], 0],
            plane_normal=[0, -1, 0],
        )
        if mesh is None:
            raise SelahException
        # mesh = mesh.slice_plane(
        #     plane_origin=[self.get_wall("Entry Front").vertices[0][0], 0, 0],
        #     plane_normal=[-1, 0, 0],
        # )
        # if mesh is None:
        #     raise SelahException
        r_wall.mesh = mesh
        v1, v2 = (r_wall.vertices[10], r_wall.vertices[20])
        print(
            f"r_wall defined by: [{r_source.position[0]}, {r_source.position[1]}, {r_source.position[2]}] [{v1[0]}, {v1[1]}, {v1[2]}] [{v2[0]}, {v2[1]}, {v2[2]}]"
        )
        if l_wall.mesh.intersection(window_box):
            l_wall.mesh = l_wall.mesh.difference(window_box)
        if r_wall.mesh.intersection(window_box):
            r_wall.mesh = r_wall.mesh.difference(window_box)
        self.walls.append(l_wall)
        self.walls.append(r_wall)

    def ceiling_absorber(
        self, height: float, length: float, width: float, position: float
    ) -> None:
        """Adds an acoustic absorber to the room suspended from the ceiling"""
        center = self._lt.y_center
        vertices = [
            [position, center + width / 2, height],
            [position, center - width / 2, height],
            [position + length, center + width / 2, height],
            [position + length, center - width / 2, height],
        ]
        faces = np.array([[2, 1, 0], [3, 2, 1]])
        self.walls.append(
            Wall(
                "Ceiling Diffuser",
                trimesh.Trimesh(vertices=vertices, faces=faces),
                self._mm.get_wall("Ceiling Diffuser"),
            )
        )
        pass

    @property
    def mesh(self) -> trimesh.Trimesh:
        """Returns a mesh representing the entirety of the shape of this room."""
        m = trimesh.util.concatenate([x.mesh for x in self.walls])
        if not isinstance(m, trimesh.Trimesh):
            raise SelahException("Failed to create mesh")
        # mm = m.process(True, True, True)
        # mm.fix_normals(False)
        return m

    def mesh_excluding(self, exc: typing.List[str]):
        m = trimesh.util.concatenate([x.mesh for x in self.walls if x not in exc])
        if not isinstance(m, trimesh.Trimesh):
            raise SelahException("Failed to create mesh")
        mm = m.process(True, True, True)
        mm.fix_normals(False)
        return mm

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
        final_source: Source = shot
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
                # This ray never terminates
                case 0:
                    if isinstance(final_source, Reflection):
                        raise ReflectionException(
                            final_source, "Reflected ray never terminates"
                        )
                    if isinstance(final_source, Shot):
                        raise ShotException(
                            final_source, "Reflected ray never terminates"
                        )
                # This ray has exactly one intersection
                case 1:
                    print("Micsoda?")
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
                        final_source = Reflection(
                            new_pos, intensity, final_source, wall
                        )
                    else:
                        SelahException("WTF?")
                # This ray has multipole intersections (even though only one of them is real)
                #
                # Usually, one of those points is a duplicate of the source, so we can always ignore that one
                #
                # Removing the duplicate intersection, the nearest intersection is the real one (the near one is on a wall that obscures any other intersections)
                case _:
                    found = False
                    for this_loc, tri_idx in sorted(
                        zip(loc, idx_tri), key=min_norm, reverse=False
                    ):
                        if np.linalg.norm(source_pos - this_loc) < 1e-6:
                            # This intersection is a dupe of the source
                            continue
                        # We proceed directly to the nearest intersection that is not a dupe of the source
                        new_pos = this_loc
                        if mesh.face_normals is None:
                            raise SelahException(
                                "code bug: face_normals should never return None"
                            )
                        norm = mesh.face_normals[tri_idx]
                        dir = dir - norm * 2 * dir.dot(norm)
                        wall = self.faces_to_wall(tri_idx)
                        intensity = intensity * (1 - wall.material.absorption())
                        final_source = Reflection(
                            new_pos, intensity, final_source, wall
                        )
                        found = True
                        break
                    if not found:
                        if isinstance(final_source, Reflection):
                            raise ReflectionException(
                                final_source, "Malformed reflection"
                            )
                        raise SelahException("Malformed reflection with wrong type")

            # Check whether this reflection passes within the RFZ
            dist_from_crit = geometry.lineseg_dist(new_pos, source_pos, listen_pos)

            source_pos = new_pos
            # Only check out to some minimum gain
            if db(intensity) < min_gain:
                break
            if isinstance(final_source, Reflection):
                # Only check out to some number of ms
                if final_source.total_dist / SPEED_OF_SOUND > max_time:
                    break
                prev_source = final_source.parent
                if isinstance(prev_source, Reflection):
                    if prev_source.wall.name in ignore_walls:
                        continue
            if dist_from_crit < rfz_radius and i > 0:
                # We only care about rays that reflect to the RFZ
                return final_source, True

        return final_source, False

    def trace_arrivals(
        self,
        source: Loudspeaker,
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
                source.position,
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
            self._lt.l_source().position[0],
            self._lt.l_source().position[1],
            marker="x",
            linewidth=8,
        )
        plt.scatter(
            self._lt.r_source().position[0],
            self._lt.r_source().position[1],
            marker="x",
            linewidth=8,
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
            self._lt.l_source().position[0],
            self._lt.l_source().position[2],
            marker="x",
            linewidth=8,
        )
        plt.scatter(
            self._lt.r_source().position[0],
            self._lt.r_source().position[2],
            marker="x",
            linewidth=8,
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
                (a.total_dist - self._lt.listening_dist) / SPEED_OF_SOUND * 1000,
                bottom=db(a.gain),
                height=self._min_gain,
                color=a.color(color),
                picker=True,
            )
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

    def show(self):
        scene = trimesh.Scene()
        exc = ["Floor"]
        scene.add_geometry([x.mesh for x in self.walls if x.name not in exc])
        # for w in self.walls:
        #     if w.name in exc:
        #         continue
        #     if w.name == "Window A":
        #         w.mesh.visual = trimesh.visual.ColorVisuals(
        #             tv.color.to_rgba(())
        #         )
        #     # w.mesh.visual = tv.ColorVisuals(w.mesh, tv.random_color())
        #     scene.add_geometry(w.mesh)
        scene.add_geometry(self._lt.l_source().mesh)
        scene.add_geometry(self._lt.r_source().mesh)
        lpos = trimesh.primitives.Sphere(radius=0.1, center=self._lt.listening_pos())
        lpos.visual.vertex_colors = tv.random_color()  # pyright: ignore
        scene.add_geometry(
            trimesh.load_path(
                [self._lt.l_source().position, self._lt.listening_pos()],
            )
        )
        scene.add_geometry(
            trimesh.load_path(
                [self._lt.r_source().position, self._lt.listening_pos()],
            )
        )
        # scene.add_geometry(
        #     trimesh.load_path(
        #         [
        #             self._lt.r_source().position,
        #             self._lt.r_source().position + self._lt.r_source().normal * 10,
        #         ],
        #     )
        # )
        scene.add_geometry(
            trimesh.load_path(
                [
                    self._lt.r_source().position,
                    self._lt.r_source().position
                    + (
                        self._lt.r_source().mesh.vertices[4]
                        - self._lt.r_source().mesh.vertices[0]
                    )
                    * 10,
                ],
            )
        )
        scene.add_geometry(
            trimesh.load_path(
                [
                    self._lt.l_source().position,
                    self._lt.l_source().position + self._lt.l_source().normal * 10,
                ],
            )
        )
        scene.add_geometry(lpos)
        # import pdb
        #
        # pdb.set_trace()
        scene.show()
