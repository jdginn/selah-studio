import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
sys.path.append(SOURCE_PATH)

from dataclasses import dataclass
import typing
import trimesh
import matplotlib.pyplot as plt
import trimesh.exchange.export as export
import trimesh.visual

import selah

from selah.material import MaterialManager, Material
from selah.sound import SPEED_OF_SOUND
from selah.loudspeaker import LoudspeakerSpec, Loudspeaker
from selah.wall import Wall
from selah.exceptions import SelahException
from selah.source import SourceException, Reflection
from selah.room import Room, CollisionException

# materials: typing.Dict[str, Material] = {
#     "brick": Material(0.04),
#     "glass": Material(0.00),
#     "gypsum": Material(0.05),
#     "diffuser": Material(0.999),
#     "wood": Material(0.1),
#     "12cm_rockwool": Material(0.999),
#     # "12cm_rockwool": Material(0.92),
#     "24cm_rockwool": Material(0.999),
#     # "24cm_rockwool": Material(0.94),
#     "30cm_rockwool": Material(0.999),
#     # "30cm_rockwool": Material(0.95),
# }

wall_materials = {
    "default": "brick",
    "Floor": "wood",
    "Front A": "gypsum",
    "Front B": "gypsum",
    "Back Diffuser": "diffuser",
    "Ceiling Diffuser": "diffuser",
    "Back A": "24cm_rockwool",
    "Back B": "24cm_rockwool",
    "Street A": "24cm_rockwool",
    "Street B": "24cm_rockwool",
    "Street C": "24cm_rockwool",
    "Street D": "24cm_rockwool",
    "Street E": "24cm_rockwool",
    "Hall A": "24cm_rockwool",
    "Hall B": "24cm_rockwool",
    "Hall E": "24cm_rockwool",
    "Entry Back": "24cm_rockwool",
    "Entry Front": "24cm_rockwool",
    "Window A": "glass",
    "Window B": "glass",
    "Door": "12cm_rockwool",
    "left speaker wall": "gypsum",
    "right speaker wall": "gypsum",
}


class ListeningPositionError(SelahException):
    """Indicates the listening position has been placed outside the valid area"""


@dataclass
class parameters:
    # filename: str = "examples/resources/studio.3mf"
    filename: str = "WIP.3mf"
    height: float = 1.4
    speaker_height: float = 1.9
    dist_from_wall: float = 0.45
    dist_from_center: float = 1.1
    deviation_from_equilateral: float = 0.3
    max_listen_pos: float = 2.7
    min_listen_pos: float = 1.3
    ceiling_diffuser_height: float = 2.3
    ceiling_diffuser_length: float = 2.5
    ceiling_diffuser_width: float = 5.0
    ceiling_diffuser_position: float = 0.75
    rfz_radius: float = 0.3
    num_samples: int = 10_000
    max_time: float = 40 / 1000
    min_gain: float = -12
    order: int = 9


if __name__ == "__main__":
    params = parameters()

    scene = trimesh.load(params.filename, process=True)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    scene = scene.scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    mm = MaterialManager()
    mm.set_wall_materials(wall_materials)
    blue = (128, 234, 255)
    window = ["Window A", "Window B"]
    walls: typing.List[Wall] = []
    for name, mesh in scene.geometry.items():
        if name in window:
            mesh.visual = trimesh.visual.ColorVisuals(
                mesh, trimesh.visual.color.to_rgba(blue)
            )
        walls.append(Wall(name, mesh))
    room = Room(walls, mm)

    try:
        room.listening_triangle(
            wall_name="Front A",
            height=params.height,
            speaker_height=params.speaker_height,
            dist_from_wall=params.dist_from_wall,
            dist_from_center=params.dist_from_center,
            deviation=params.deviation_from_equilateral,
            source=LoudspeakerSpec(
                x_dim=0.380,
                y_dim=0.256,
                z_dim=0.529,
                y_offset=0.150,
                z_offset=0.235,
                vert_disp={0: 0, 25: -5, 60: -6, 80: -12, 90: -100},
                horiz_disp={0: 0, 30: -3, 50: -6, 60: -9, 90: -100},
            ),
            rfz_radius=params.rfz_radius,
        )
    except CollisionException as ex:
        print(f"Exception: {type(ex)}")
        room.show()
    listen_pos = room._lt.listening_pos()
    if listen_pos[0] <= params.min_listen_pos:
        raise ListeningPositionError("Too close to front wall")
    if listen_pos[0] >= params.max_listen_pos:
        raise ListeningPositionError("Too close to back wall")
    room.ceiling_absorber(
        params.ceiling_diffuser_height,
        params.ceiling_diffuser_length,
        params.ceiling_diffuser_width,
        params.ceiling_diffuser_position,
    )
    try:
        l_arrivals = room.trace_arrivals(
            room._lt.l_source(),
            room._lt.listening_pos(),
            num_samples=params.num_samples,
            max_time=params.max_time,
            min_gain=params.min_gain,
            order=params.order,
            ignore_walls="Floor",
            frequency=10_000,
        )
        r_arrivals = room.trace_arrivals(
            room._lt.r_source(),
            room._lt.listening_pos(),
            num_samples=params.num_samples,
            max_time=params.max_time,
            min_gain=params.min_gain,
            order=params.order,
            ignore_walls="Floor",
            frequency=10_000,
        )
        arrivals = l_arrivals + r_arrivals
        arrivals.sort(key=lambda a: a.total_dist)
        ITD = float(
            (arrivals[0].total_dist - room._lt.listening_dist) / SPEED_OF_SOUND * 1000
        )
        print(f"ITD: {ITD:.1f}ms")
        print(f"Critical distance: {room.critical_distance():.2f}m")
        print(f"Listening distance: {room._lt.listening_dist:.2f}m")
        print(
            f"Deviation from equilateral: {abs(room._lt.listening_dist - room._lt.dist_from_center*2):.2f}m"
        )
        print(f"Schroeder frequency: {room.schroeder():.1f}Hz")
        print(f"Volume: {room.volume():.1f}m3")
        plt.ion()
        fig = plt.figure()
        room.plot_arrivals_interactive(fig, arrivals, False)
        plt.show(block=True)
        room.show()
    except SourceException as ex:
        print(ex.message)
        arrivals = [ex.source]
        plt.ion()
        fig = plt.figure()
        room.plot_arrivals_interactive(fig, arrivals, False)
        plt.show(block=True)
        room.show()
