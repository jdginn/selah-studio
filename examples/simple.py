import argparse
from dataclasses import dataclass
import typing
import trimesh
import matplotlib.pyplot as plt

from selah.material import MaterialManager, Material
from selah.source import Source
from selah.wall import Wall
from selah.exceptions import SelahException
from selah.room import Room

materials: typing.Dict[str, Material] = {
    "brick": Material(0.04),
    "glass": Material(0.00),
    "gypsum": Material(0.05),
    "diffuser": Material(0.99),
    "wood": Material(0.1),
    "12cm_rockwool": Material(0.99),
    "24cm_rockwool": Material(0.95),
    "30cm_rockwool": Material(0.95),
}

wall_materials = {
    "default": "brick",
    "Floor": "wood",
    "Front": "gypsum",
    "Back Diffuser": "diffuser",
    "Ceiling Diffuser": "12cm_rockwool",
    "Cutout Diffuser": "24cm_rockwool",
    "Street Absorber": "12cm_rockwool",
    "Street Absorber Shelf": "wood",
    "Back Hallway Absorber": "12cm_rockwool",
    "Back Hallway Absorber Shelf": "wood",
    "Front Hallway Absorber": "12cm_rockwool",
    "Front Hallway Absorber Shelf": "wood",
    "Window": "glass",
    "Floor Wedge": "12cm_rockwool",
    "Door": "12cm_rockwool",
    "Doorway Front": "12cm_rockwool",
    "left speaker wall": "gypsum",
    "right speaker wall": "gypsum",
}


class ListeningPositionError(SelahException):
    """Indicates the listening position has been placed outside the valid area"""


@dataclass
class parameters:
    filename: str = "examples/resources/studio.3mf"
    height: float = 1.4
    speaker_height: float = 2.4
    dist_from_wall: float = 0.3
    dist_from_center: float = 0.95
    deviation_from_equilateral: float = 0.5
    max_listen_pos: float = 2.4
    min_listen_pos: float = 1.3
    ceiling_diffuser_height: float = 2.3
    ceiling_diffuser_length: float = 1.0
    ceiling_diffuser_width: float = 1.0
    ceiling_diffuser_position: float = 1.5
    rfz_radius: float = 0.3
    num_samples: int = 1000
    max_time: float = 80 / 1000

    min_gain: float = -15
    order: int = 4


if __name__ == "__main__":
    params = parameters()

    scene = trimesh.load(params.filename)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    scene = scene.scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    mm = MaterialManager(materials)
    mm.set_wall_materials(wall_materials)
    room = Room([Wall(name, mesh) for (name, mesh) in scene.geometry.items()], mm)
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
    room.ceiling_absorber(
        params.ceiling_diffuser_height,
        params.ceiling_diffuser_length,
        params.ceiling_diffuser_width,
        params.ceiling_diffuser_position,
    )
    l_arrivals = room.trace(
        room._lt.source,
        room._lt.l_source(),
        room._lt.listening_pos(),
        num_samples=params.num_samples,
        max_time=params.max_time,
        min_gain=params.min_gain,
        order=params.order,
    )
    r_arrivals = room.trace(
        room._lt.source,
        room._lt.r_source(),
        room._lt.listening_pos(),
        num_samples=params.num_samples,
        max_time=params.max_time,
        min_gain=params.min_gain,
        order=params.order,
    )
    arrivals = l_arrivals + r_arrivals

    plt.ion()
    fig = plt.figure()
    room.plot_arrivals_interactive(fig, arrivals, False)
    plt.show(block=True)
    room.mesh.show()
