import os
import sys

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(PROJECT_PATH, "src")
sys.path.append(SOURCE_PATH)

from dataclasses import dataclass
import typing
import pprint
import math

import trimesh
import matplotlib.pyplot as plt
import pygad

from selah.room import CollisionException, Room
from selah.material import MaterialManager, Material
from selah.wall import Wall
from selah.source import Source
from selah.exceptions import SelahException
from selah.loudspeaker import Directivity, LoudspeakerSpec
from selah.sound import SPEED_OF_SOUND

materials: typing.Dict[str, Material] = {
    "brick": Material(0.04),
    "glass": Material(0.00),
    "gypsum": Material(0.05),
    "diffuser": Material(0.99),
    "wood": Material(0.1),
    "12cm_rockwool": Material(0.94),
    "24cm_rockwool": Material(0.96),
    "30cm_rockwool": Material(0.97),
}

wall_materials = {
    "default": "brick",
    "Floor": "wood",
    "Front A": "gypsum",
    "Front B": "gypsum",
    "Back Diffuser": "diffuser",
    "Ceiling Diffuser": "diffuser",
    "Cutout Top": "24cm_rockwool",
    "Back A": "24cm_rockwool",
    "Back B": "24cm_rockwool",
    "Street A": "24cm_rockwool",
    "Street B": "24cm_rockwool",
    "Street C": "24cm_rockwool",
    "Street D": "24cm_rockwool",
    "Street E": "24cm_rockwool",
    # "Arch C1": "24cm_rockwool",
    "Hall A": "24cm_rockwool",
    "Hall B": "24cm_rockwool",
    "Hall E": "24cm_rockwool",
    "Entry Back": "24cm_rockwool",
    "Entry Front": "24cm_rockwool",
    "Window A": "glass",
    "Window B": "glass",
    "Door": "12cm_rockwool",
    "Door Side A": "24cm_rockwool",
    "Door Side B": "24cm_rockwool",
    "left speaker wall": "gypsum",
    "right speaker wall": "gypsum",
}
kh310_horiz_disp: dict[float, float] = {0: 0, 30: 0, 50: -3, 70: -6, 80: -9, 90: -20}
kh310_vert_disp: dict[float, float] = {0: 0, 30: -3, 60: -6, 90: -9, 100: -30}

mum8 = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.256,
    z_dim=0.52,
    y_offset=0.096,
    z_offset=0.412,
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)
mum8_swap = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.52,
    z_dim=0.256,
    y_offset=0.195,
    z_offset=0.128,
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)
mum8_horiz = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.52,
    z_dim=0.256,
    y_offset=0.195,
    z_offset=0.128,
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)
mum8_horiz_swap = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.52,
    z_dim=0.256,
    y_offset=(0.52 - 0.195),
    z_offset=0.128,
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)
mum8_upside_down = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.256,
    z_dim=0.52,
    y_offset=0.096,
    z_offset=(0.52 - 0.412),
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)
mum8_upside_down_swap = LoudspeakerSpec(
    x_dim=0.38,
    y_dim=0.256,
    z_dim=0.52,
    y_offset=(0.256 - 0.096),
    z_offset=(0.52 - 0.412),
    directivity=Directivity(kh310_horiz_disp, kh310_vert_disp),
)


class ListeningPositionError(SelahException):
    """Indicates the listening position has been placed outside the valid area"""


@dataclass
class fixed_parameters:
    # filename: str = "examples/resources/studio.3mf"
    # filename: str = "WIP.3mf"
    filename: str = "Cutout.3mf"
    rfz_radius: float = 0.3
    num_samples: int = 2_000
    max_time: float = 35 / 1000
    min_gain: float = -10
    order: int = 10
    max_listen_pos: float = 2.4
    min_listen_pos: float = 1.3


global_params = fixed_parameters()

global_speaker = mum8


@dataclass
class training_parameters:
    height: typing.Union[float, dict[str, float]] = 1.4
    speaker_height: typing.Union[float, dict[str, float]] = 1.4
    dist_from_wall: typing.Union[float, dict[str, float]] = 0.3
    dist_from_center: typing.Union[float, dict[str, float]] = 0.9
    deviation_from_equilateral: typing.Union[float, dict[str, float]] = 0.0
    ceiling_diffuser_height: typing.Union[float, dict[str, float]] = 2.3
    ceiling_diffuser_length: typing.Union[float, dict[str, float]] = 1.0
    ceiling_diffuser_width: typing.Union[float, dict[str, float]] = 1.0
    ceiling_diffuser_position: typing.Union[float, dict[str, float]] = 1.5

    def aslist(self):
        retlist = []
        for name, val in self.__dict__.items():
            if isinstance(val, dict):
                retlist.append(val)
            else:
                retlist.append([val])
        return retlist


def optimize_to_target(target, scale, x, is_abs=True) -> float:
    scale = scale + 1
    if is_abs:
        return scale ** (-(abs(x - target) ** 2) / scale)
    return abs(scale ** (-((x - target) ** 2) / scale))


def fitness_func(ga_instance, solution, solution_idx):
    genetic_params = training_parameters(*solution)
    fixed_params = global_params
    dev_fom = optimize_to_target(0, 1, abs(genetic_params.deviation_from_equilateral))
    height_fom = optimize_to_target(
        2.75, 1, abs(genetic_params.ceiling_diffuser_height)
    )
    dist_fom = optimize_to_target(
        fixed_params.min_listen_pos, 5, genetic_params.dist_from_wall
    )
    valid = True

    scene = trimesh.load(fixed_params.filename)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    scene = scene.scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    mm = MaterialManager(materials)
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
            height=genetic_params.height,
            speaker_height=genetic_params.speaker_height,
            dist_from_wall=genetic_params.dist_from_wall,
            dist_from_center=genetic_params.dist_from_center,
            deviation=genetic_params.deviation_from_equilateral,
            source_spec=global_speaker,
            rfz_radius=fixed_params.rfz_radius,
        )
    except CollisionException:
        return [
            0,
            -abs(genetic_params.deviation_from_equilateral),
            genetic_params.ceiling_diffuser_height,
            0,
            -genetic_params.dist_from_wall,
            False,
        ]
    wall_area = (
        room.get_wall("right speaker wall").mesh.area
        + room.get_wall("left speaker wall").mesh.area
    )
    area_fom = optimize_to_target(5, 2, wall_area, False)
    listen_pos = room._lt.listening_pos
    if listen_pos[0] <= fixed_params.min_listen_pos:
        valid = False
    if listen_pos[0] >= fixed_params.max_listen_pos:
        valid = False
    room.ceiling_absorber(
        genetic_params.ceiling_diffuser_height,
        genetic_params.ceiling_diffuser_length,
        genetic_params.ceiling_diffuser_width,
        genetic_params.ceiling_diffuser_position,
    )
    try:
        l_arrivals = room.trace_arrivals(
            room._l_source,
            room._lt.listening_pos,
            num_samples=fixed_params.num_samples,
            max_time=fixed_params.max_time,
            min_gain=fixed_params.min_gain,
            order=fixed_params.order,
            ignore_walls="Floor",
        )
        r_arrivals = room.trace_arrivals(
            room._r_source,
            room._lt.listening_pos,
            num_samples=fixed_params.num_samples,
            max_time=fixed_params.max_time,
            min_gain=fixed_params.min_gain,
            order=fixed_params.order,
            ignore_walls="Floor",
        )
    except SelahException as ex:
        print(f"Arrival tracing exception: {ex.message}")
        return [
            0,
            -abs(genetic_params.deviation_from_equilateral),
            genetic_params.ceiling_diffuser_height,
            0,
            -genetic_params.dist_from_wall,
            False,
        ]
    arrivals = l_arrivals + r_arrivals
    arrivals.sort(key=lambda a: a.total_dist)
    if len(arrivals) == 0:
        print("No arrivals")
        return [
            0,
            -abs(genetic_params.deviation_from_equilateral),
            genetic_params.ceiling_diffuser_height,
            -wall_area,
            -genetic_params.dist_from_wall,
            False,
        ]
    ITD = float(
        (arrivals[0].total_dist - room._lt.listening_dist) / SPEED_OF_SOUND * 1000
    )
    itd_fom = ITD / (20)
    print(
        f"ITD: {ITD:.2f}, deviation: {genetic_params.deviation_from_equilateral:.2f}, area: {wall_area:.2f}, height: {genetic_params.ceiling_diffuser_height:.2f}, dist: {genetic_params.dist_from_wall:.2f}",
    )
    return [
        ITD,
        -abs(genetic_params.deviation_from_equilateral),
        genetic_params.ceiling_diffuser_height,
        -wall_area,
        -genetic_params.dist_from_wall,
        valid,
    ]


def get_arrivals(solution) -> typing.Tuple[Room, typing.List[Source]]:
    genetic_params = training_parameters(*solution)
    fixed_params = global_params
    scene = trimesh.load(fixed_params.filename)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    scene = scene.scaled(1 / 1000)
    if not isinstance(scene, trimesh.Scene):
        raise RuntimeError
    mm = MaterialManager(materials)
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

    room.listening_triangle(
        wall_name="Front A",
        height=genetic_params.height,
        speaker_height=genetic_params.speaker_height,
        dist_from_wall=genetic_params.dist_from_wall,
        dist_from_center=genetic_params.dist_from_center,
        deviation=genetic_params.deviation_from_equilateral,
        source_spec=global_speaker,
        rfz_radius=fixed_params.rfz_radius,
    )
    room.ceiling_absorber(
        genetic_params.ceiling_diffuser_height,
        genetic_params.ceiling_diffuser_length,
        genetic_params.ceiling_diffuser_width,
        genetic_params.ceiling_diffuser_position,
    )
    l_arrivals = room.trace_arrivals(
        room._l_source,
        room._lt.listening_pos,
        num_samples=fixed_params.num_samples,
        max_time=fixed_params.max_time,
        min_gain=fixed_params.min_gain,
        order=fixed_params.order,
        ignore_walls="Floor",
    )
    r_arrivals = room.trace_arrivals(
        room._r_source,
        room._lt.listening_pos,
        num_samples=fixed_params.num_samples,
        max_time=fixed_params.max_time,
        min_gain=fixed_params.min_gain,
        order=fixed_params.order,
        ignore_walls="Floor",
    )
    arrivals = l_arrivals + r_arrivals
    arrivals.sort(key=lambda a: a.total_dist)
    return room, arrivals


if __name__ == "__main__":
    gene_space = training_parameters(
        speaker_height={"low": 1.1, "high": 2.3},
        dist_from_center={"low": 1.2, "high": 1.5},
        dist_from_wall={"low": 0.4, "high": 0.6},
        deviation_from_equilateral={"low": -0.2, "high": 0.2},
        ceiling_diffuser_height={"low": 2.4, "high": 2.75},
        ceiling_diffuser_width={"low": 2.0, "high": 4.75},
        ceiling_diffuser_length={"low": 2.0, "high": 2.75},
        ceiling_diffuser_position={"low": 0.0, "high": 2.5},
    )
    speakers = {
        "mum8": mum8,
        "mum8_swap": mum8_swap,
        "mum8_horiz": mum8_horiz,
        "mum8_horiz_swap": mum8_horiz_swap,
        "mum8_upside_down": mum8_upside_down,
        "mum8_upside_down_swap": mum8_upside_down_swap,
    }
    for k, speaker in speakers.items():
        ga_instance = pygad.GA(
            num_generations=4,
            num_parents_mating=16,
            fitness_func=fitness_func,
            sol_per_pop=24,
            num_genes=len(gene_space.aslist()),
            gene_space=gene_space.aslist(),
            mutation_probability=0.7,
            # mutation_type="adaptive",
            parent_selection_type="nsga2",
            crossover_type="two_points",
            crossover_probability=0.7,
            keep_elitism=4,
            parallel_processing=["process", 32],
            save_solutions=True,
            save_best_solutions=True,
        )
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        pprint.pprint(
            f"Parameters of the best solution : {training_parameters(*solution)}"
        )
        pprint.pprint(f"Fitness value of the best solution = {solution_fitness}")
        # ga_instance.plot_fitness(
        #     label=[
        #         "ITD",
        #         "Deviation",
        #         "Ceiling Height",
        #         "Speaker Wall Area",
        #         "Distance From Front Wall",
        #         "Valid",
        #     ]
        # )

        ga_instance.save(k)
    # import IPython
    #
    # IPython.embed()

    # solutions = ga_instance.solutions_fitness
    # solutions = [x for x in solutions if x[5]]
    # solutions = sorted(solutions, key=lambda x: x[0], reverse=True)
    # # Best 10% by ITD
    # solutions = sorted(solutions, key=lambda x: x[0], reverse=True)[
    #     0 : (len(solutions) // 10 + 1)
    # ]
    # # Best 10% by Ceiling Height
    # solutions = sorted(solutions, key=lambda x: x[2], reverse=True)[
    #     0 : (len(solutions) // 10 + 1)
    # ]
    # # Best 10% by Deviation
    # solutions = sorted(solutions, key=lambda x: -x[1], reverse=True)[
    #     0 : (len(solutions) // 10 + 1)
    # ]
    # # Best 10% by Wall Area
    # solutions = sorted(solutions, key=lambda x: -x[3], reverse=True)[
    #     0 : (len(solutions) // 10 + 1)
    # ]

    # import IPython
    #
    # IPython.embed()

    # room, arrivals = get_arrivals(solution)
    # plt.ion()
    # fig = plt.figure()
    # room.plot_arrivals_interactive(fig, arrivals, False)
    # plt.show(block=True)
    # room.show()
