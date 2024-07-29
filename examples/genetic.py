from dataclasses import dataclass
import typing
import pprint

import trimesh
import matplotlib.pyplot as plt
import pygad

from selah.room import Arrival, Room
from selah.material import MaterialManager, Material
from selah.source import Loudspeaker
from selah.wall import Wall
from selah.exceptions import SelahException
from selah.sound import SPEED_OF_SOUND

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
class fixed_parameters:
    filename: str = "examples/resources/studio.3mf"
    rfz_radius: float = 0.3
    num_samples: int = 80_000
    max_time: float = 80 / 1000
    min_gain: float = -18
    order: int = 8
    max_listen_pos: float = 2.4
    min_listen_pos: float = 1.3


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


def get_arrivals(solution) -> tuple[Room, typing.List[Arrival]]:
    genetic_params = training_parameters(*solution)
    fixed_params = fixed_parameters()

    scene = trimesh.load(fixed_params.filename)
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
        height=genetic_params.height,
        speaker_height=genetic_params.speaker_height,
        dist_from_wall=genetic_params.dist_from_wall,
        dist_from_center=genetic_params.dist_from_center,
        deviation=genetic_params.deviation_from_equilateral,
        source=Loudspeaker(
            vert_disp={0: 0, 25: -5, 60: -6, 80: -12, 90: -100},
            horiz_disp={0: 0, 30: -3, 50: -6, 60: -9, 90: -100},
        ),
        rfz_radius=fixed_params.rfz_radius,
    )
    listen_pos = room._lt.listening_pos()
    if listen_pos[0] <= fixed_params.min_listen_pos:
        raise ListeningPositionError("Too close to front wall")
    if listen_pos[0] >= fixed_params.max_listen_pos:
        raise ListeningPositionError("Too close to back wall")
    room.ceiling_absorber(
        genetic_params.ceiling_diffuser_height,
        genetic_params.ceiling_diffuser_length,
        genetic_params.ceiling_diffuser_width,
        genetic_params.ceiling_diffuser_position,
    )
    l_arrivals = room.trace_arrivals(
        room._lt.source,
        room._lt.l_source(),
        room._lt.listening_pos(),
        num_samples=fixed_params.num_samples,
        max_time=fixed_params.max_time,
        min_gain=fixed_params.min_gain,
        order=fixed_params.order,
    )
    r_arrivals = room.trace_arrivals(
        room._lt.source,
        room._lt.r_source(),
        room._lt.listening_pos(),
        num_samples=fixed_params.num_samples,
        max_time=fixed_params.max_time,
        min_gain=fixed_params.min_gain,
        order=fixed_params.order,
    )
    arrivals = l_arrivals + r_arrivals
    return room, arrivals


def fitness_func(ga_instance, solution, solution_idx) -> float:
    params = training_parameters(*solution)
    fixed_params = fixed_parameters()
    try:
        _, arrivals = get_arrivals(solution)
    except SelahException as ex:
        print(f"Invalid solution: {ex}")
        return 0
    arrivals.sort(key=lambda a: a.total_dist)
    if len(arrivals) == 0:
        print(f"Too good to be true: {fixed_params.max_time * 1000}")
        return 0
        return params.max_time * 1000
    ITD = float(arrivals[0].total_dist / SPEED_OF_SOUND * 1000)
    print(f"ITD: {ITD:.1f}")
    return ITD


if __name__ == "__main__":
    gene_space = training_parameters(
        speaker_height={"low": 1.3, "high": 2.3},
        dist_from_center={"low": 0.85, "high": 1.3},
        dist_from_wall={"low": 0.3, "high": 0.6},
        deviation_from_equilateral={"low": -0.5, "high": 0.5},
        ceiling_diffuser_height={"low": 2.5, "high": 2.75},
        ceiling_diffuser_width={"low": 1.0, "high": 4.75},
        ceiling_diffuser_length={"low": 1.0, "high": 2.75},
        ceiling_diffuser_position={"low": 0.0, "high": 2.5},
    )
    ga_instance = pygad.GA(
        num_generations=8,
        num_parents_mating=4,
        fitness_func=fitness_func,
        sol_per_pop=32,
        num_genes=len(gene_space.aslist()),
        gene_space=gene_space.aslist(),
        mutation_probability=0.4,
        parent_selection_type="tournament",
        K_tournament=4,
        crossover_type="two_points",
        crossover_probability=0.7,
        keep_elitism=8,
        parallel_processing=["process", 32],
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
