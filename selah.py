import argparse
import zipfile
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
import pyfqmr
import typing

import IPython

namespace = {"schema": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

# material_map = {
#     # "Default": pra.materials_data["brick_wall_rough"],
#     # "Arch": pra.materials_data["brick_wall_rough"],
#     "Default": pra.materials_data["brickwork"],
#     "Arch": pra.materials_data["brickwork"],
# }


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
    def __init__(self, tree: ET.Element):
        self.name = tree.get("name")
        self._pra_walls = typing.List[pra.Wall]
        mesh = tree.find("schema:mesh", namespace)
        if mesh is None:
            raise RuntimeError
        meshv = mesh.find("schema:vertices", namespace)
        if meshv is None:
            raise RuntimeError
        vertices = [v for v in meshv.iter()]
        self.vertices = np.ndarray(shape=(len(vertices), 3))
        for i, v in enumerate(vertices[1:]):
            self.vertices[i] = [v.get("x"), v.get("y"), v.get("z")]
        mesht = mesh.find("schema:triangles", namespace)
        if mesht is None:
            raise RuntimeError
        triangles = [t for t in mesht.iter()]
        self.triangles = np.ndarray(shape=(len(triangles), 3))
        for i, t in enumerate(triangles[1:]):
            self.triangles[i] = [t.get("v1"), t.get("v2"), t.get("v3")]

    def simplify(self):
        # TODO: make this take the simplifier as a closure
        simplifier = pyqfmr.Simplify()
        simplifier.setMesh(self.vertices, self.triangles)
        simplifier.simplify_mesh(
            target_count=100, aggressiveness=8, preserve_border=True, verbose=10
        )
        self.vertices, self.triangles, _ = simplifier.getMesh()


def build_room(walls: typing.List[Wall]) -> pra.Room:
    reduc = 1000.0
    mat = pra.Material(energy_absorption=0.2, scattering=0.1)
    _walls = []
    for w in walls:
        for tri in w.triangles:
            corner = np.array(
                [w.vertices[tri[0]], w.vertices[tri[1]], w.vertices[tri[2]]],
                dtype=np.float32,
            )
            _walls.append(
                pra.Wall(
                    corner.T / reduc,
                    mat.energy_absorption["coeffs"],
                    mat.scattering["coeffs"],
                )
            )
    return pra.Room(
        _walls,
        fs=8000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )


class Room:
    reduc = 1000.0
    mat = pra.Material(energy_absorption=0.2, scattering=0.1)

    def __init__(self, walls: typing.List[Wall]):
        self._walls = []
        for w in walls:
            for tri in w.triangles:
                w.vertices[tri[0]]
                corner = np.array(
                    [w.vertices[tri[0]], w.vertices[tri[1]], w.vertices[tri[2]]],
                    dtype=np.float32,
                )
                self._walls.append(
                    pra.Wall(
                        corner.T / Room.reduc,
                        Room.mat.energy_absorption["coeffs"],
                        Room.mat.scattering["coeffs"],
                    )
                )

        pass

    # reduc = 1000.0
    #
    # material = pra.Material(energy_absorption=0.2, scattering=0.1)
    # walls = []
    # # for w in range(ntriang):
    # #     print(the_mesh.vectors[w].T / reduc)
    # #     walls.append(
    # #         pra.wall_factory(
    # #             the_mesh.vectors[w].T / reduc,
    # #             material.energy_absorption["coeffs"],
    # #             material.scattering["coeffs"],
    # #         )
    # #     )
    # for f in faces:
    #     corner = np.array(
    #         [vertices[f[0]], vertices[f[1]], vertices[f[2]]],
    #         dtype=np.float32,
    #     )
    #     # IPython.embed()
    #     walls.append(
    #         pra.wall_factory(
    #             corner.T / reduc,
    #             material.energy_absorption["coeffs"],
    #             material.scattering["coeffs"],
    #         )
    #     )
    #
    # room = (
    #     pra.Room(
    #         walls,
    #         fs=8000,
    #         max_order=3,
    #         ray_tracing=True,
    #         air_absorption=True,
    #     )
    #     .add_source([0.1, 0.1, 0.5])
    #     .add_microphone_array(np.c_[[1.0, 1.0, 0.4]])
    # )


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

    z = zipfile.ZipFile(args.file)
    f = z.extract("3D/3dmodel.model")
    tree = ET.parse(f)
    res = tree.getroot().find("schema:resources", namespace)
    if res is None:
        raise RuntimeError
    objects = [Wall(o) for o in res.findall("schema:object", namespace)]
    room = build_room(objects)
    room.add_source([0.1, 0.1, 0.5]).add_microphone_array(np.c_[[1.0, 1.0, 0.4]])

    # compute the rir
    room.image_source_model()
    room.ray_tracing()
    room.compute_rir()
    for s in room.sources:
        # s.set_ordering("strongest")
        for i in s.images:
            print("image:")
            print(i)
    room.plot_rir()
    # IPython.embed()

    # # show the room
    room.plot(img_order=1, mic_marker_size=25)
    plt.ylim(0, 10)
    plt.xlim(0, 10)
    plt.savefig("imgs/stl_room.png")
    plt.show()
