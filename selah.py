import argparse
import zipfile
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
import pyfqmr
import trimesh as tr

# from stl import mesh

import IPython

namespace = {"schema": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

# Objects contain:
#   name: string
#   material: string
#   vertices: numpy.ndarray
#       array of vertices of shape (n_vertices, 3)
#   faces: numpy.ndarray
#       array of faces of shape (n_faces, 3)
#


class Material:
    # name: string
    # absorption: ?
    # scattering: ?
    # diffusion: ?
    def __init__(self):
        pass


class Wall:
    def __init__(self, tree: ET.Element):
        self.name = tree.get("name")
        mesh = tree.find("schema:mesh", namespace)
        if mesh is None:
            raise RuntimeError
        meshv = mesh.find("schema:vertices", namespace)
        if meshv is None:
            raise RuntimeError
        vertices = [v for v in meshv.iter()]
        self.vertices = np.ndarray(shape=(len(vertices), 3))
        for i, v in enumerate(vertices):
            self.vertices[i] = [v.get("x"), v.get("y"), v.get("z")]
        mesht = mesh.find("schema:triangles", namespace)
        if mesht is None:
            raise RuntimeError
        triangles = [t for t in mesht.iter()]
        self.triangles = np.ndarray(shape=(len(triangles), 3))
        for i, t in enumerate(triangles):
            self.triangles[i] = [t.get("v1"), t.get("v2"), t.get("v3")]
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process room from 3mf file")
    parser.add_argument("--file", type=str, required=True, help="Path to 3mf file")
    args = parser.parse_args()

    z = zipfile.ZipFile(args.file)
    f = z.extract("3D/3dmodel.model")
    tree = ET.parse(f)
    res = tree.getroot().find("schema:resources", namespace)
    if res is None:
        raise RuntimeError
    objects = [Wall(o) for o in res.findall("schema:object", namespace)]

    # simplifier = pyfqmr.Simplify()
    # simplifier.setMesh(mg.vertices, mg.faces)
    # simplifier.simplify_mesh(
    #     target_count=100, aggressiveness=8, preserve_border=True, verbose=10
    # )

    IPython.embed()

    # mesh = tr.load(args.file, file_type="3mf")
    # ntriang, nvec, npts = the_mesh.vectors.shape
    #
    # simplifier = pyfqmr.Simplify()
    # simplifier.setMesh(mg.vertices, mg.faces)
    # simplifier.simplify_mesh(
    #     target_count=100, aggressiveness=8, preserve_border=True, verbose=10
    # )
    # vertices, faces, normals = simplifier.getMesh()
    #
    # # vertices = mg.vertices
    # # faces = mg.faces
    # #
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
    #
    # # compute the rir
    # room.image_source_model()
    # room.ray_tracing()
    # room.compute_rir()
    # # for s in room.sources:
    # #     # s.set_ordering("strongest")
    # #     for i in s.images:
    # #         print("image:")
    # #         print(i)
    # room.plot_rir()
    # # IPython.embed()
    #
    # # # show the room
    # room.plot(img_order=1, mic_marker_size=25)
    # plt.ylim(0, 10)
    # plt.xlim(0, 10)
    # plt.savefig("imgs/stl_room.png")
    # plt.show()
