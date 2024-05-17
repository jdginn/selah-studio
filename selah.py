import argparse
import zipfile
import xml.etree.ElementTree as ET
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
import pyfqmr
import trimesh as tr
from stl import mesh

import IPython

namespace = {"schema": "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process room from 3mf file")
    parser.add_argument("--file", type=str, required=True, help="Path to 3mf file")
    args = parser.parse_args()

    z = zipfile.ZipFile(args.file)
    f = z.extract("3D/3dmodel.model")
    tree = ET.parse(f)
    objects = (
        tree.getroot()
        .find("schema:resources", namespace)
        .findall("schema:object", namespace)
    )

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
