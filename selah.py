import argparse
import pyroomacoustics as pra
import matplotlib.pyplot as plt
import numpy as np
import pyfqmr
import trimesh as tr
from stl import mesh

import IPython

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process room from STL")
    parser.add_argument("--file", type=str, required=True, help="Path to STL file")
    args = parser.parse_args()

    mg = tr.load(args.file)
    the_mesh = mesh.Mesh.from_file(args.file)
    ntriang, nvec, npts = the_mesh.vectors.shape

    simplifier = pyfqmr.Simplify()
    simplifier.setMesh(mg.vertices, mg.faces)
    simplifier.simplify_mesh(
        target_count=100, aggressiveness=7, preserve_border=True, verbose=10
    )
    vertices, faces, normals = simplifier.getMesh()

    # vertices = mg.vertices
    # faces = mg.faces
    #
    reduc = 1000.0

    trs1 = []
    trs2 = []

    for f in range(3):
        print(the_mesh.vectors[f].T)
        trs1.append(the_mesh.vectors[f].T)

    print("-----")

    for f in range(3):
        corner = np.array(
            [vertices[faces[f][0]], vertices[faces[f][1]], vertices[faces[f][2]]],
            dtype=np.float32,
        )
        print(corner.T)
        trs2.append(corner.T)

    material = pra.Material(energy_absorption=0.2, scattering=0.1)
    walls = []
    # for w in range(ntriang):
    #     print(the_mesh.vectors[w].T / reduc)
    #     walls.append(
    #         pra.wall_factory(
    #             the_mesh.vectors[w].T / reduc,
    #             material.energy_absorption["coeffs"],
    #             material.scattering["coeffs"],
    #         )
    #     )
    for f in faces:
        print(f)
        corner = np.array(
            # [vertices[f[0]] / reduc, vertices[f[1]] / reduc, vertices[f[2]] / reduc],
            [vertices[f[0]], vertices[f[1]], vertices[f[2]]],
            dtype=np.float32,
        )
        print(corner.T / reduc)
        # IPython.embed()
        walls.append(
            pra.wall_factory(
                corner.T / reduc,
                material.energy_absorption["coeffs"],
                material.scattering["coeffs"],
            )
        )

    room = (
        pra.Room(
            walls,
            fs=8000,
            max_order=3,
            ray_tracing=False,
            air_absorption=True,
        )
        # .add_source([0.5, 0.5, 0.5])
        # .add_microphone_array(np.c_[[1.0, 1.0, 1.0]])
    )

    # compute the rir
    # room.image_source_model()
    # room.ray_tracing()
    # room.compute_rir()
    # room.plot_rir()
    # plt.savefig("imgs/stl_rir_plot.png")

    # show the room
    room.plot(img_order=1, xlim=10.0, ylim=10.0, zlim=10.0)
    plt.ylim(0, 10)
    plt.xlim(0, 10)
    plt.savefig("imgs/stl_room.png")
    plt.show()
