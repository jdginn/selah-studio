import numpy as np
import numpy.typing as npt
import trimesh


def dist(p: npt.NDArray, q: npt.NDArray, rs: npt.NDArray) -> float:
    x = p - q
    return np.linalg.norm(
        np.outer(np.dot(rs - q, x) / np.dot(x, x), x) + q - rs, axis=1
    )


def rotation_matrix(A: npt.NDArray, B: npt.NDArray) -> npt.NDArray:
    # a and b are in the form of numpy array

    ax = A[0]
    ay = A[1]
    az = A[2]

    bx = B[0]
    by = B[1]
    bz = B[2]

    au = A / (np.sqrt(ax * ax + ay * ay + az * az))
    bu = B / (np.sqrt(bx * bx + by * by + bz * bz))

    R = np.array(
        [
            [bu[0] * au[0], bu[0] * au[1], bu[0] * au[2]],
            [bu[1] * au[0], bu[1] * au[1], bu[1] * au[2]],
            [bu[2] * au[0], bu[2] * au[1], bu[2] * au[2]],
        ]
    )

    return R


def dir_from_points(p1: npt.NDArray, p2: npt.NDArray) -> npt.NDArray:
    unscaled = p2 - p1
    return unscaled / np.linalg.norm(unscaled)


def test_intersection(
    mesh: trimesh.Trimesh, point: npt.NDArray, normal: npt.NDArray
) -> bool:
    mp = trimesh.intersections.mesh_plane(mesh, normal, point)
    return len(mp) > 0
