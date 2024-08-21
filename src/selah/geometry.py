import numpy as np
import numpy.typing as npt
import trimesh


def lineseg_dist(a: npt.NDArray, b: npt.NDArray, p: npt.NDArray) -> npt.NDArray:
    """Returns the minimum distance between point p and the line segment defined by points a and b"""

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)

    return np.hypot(h, np.linalg.norm(c))


def dir_from_points(p1: npt.NDArray, p2: npt.NDArray) -> npt.NDArray:
    """Returns the unit vector representing the direction of a line segment between poitns p1 and p2"""
    unscaled = p2 - p1
    return unscaled / np.linalg.norm(unscaled)


def test_intersection(
    mesh: trimesh.Trimesh, point: npt.NDArray, normal: npt.NDArray
) -> bool:
    """Returns true if the wall defiend by a point and normal intersects the mesh"""
    mp = trimesh.intersections.mesh_plane(mesh, normal, point)
    return len(mp) > 0
