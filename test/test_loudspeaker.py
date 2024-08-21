import typing

import pytest

import numpy as np
import numpy.typing as npt
import trimesh
import trimesh.visual as tv

from selah.loudspeaker import Loudspeaker


def box_from_origin(
    extents: typing.Union[npt.NDArray, list[float]],
) -> trimesh.Trimesh:
    if isinstance(extents, list):
        extents = np.array(extents)
    box = trimesh.primitives.Box(extents)
    box.apply_translation(extents / 2)
    box.visual.vertex_colors = tv.random_color()  # pyright: ignore
    return box


def test_test_intersection_mesh():
    # Speaker outside room
    assert not Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(box_from_origin([1, 1, 1]), placement=[4, 4, 4])

    # Speaker fully inside room
    assert not Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(box_from_origin([10, 10, 10]), placement=[3, 3, 3])

    # Speaker sharing one wall with room
    assert Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(box_from_origin([10, 10, 10]), placement=[0, 2, 2])

    # Speaker same dimensions as room
    assert Loudspeaker(
        x_dim=10, y_dim=10, z_dim=10, y_offset=5, z_offset=5
    ).test_intersection(box_from_origin([10, 10, 10]), placement=[0, 0, 0])

    # Speaker partially outside room
    assert Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(box_from_origin([10, 10, 10]), placement=[0, 9, 9])

    # One edge of speaker rotated 45 degrees intersects wall
    # First, if we don't rotate, we don't hit the wall
    assert not Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(box_from_origin([10, 10, 10]), placement=[3, 3, 8.8])
    # Now rotate 45deg and one edge intersects
    assert Loudspeaker(
        x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1
    ).test_intersection(
        box_from_origin([10, 10, 10]), placement=[3, 3, 8.8], norm=[1, 1, 0]
    )
