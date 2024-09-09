import typing

import pytest

import numpy as np
import numpy.typing as npt
from numpy.testing import assert_allclose
import trimesh
import trimesh.visual as tv

from selah.loudspeaker import Loudspeaker, LoudspeakerSpec


def box_from_origin(
    extents: typing.Union[npt.NDArray, list[float]],
) -> trimesh.Trimesh:
    if isinstance(extents, list):
        extents = np.array(extents)
    box = trimesh.primitives.Box(extents)
    box.apply_translation(extents / 2)
    box.visual.vertex_colors = tv.random_color()  # pyright: ignore
    return box


def show_scene(items: list[trimesh.Trimesh]):
    s = trimesh.Scene()
    for item in items:
        s.add_geometry(item)
    s.show()


def test_loudspeaker_position():
    base = LoudspeakerSpec(x_dim=2, y_dim=2, z_dim=2, y_offset=1, z_offset=1)

    assert np.allclose(
        Loudspeaker(base).mesh.vertices,
        np.array(
            [
                [2, 1, -1],
                [2, 1, 1],
                [2, -1, -1],
                [2, -1, 1],
                [0, 1, -1],
                [0, 1, 1],
                [0, -1, -1],
                [0, -1, 1],
            ]
        ),
    )

    assert np.allclose(
        Loudspeaker(base, position=[1, 1, 1]).mesh.vertices,
        np.array(
            [
                [3, 2, 0],
                [3, 2, 2],
                [3, 0, 0],
                [3, 0, 2],
                [1, 2, 0],
                [1, 2, 2],
                [1, 0, 0],
                [1, 0, 2],
            ]
        ),
    )

    assert np.allclose(
        Loudspeaker(base, normal=[0, 1, 0]).mesh.vertices,
        np.array(
            [
                [1, -2, -1],
                [1, -2, 1],
                [-1, -2, -1],
                [-1, -2, 1],
                [1, 0, -1],
                [1, 0, 1],
                [-1, 0, -1],
                [-1, 0, 1],
            ]
        ),
    )


def test_test_intersection_mesh():
    base = LoudspeakerSpec(2, 2, 2, y_offset=1, z_offset=1)

    # Speaker outside room
    assert not Loudspeaker(base, position=[4, 4, 4]).test_intersection(
        box_from_origin([1, 1, 1])
    )

    # Speaker fully inside room
    assert not Loudspeaker(base, position=[3, 3, 3]).test_intersection(
        box_from_origin([10, 10, 10])
    )

    # Speaker sharing one wall with room
    assert Loudspeaker(base, position=[0, 9, 9]).test_intersection(
        box_from_origin([10, 10, 10])
    )

    # Speaker same dimensions as room
    assert Loudspeaker(
        LoudspeakerSpec(10, 5, 5, y_offset=5, z_offset=5)
    ).test_intersection(box_from_origin([10, 10, 10]))

    # Speaker partially outside room
    assert Loudspeaker(base, position=[0, 9, 9]).test_intersection(
        box_from_origin([10, 10, 10])
    )

    # One edge of speaker rotated 45 degrees intersects wall
    # First, if we don't rotate, we don't hit the wall
    # For this test, rotate around a corner rather than an axis in the middle of X face
    assert not Loudspeaker(
        LoudspeakerSpec(2, 2, 2), position=[7.8, 3, 3]
    ).test_intersection(box_from_origin([10, 10, 10]))

    # Now rotate 45deg and one edge intersects
    assert Loudspeaker(
        LoudspeakerSpec(2, 2, 5), position=[7.8, 3, 3], normal=[-1, 0, 1]
    ).test_intersection(box_from_origin([10, 10, 10]))
