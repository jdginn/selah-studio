import numpy as np
import numpy.typing as npt
import trimesh

from . import geometry

kh420_horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100}
kh420_vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30}

kh310_horiz_disp: dict[float, float] = {0: 0, 30: 0, 50: -3, 70: -6, 80: -9, 90: -20}
kh310_vert_disp: dict[float, float] = {0: 0, 30: -3, 60: -6, 90: -9, 100: -30}


class Source:
    """Dispersions in degrees"""

    # Takes arguments mapping degrees to gain in dB
    def __init__(
        self,
        horiz_disp: dict[float, float] = {0: 0, 30: 0, 60: -12, 70: -100},
        vert_disp: dict[float, float] = {0: 0, 30: -9, 60: -15, 70: -19, 80: -30},
        x_dim: float = 0.520,
        y_dim: float = 0.256,
        z_dim: float = 0.380,
        y_offset: float = 0.128,
        z_offset: float = 0.128,
        x_margin: float = 0.05,
        y_margin: float = 0.05,
        z_margin: float = 0.05,
    ):
        """
        Source represents a directional sound source.

        horiz_disp and vert_disp map dispersions angles in degrees to gain at that
        angle relative to the main acoustic axis in decibels.
        """
        self._h_x = np.array(list(horiz_disp.keys()), np.float32)
        self._h_y = np.array(list(horiz_disp.values()), np.float32)
        self._v_x = np.array(list(vert_disp.keys()), np.float32)
        self._v_y = np.array(list(vert_disp.values()), np.float32)

        self._x_dim = x_dim
        self._y_dim = y_dim
        self._z_dim = z_dim
        self._y_offset = y_offset
        self._z_offset = z_offset

    def gain(self, vert_angle: float, horiz_angle: float) -> float:
        """
        Returns the gain of the source at the given angle in decibels.

        Angles in degrees.
        """
        val = np.interp(abs(vert_angle), self._v_x, self._v_y) + np.interp(
            abs(horiz_angle), self._h_x, self._h_y
        )
        if not isinstance(val, float):
            raise RuntimeError
        return val

    def test_intersection(
        self, placement: npt.NDArray, norm: npt.NDArray, test_point: npt.NDArray
    ) -> bool:
        """work in progress"""
        box = trimesh.primitives.Box(np.array([self._x_dim, self._y_dim, self._z_dim]))
        translation = trimesh.transformations.translation_matrix(
            placement - np.array([0, self._y_offset, self._z_offset])
        )
        rotation = geometry.rotation_matrix(np.array([0, 0, 0]), norm)
        return (
            box.apply_transform(translation)
            .apply_transform(rotation)
            .contains(test_point)[0]
        )
