import typing

import numpy as np
import numpy.typing as npt
from tmm.tmm import TMM

from .exceptions import SelahException


class MaterialException(SelahException):
    pass


numeric = typing.Union[int, float, np.number]


class Material:
    # TODO: accept float, dict[freq, abs], or func for args
    def __init__(
        self,
        absorption: typing.Tuple[typing.List[numeric], typing.List[numeric]] = (
            [1000],
            [0],
        ),
        scattering: float = 0,
        diffusion: float = 0,
    ):
        (self._abs_freq, self._abs_alpha) = absorption
        self._scattering = scattering
        self._diffusion = diffusion

    def absorption(self, freq: float = 1000) -> float:
        """Returns the absorption coefficient for the passed frequency"""
        a = np.interp(freq, np.array(self._abs_freq), np.array(self._abs_alpha))
        if a == 0:
            # Absorption 0 breaks the universe
            return 0.001
        if isinstance(a, np.ndarray):
            a = a[0]
        return a

    def scattering(self, freq: float = 1000) -> float:
        """Returns the scattering coefficient for the passed frequency"""
        return self._scattering

    def diffusion(self, freq: float = 1000) -> float:
        """Returns the diffusion coefficient for the passed frequency"""
        return self._diffusion


rockwool_12cm = TMM(fmin=20, fmax=10_000, df=5)
rockwool_12cm.porous_layer(sigma=12, t=120)
rockwool_12cm.compute()

rockwool_24cm = TMM(fmin=20, fmax=10_000, df=5)
rockwool_24cm.porous_layer(sigma=12, t=240)
rockwool_24cm.compute()

rockwool_30cm = TMM(fmin=20, fmax=10_000, df=5)
rockwool_30cm.porous_layer(sigma=12, t=300)
rockwool_30cm.compute()


default_materials: typing.Dict[str, Material] = {
    "brick": Material(
        ([125, 250, 500, 1000, 2000, 4000], [0.3, 0.4, 0.3, 0.3, 0.4, 0.3])
    ),
    "gypsum": Material(
        ([125, 250, 500, 1000, 2000, 4000], [0.01, 0.02, 0.02, 0.03, 0.04, 0.05])
    ),
    "diffuser": Material(([1000], [0.8]), scattering=0.5, diffusion=0.95),
    "wood": Material(
        ([125, 250, 500, 1000, 2000, 4000], [0.04, 0.04, 0.07, 0.06, 0.06, 0.07])
    ),
    "12cm_rockwool": Material((rockwool_12cm.freq, rockwool_12cm.alpha)),
    "24cm_rockwool": Material((rockwool_24cm.freq, rockwool_24cm.alpha)),
    "30cm_rockwool": Material((rockwool_30cm.freq, rockwool_30cm.alpha)),
    "glass": Material(
        ([125, 250, 500, 1000, 2000, 4000], [0.18, 0.06, 0.04, 0.03, 0.02, 0.02])
    ),
}


class MaterialManager:
    def __init__(self, materials: typing.Dict[str, Material] | None = None):
        """
        Assigns materials by wall name, respecting defaults

        Optional materials argument maps materials by name to their properties. If
        no dict is passed, a builtin default materials dict is used.
        """

        if materials is None:
            self._materials = default_materials
        else:
            self._materials = materials
        self._wall_materials = {"default": "brick"}

    def set_wall_materials(self, wall_materials: typing.Dict[str, str]):
        """
        Sets the material to be used for each wall.

        wall_materials dict must contain a "default" field, which will be
        used for any wall not explicitly set in the dict.
        """
        if "default" not in wall_materials:
            raise MaterialException(
                "Wall materials dict must specify a default material"
            )
        for name, mat in wall_materials.items():
            if mat not in self._materials:
                raise MaterialException(
                    f"Wall {name} specifies material {mat}, which is missing from materials dict: {self._materials}"
                )
        self._wall_materials = wall_materials

    def get_wall(self, name: str) -> Material:
        """
        Returns the material of the requested wall. If the requested wall
        is not found in the manager, returns the default material.
        """
        if name not in self._wall_materials.keys():
            name = "default"
        material_name = self._wall_materials[name]
        return self._materials[material_name]
