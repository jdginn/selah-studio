import typing

from .exceptions import SelahException


class MaterialException(SelahException):
    pass


class Material:
    # TODO: accept float, dict[freq, abs], or func for args
    def __init__(self, absorption: float, scattering: float = 0, diffusion: float = 0):
        self._absorption = absorption
        self._scattering = scattering
        self._diffusion = diffusion

    def absorption(self, freq: float = 1000) -> float:
        """Returns the absorption coefficient for the passed frequency"""
        return self._absorption

    def scattering(self, freq: float = 1000) -> float:
        """Returns the scattering coefficient for the passed frequency"""
        return self._scattering

    def diffusion(self, freq: float = 1000) -> float:
        """Returns the diffusion coefficient for the passed frequency"""
        return self._diffusion


default_materials: typing.Dict[str, Material] = {
    "brick": Material(0.04),
    "gypsum": Material(0.05),
    "diffuser": Material(0.1, scattering=0.5, diffusion=0.95),
    "wood": Material(0.1),
    "12cm_rockwool": Material(0.95),
    "24cm_rockwool": Material(0.95),
    "30cm_rockwool": Material(0.95),
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
