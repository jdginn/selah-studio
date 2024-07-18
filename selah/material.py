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
        return self._absorption

    def scattering(self, freq: float = 1000) -> float:
        return self._scattering

    def diffusion(self, freq: float = 1000) -> float:
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
        if materials is None:
            self._materials = default_materials
        else:
            self._materials = materials
        self._wall_materials = {"default": "brick"}

    def set_wall_materials(self, wall_materials: typing.Dict[str, str]):
        if "default" not in wall_materials:
            raise MaterialException(
                "Wall materials dict must specify a default material"
            )
        for name, mat in wall_materials:
            if mat not in self._materials:
                raise MaterialException(
                    f"Wall {name} specifies material {mat}, which is missing from materials dict: {self._materials}"
                )
        self._wall_materials = wall_materials

    def get_wall(self, name: str) -> Material:
        material_name = self._wall_materials.get(name, "default")
        return self._materials[material_name]
