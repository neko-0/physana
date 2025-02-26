import sys
from copy import deepcopy

import numpy as np


class SystematicsBand:
    def __init__(self, name, type, shape):
        self.name = name
        self.type = type
        self.shape = shape
        # The _components_up/down hold every single components
        self._components_up = {}
        self._components_down = {}
        # the _sub_bands is a sub-structure that contains other
        # SystematicsBand object, it's compoents are all covered in _up/_down
        # it's convinent for exploring a group of components.
        self._sub_bands = {}
        self._up = None
        self._down = None
        self._cache_total = False

    def __sizeof__(self):
        size = 0
        for comp in self.components.values():
            value = next(iter(comp.values()))
            size += sys.getsizeof(value)
            size *= len(comp)
        for sub_band in self._sub_bands.values():
            size += sys.getsizeof(sub_band)
        return size

    def __getitem__(self, band_type):
        return self.get_band(band_type)

    def __add__(self, rhs):
        new_obj = SystematicsBand(self.name, self.type, self.shape)
        new_obj.combine(self)
        new_obj.combine(rhs)
        return new_obj

    def combine(self, other):
        """
        combining with other systematic band in quadrature.

        Args:
            other : SystematicsBand
                SystematicsBand object for combining.

        Return:
            no return
        """

        new_components = {"up": {}, "down": {}}
        # combining same components first
        for side in ["up", "down"]:
            self_components = getattr(self, f"_components_{side}")
            other_components = getattr(other, f"_components_{side}")
            for name, comp in other_components.items():
                if name in self_components:
                    self_components[name] = np.sqrt(
                        self_components[name] ** 2 + comp**2
                    )
                else:
                    new_components[side].update({name: deepcopy(comp)})

        # combining sub bands.
        for name, sub_band in other._sub_bands.items():
            if name not in self._sub_bands:
                self.update_sub_bands(sub_band)

        # check if any missing top level components after sub-band merged.
        for side in ["up", "down"]:
            self_components = getattr(self, f"_components_{side}")
            for name, comp in new_components[side].items():
                if name not in self_components:
                    self_components.update({name: comp})

    def add_component(self, type, name, band: np.array):
        if type not in ["up", "down"]:
            raise ValueError(f"Invalid {type}. Use up/down")
        assert band.shape == self.shape
        getattr(self, f"_components_{type}")[name] = band

    def update(self, other, copy=True):
        """
        Update band information from another band
        """
        other = deepcopy(other) if copy else other
        for sub_band in other._sub_bands.values():
            self.update_sub_bands(sub_band, copy=False)
        self.update_components(other, copy=False)

    def update_sub_bands(self, band, exist_ok=True, copy=True):
        """
        method for update sub-band structure from another band.

        Args:
            band : SystematicsBand
                a SystematicsBand instance to be stored in sub-bands dict.

            exist_ok : bool, default=False
                if it's True, then just updating/overwrite
                sub-band components name collides.
                if it's False, expand sub-band name into it's components
        """
        if band.name not in self._sub_bands:
            self._sub_bands[band.name] = deepcopy(band) if copy else band
        else:
            self._sub_bands[band.name].update_components(band, exist_ok, copy)
        # update the top level components from it's sub-band components
        # no need to make copy here
        for sub_band in self._sub_bands.values():
            self.update_components(sub_band, exist_ok, copy=False)

    def update_components(self, band, exist_ok=True, copy=True):
        if exist_ok:
            _up = band._components_up
            _down = band._components_down
        else:
            bname = band.name
            _up = {f"{bname}/{x}": y for x, y in band._components_up.items()}
            _down = {f"{bname}/{x}": y for x, y in band._components_down.items()}
        if copy:
            _up = deepcopy(_up)
            _down = deepcopy(_down)
        self._components_up.update(_up)
        self._components_down.update(_down)

    def get_band(self, type):
        _band = np.zeros(self.shape)
        for component in getattr(self, f"_components_{type}").values():
            _band += component * component
        return np.sqrt(_band)

    def remove_sub_band(self, name):
        band = self._sub_bands[name]
        for key in band._components_up.keys():
            del self._components_up[key]
        for key in band._components_down.keys():
            del self._components_down[key]
        del self._sub_bands[name]

    @property
    def up(self):
        """
        return the total up band
        """
        if not self._cache_total or self._up is None:
            self._up = self.get_band("up")
        return self._up

    @up.setter
    def up(self, value):
        self._cache_total = True
        self._up = value

    @property
    def down(self):
        """
        return the total down band
        """
        if not self._cache_total or self._down is None:
            self._down = self.get_band("down")
        return self._down

    @down.setter
    def down(self, value):
        self._cache_total = True
        self._down = value

    @property
    def components(self):
        return {"up": self._components_up, "down": self._components_down}

    @property
    def sub_bands(self):
        return self._sub_bands

    def use_cache_total(self, value):
        self._cache_total = value
        if not value:
            self._up = None
            self._down = None

    def scale_nominal(self, nominal: np.array):
        """
        convert ratio band to actual band of difference with respect to bin content
        """
        _up = self.get_band("up") * nominal
        _down = self.get_band("down") * nominal
        return {"up": _up, "down": _down}

    def scale_components(self, scale):
        for value in self._components_up.values():
            value *= scale
        for value in self._components_down.values():
            value *= scale

    def list_sub_bands(self):
        return set(self._sub_bands.keys())

    def flatten(self):
        """
        flatten the band structure and dump it into dict format
        """
        output = {}
        for name in {"name", "type", "shape"}:
            output[name] = getattr(self, name)
        output["components"] = self.components
        return output

    def average(self):
        """
        return the average of the up+down band
        """
        return (self.up + self.down) / 2.0

    def component_names(self):
        return self._components_up.keys()

    def components_as_bands(self, filter_zero=True):
        comp_names = self._components_up.keys()
        for name in comp_names:
            _up = self._components_up[name]
            _dn = self._components_down[name]
            # if filter_zero and np.all(_up == 0.0) and np.all(_dn == 0.0):
            #     continue
            if filter_zero and np.all(_up == 0.0):
                _up[_up == 0.0] = 1e-9
            if filter_zero and np.all(_dn == 0.0):
                _dn[_dn == 0.0] = 1e-9
            if filter_zero and np.abs(np.sum(_up)) < 1e-5:
                _up[_up == 0.0] = 1e-9
            if filter_zero and np.abs(np.sum(_dn)) < 1e-5:
                _dn[_dn == 0.0] = 1e-9
            comp_band = SystematicsBand(name, self.type, self.shape)
            comp_band.add_component("up", name, _up)
            comp_band.add_component("down", name, _dn)
            yield comp_band

    def clear(self):
        self._components_up = {}
        self._components_down = {}
        self._sub_bands = {}
        self._up = None
        self._down = None
        self._cache_total = False

    @classmethod
    def loads(cls, band_data):
        """
        loading flatten data from the `flatten` method output
        """
        band = cls(band_data["name"], band_data["type"], band_data["shape"])
        band._components_up.update(band_data["components"]["up"])
        band._components_down.update(band_data["components"]["down"])
        return band
