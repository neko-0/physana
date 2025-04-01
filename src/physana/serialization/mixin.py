from typing import Dict, Any

from .base import SerializationBase
from .histo import SerialHistogram


class SerialProcessSet(SerializationBase):
    def to_dict(self, pset: Any) -> Dict[str, Any]:
        process_s = SerialProcess().to_dict
        pset_dict = {
            "name": pset.name,
            "nominal": process_s(pset.nominal) if pset.nominal else None,
        }
        pset_dict["computed_systematics"] = {
            x: process_s(y) for x, y in pset.computed_systematics.items()
        }
        pset_dict["systematics"] = [process_s(x) for x in pset.systematics]

        return pset_dict


class SerialProcess(SerializationBase):
    def to_dict(self, process: Any) -> Dict[str, Any]:
        region_s = SerialRegion().to_dict
        p_dict = {
            "name": process.name,
            "treename": process.name,
            "selection": process.selection,
            "weights": process.weights,
        }
        p_dict["regions"] = {r.name: region_s(r) for r in process}
        syst = process.systematic
        if syst:
            p_dict["systematic"] = {
                "name": syst.name,
                "full_name": syst.full_name,
                "source": syst.source,
                "sys_type": syst.sys_type,
                "handle": syst.handle,
                "symmetrize": syst.symmetrize,
            }
        else:
            p_dict["systematic"] = None

        return p_dict


class SerialRegion(SerializationBase):
    def to_dict(self, region: Any) -> Dict[str, Any]:
        hist_s = SerialHistogram()
        r_dict = {
            "name": region.name,
            "weights": region.weights,
            "selection": region.selection,
        }
        r_dict["histograms"] = {
            h.name: hist_s.to_dict(h) for h in region if h.hist_type != "2d"
        }  # currently don't support 2D

        return r_dict
