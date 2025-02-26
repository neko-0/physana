import json
from pathlib import Path

from .base import SerializationBase


class SerialXSecFile(SerializationBase):
    _data_name = {
        "xsec_file",
        "lumi",
        "dsid",
        "xsec_name",
        "sumw_name",
        "nominal_token",
        "token_groups",
        "token_groups_rule",
        "campaign_sensitive",
        "campaign_files",
        "campaign_lumi",
        "campaign_xsec",
        "do_check_process",
        "check_map",
        "weight_base_token",
        "use_process_nominal_tree",
        "duplicated_list",
        "duplicated_sets",
        "duplicated_skip_campaign",
        "duplicated_accept",
        "remove_wsyst_ptag",
    }

    def to_json(self, xsec, fname):
        data = {}
        for name in self._data_name:
            data[name] = getattr(xsec, name)
        data["expr"] = xsec.expr()
        fname = Path(fname)
        fname.parent.mkdir(parents=True, exist_ok=True)
        with open(str(fname), "w") as f:
            json.dump(data, f)

        return fname

    def from_json(self, fname):
        with open(fname) as f:
            return json.load(f)
