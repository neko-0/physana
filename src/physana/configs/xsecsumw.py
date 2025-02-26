import json
import re

from ..serialization import Serialization


class XSecSumEvtW:
    __slots__ = (
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
        "_xsec",
        "_is_data",
        "_cached_w",
        "_re_expr",
        "_c_re",
    )

    def __init__(self, xsec_file=None):
        self.xsec_file = xsec_file
        self.lumi = 1.0
        self.dsid = "DatasetNumber"
        self.xsec_name = "xsec"
        self.sumw_name = "AllExecutedEvents_sumOfEventWeights"
        self.nominal_token = "weight_1"
        self.token_groups = {}
        self.token_groups_rule = {
            "path": 1,
            "campaign": 2,
            "prefix": 2,
            "user": 5,
            "process": 6,
            "dsid": 7,
            "ptag": 8,
            "user-tag": 9,
            "syst": 10,
            "suffix": 11,
        }
        self.campaign_sensitive = False
        self.campaign_files = {}
        self.campaign_lumi = {}
        self.campaign_xsec = None
        self.do_check_process = False
        self.check_map = {}
        self.weight_base_token = None
        self.use_process_nominal_tree = False
        self.duplicated_list = ["LHE3Weight", "pileupWeight", "triggerWeight"]
        self.duplicated_sets = None  # e.g. {'triggerWeight' : 'weight_2'}
        self.duplicated_skip_campaign = None  # e.g. {'e' : 'pileupWeight'}
        self.duplicated_accept = None  # e.g. {'weight_2', 'kinematic_1'}
        # self.remove_wsyst_ptag has the form e.g. {'zjets_2211':{'d':{'bTagWeight' : 'p4512'}}}
        self.remove_wsyst_ptag = None
        self._xsec = None
        self._is_data = False
        self._cached_w = None
        # default regular expression
        self._re_expr = "(.*)mc16([ade])(.*)(user).(.\\w*).(.\\w*).(.\\d*).(.*).(CollinearW.SMA.v1)_(.*)_(t2_tree.root)"
        self._c_re = re.compile(self._re_expr)

    def __getitem__(self, x):
        return self.token_groups[x]

    def keys(self):
        return self.token_groups.keys()

    def items(self):
        return self.token_groups.items()

    def nominal(self, fname):
        if self.nominal_token not in fname:
            return False
        else:
            return True

    def check_syst_set(self, fname, syst_set=None):
        if syst_set is None:
            return self.nominal(fname)
        else:
            return syst_set in fname

    def set_campaign_files(self, values):
        self.campaign_files = values

    def set_campaign_lumi(self, lumi):
        self.campaign_lumi = lumi

    def check_process(self, process: str):
        if not self.do_check_process:
            return True
        current_process = "data" if self._is_data else self["process"]
        return self.check_map.get(process, process) == current_process

    def get_xsec_sumw(self, dsid=None, syst=None):
        if self._is_data:
            return 1.0
        if self.campaign_sensitive:
            return self.get_campaign_xsec_sumw()
        if self._xsec is None:
            if self.xsec_file is None:
                raise IOError("no xsec file specified.")
            with open(self.xsec_file) as f:
                self._xsec = json.load(f)
        if dsid is None:
            dsid = self["dsid"]
        if syst is None:
            syst = self["syst"]
        dataset = self._xsec[dsid][syst]
        return dataset[self.xsec_name] / dataset[self.sumw_name] * self.lumi

    def get_campaign_xsec_sumw(self):
        if self._is_data:
            return 1.0
        if self.campaign_xsec is None:
            self.campaign_xsec = {}
            for mc, fname in self.campaign_files.items():
                with open(fname) as f:
                    self.campaign_xsec[mc] = json.load(f)
        current_campaign = self["campaign"]
        dsid = self["dsid"]
        syst = self["syst"]
        lumi = self.campaign_lumi.get(current_campaign, 1.0)
        dataset = self.campaign_xsec[current_campaign][dsid][syst]
        return dataset[self.xsec_name] / dataset[self.sumw_name] * lumi

    def get_auto(self):
        if self._cached_w is None:
            self._cached_w = self.get_xsec_sumw()
        return self._cached_w

    def event_weight(self, *args, **kwargs):
        return self.get_xsec_sumw(*args, **kwargs)

    def set_expression(self, expr):
        self._re_expr = expr
        self._c_re = re.compile(expr)

    def expr(self):
        return self._re_expr

    def match(self, fname):
        tokens = self._c_re.match(fname)
        if self.token_groups:
            self.token_groups.update(
                {x: tokens.group(y) for x, y in self.token_groups_rule.items()}
            )
        else:
            self.token_groups = {
                x: tokens.group(y) for x, y in self.token_groups_rule.items()
            }
        self._is_data = "data" in self["process"]
        self._cached_w = None

    @classmethod
    def open(cls, xsecf):
        obj = cls()
        m_serial = Serialization("xsec")
        json_data = m_serial.from_json(xsecf)
        obj.set_expression(json_data.pop("expr"))
        for name, value in json_data.items():
            setattr(obj, name, value)
        return obj

    def save(self, fname):
        m_serial = Serialization("xsec")
        return m_serial.to_json(self, fname)
