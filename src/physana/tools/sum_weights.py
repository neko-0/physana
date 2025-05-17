import numpy as np
from numexpr import evaluate as ne_eval
import uproot
from tqdm import tqdm

from ..serialization.base import to_json, from_json


class SumWeightTool:

    def __init__(self, file=None):
        self.dsid_branch = "mcChannelNumber"
        self.run_number_branch = "runNumber"
        self.is_loaded = False

        # This sum of weights dict is map with
        # {systematics : {(dsid, run_number) : sum_weights}}
        self.sum_weights_dict = {}

        # sum of weights file name
        self.sum_weights_file = file
        if file:
            self.load_sum_weights()

        # state variables
        self.is_data = False
        self.syst = "NOSYS"

    def __call__(self, events):
        return self.get_sum_weights(events)

    def load_state_from_process(self, process):
        if process.is_data:
            self.is_data = True
            return
        self.is_data = False
        if process.systematics:
            self.syst = process.systematics.tag
        else:
            self.syst = "NOSYS"

    def load_sum_weights(self):
        if self.is_loaded:
            return
        if self.sum_weights_file is None:
            return
        json_data = from_json(self.sum_weights_file)
        self.dsid_branch = json_data["dsid_branch"]
        self.run_number_branch = json_data["run_number_branch"]
        # JSON cannot use tuple as key
        stored_sum_w = json_data["sum_weights"]
        self.sum_weights_dict = {}
        for syst, syst_w in stored_sum_w.items():
            self.sum_weights_dict[syst] = {}
            for dsid, run in syst_w.items():
                for run_num, sum_w in run.items():
                    self.sum_weights_dict[syst][(int(dsid), int(run_num))] = sum_w
        self.is_loaded = True

    def save(self, file):
        output_sum_w = {}
        for syst, syst_w in self.sum_weights_dict.items():
            output_sum_w[syst] = {}
            for lookup, sum_w in syst_w.items():
                dsid, run_number = lookup
                if dsid not in output_sum_w[syst]:
                    output_sum_w[syst][dsid] = {}
                if run_number not in output_sum_w[syst][dsid]:
                    output_sum_w[syst][dsid][run_number] = 0.0
                output_sum_w[syst][dsid][run_number] += sum_w
        json_data = {
            "dsid_branch": self.dsid_branch,
            "run_number_branch": self.run_number_branch,
            "sum_weights": output_sum_w,
        }
        return to_json(json_data, file)

    def get_sum_weights(self, events):

        if self.is_data or not self.is_loaded:
            return 1.0

        dsid = ne_eval(self.dsid_branch, events)
        run_number = ne_eval(self.run_number_branch, events)

        unique_pairs = np.unique((dsid, run_number), axis=1)
        idx = np.searchsorted(unique_pairs[0], dsid)
        idx_run = np.searchsorted(unique_pairs[1], run_number)

        # sum_weights_dict = self.sum_weights_dict[self.syst]
        w_get = self.sum_weights_dict[self.syst].get

        w_values = np.array([w_get(key) for key in zip(dsid, run_number)])

        return np.take(
            w_values.ravel(),
            np.ravel_multi_index(
                (idx, idx_run), (len(unique_pairs[0]), len(unique_pairs[1]))
            ),
        )

    def get_sum_weight(self, dsid, run_number):
        if self.is_data or not self.is_loaded:
            return 1.0
        return self.sum_weights_dict[self.syst][(dsid, run_number)]


def _extract_cutbook_sum_weights(
    list_of_files, output, dsid_branch="mcChannelNumber", run_number_branch="runNumber"
):
    sum_weights = {}

    for file in tqdm(list_of_files):
        with uproot.open(file) as root_file:
            for obj_name in root_file.keys():
                if "CutBookkeeper" not in obj_name:
                    continue

                _, dsid, run, syst = obj_name.split("_")

                syst = syst.split(";")[0]

                if syst not in sum_weights:
                    sum_weights[syst] = {}

                lookup = (dsid, run)
                if lookup not in sum_weights[syst]:
                    sum_weights[syst][lookup] = 0.0

                sum_weights[syst][lookup] += float(root_file[obj_name].values()[1])

    sum_weight_tool = SumWeightTool()
    sum_weight_tool.dsid_branch = dsid_branch
    sum_weight_tool.run_number_branch = run_number_branch
    sum_weight_tool.sum_weights_dict = sum_weights

    return sum_weight_tool.save(output)


def extract_cutbook_sum_weights(config, *args, **kwargs):
    def ntuple_files():
        for pset in config.process_sets:
            for p in pset:
                for file in p.input_files:
                    yield file

    return _extract_cutbook_sum_weights(ntuple_files(), *args, **kwargs)
