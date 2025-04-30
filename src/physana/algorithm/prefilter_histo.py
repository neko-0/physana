from collections import defaultdict
from typing import Dict, List, Tuple, TYPE_CHECKING
import logging

from .histmaker import HistMaker
from .sum_weights import SumWeightTool
from ..tools.xsec import PMGXsec
from ..histo.histo1d import from_uproot_histo
from ..histo.region import Region

if TYPE_CHECKING:
    from ..config import ConfigMgr
    from ..histo.histo1d import Histogram

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PrefilterHistReader(HistMaker):

    def process(self, config: "ConfigMgr") -> None:

        processes = (p for pset in config.process_sets for p in pset)

        self.sum_weight_tool = SumWeightTool(self.sum_weights_file)
        self.xsec_tool = PMGXsec(self.xsec_file)

        get_prefilt_histo = self.parse_prefilter_histogram

        for proc in processes:

            self.sum_weight_tool.load_state_from_process(proc)

            empty_proc = proc.copy(shallow=True)
            empty_proc.clear()

            nfiles = len(proc.input_files)

            for i, fname in enumerate(proc.input_files):
                tmp_proc = empty_proc.copy()
                hist_map = get_prefilt_histo(fname, proc.is_data)
                if not hist_map:
                    continue
                # only handle NOSYS
                regions_setting = hist_map["NOSYS"]
                region_map = {}
                for name, hist_name, hist in regions_setting:
                    if name not in region_map:
                        region_map[name] = Region(name, "", "")
                    region_map[name].append(hist)

                for region in region_map.values():
                    tmp_proc.append(region)
                proc.add(tmp_proc)

                if i % 20 == 0:
                    logger.info(f"Processed {i}/{nfiles} files.")

    def parse_prefilter_histogram(
        self, fname: str, is_data=False
    ) -> Dict[str, List[Tuple[str, str, "Histogram"]]]:
        open_file = self.open_file

        with open_file(fname) as tfile:
            # get list of object names from Tfile
            # the histogram name is of the form
            # VjjHist__sel_<selecion_name>__<object_name>_<syst_name>
            hists = [(x, y) for x, y in tfile.items() if x.startswith("VjjHist__sel_")]

            hists_syst_map = defaultdict(list)
            for name, hist in hists:
                sel_name, hist_syst = name.removeprefix("VjjHist__").split("__")
                hist_name, syst = hist_syst.split("_")
                syst = syst.split(";")[0]  # removing cyclye e.g. "NOSYS;1"
                parsed_hist = from_uproot_histo(hist)
                parsed_hist.name = hist_name
                parsed_hist.xtitle = hist_name
                hists_syst_map[syst].append((sel_name, hist_name, parsed_hist))

            # return histogram map if data, no need to scale xsec and sum of weights
            if is_data:
                return hists_syst_map

            # get cutbookkeeper for DSID and run number
            # the cutbookkeeper name is of the form
            # CutBookkeeper_<dsid>_<run_number>_<syst_name>
            cutbookkeepers = [x for x in tfile.keys() if x.startswith("CutBookkeeper")]

            get_sum_weight = self.sum_weight_tool.get_sum_weight
            get_xsec = self.xsec_tool.get_xsec
            # create a map of cutbookkeeper syst to sum of weights and xsec
            cutbook_syst_map: Dict[str, Tuple[float, float]] = {}
            for cutbookkeeper in cutbookkeepers:
                _, dsid, run, syst = cutbookkeeper.split("_")
                dsid, run = int(dsid), int(run)
                syst = syst.split(";")[0]  # removing cyclye e.g. "NOSYS;1"
                cutbook_syst_map[syst] = (get_sum_weight(dsid, run), get_xsec(dsid))

            # scale histograms by sum of weights and cross section
            for syst, hist_list in hists_syst_map.items():
                for _, _, hist in hist_list:
                    sum_weight, xsec = cutbook_syst_map[syst]
                    hist.mul(xsec / sum_weight)

            return hists_syst_map
