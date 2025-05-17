import os
import json
import uproot
import logging
import subprocess
from tqdm import tqdm
from numexpr import evaluate as ne_evaluate
from pathlib import Path
from collections import defaultdict

from .xsec import PMGXsec
from .sum_weights import SumWeightTool, _extract_cutbook_sum_weights
from .file_metadata import FileMetaData
from ..histo.tools import get_expression_variables

up_open = uproot.open
up_new = uproot.recreate


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SkimConfig:

    def __init__(self):
        self.do_sum_of_weights: bool = False
        self.remove_combined: bool = True
        self.combine_branches: dict[str, list[str]] = {}
        self.filter: str | None = None
        self.output_directory: str | None = None
        self.file_groups: dict[int, tuple[str, str]] = {}
        self.file_campaign_map: dict[str, dict[int, list[tuple[FileMetaData, str]]]] = (
            {}
        )
        self.sum_weights_file: str | None = None
        self.xsec_file: str | None = None
        self.reco_tree_name: str = "reco"

    def load_from_json(self, json_setting: str | Path | dict):

        if isinstance(json_setting, (Path, str)):
            with open(json_setting) as f:
                setting = json.load(f)
        else:
            setting = json_setting

        self.remove_combined = setting["remove_combined"]
        self.combine_branches = setting["combine_branches"]
        self.filter = setting["filter"]
        self.output_directory = setting["output_directory"]
        self.sum_weights_file = setting.get("sum_weights_file", None)
        self.xsec_file = setting.get("xsec_file", None)
        self.file_groups = {}

        with open(setting["file_groups"]) as name_grp_file:
            for line in name_grp_file.readlines():
                dsid, short_name, long_name = line.split()
                self.file_groups[int(dsid)] = short_name, long_name

    def prepare_files(self, file_list: list[str]):
        os.makedirs(self.output_directory, exist_ok=True)
        with open(f"{self.output_directory}/missing_files.txt", "w") as missing_files:
            for filename in file_list:
                try:
                    metadata = FileMetaData(filename)
                except ValueError:
                    logger.debug(f"Cannot find metadata for {filename}")
                    missing_files.write(f"{filename}\n")
                    continue
                campaign = metadata.campaign
                dsid = metadata.dataset_id
                if campaign not in self.file_campaign_map:
                    self.file_campaign_map[campaign] = {}
                if metadata.dataset_id not in self.file_campaign_map[campaign]:
                    self.file_campaign_map[campaign][dsid] = []
                self.file_campaign_map[campaign][dsid].append((metadata, filename))

        # Show list of campaigns
        logger.info(f"Found campaign {list(self.file_campaign_map)}")

    def get_list_of_files(self) -> list[str]:
        for campaign in self.file_campaign_map:
            for dsid in self.file_campaign_map[campaign]:
                for _, filename in self.file_campaign_map[campaign][dsid]:
                    yield filename


class EventSkimmer:

    def __init__(self, config):
        self.config = config
        self.sum_weight_tool = None
        self.xsec_tool = None
        self.do_sum_of_weights = False

    def run_skimming(self, show_progress=True):

        # Check for sum weights
        if self.config.combine_branches:
            self.do_sum_of_weights = True

        # Check for sum weights file
        if self.do_sum_of_weights:
            sum_weight_file = (
                self.config.sum_weights_file
                or f"{self.config.output_directory}/SumWeights.txt"
            )
            if Path(sum_weight_file).exists():
                self.sum_weight_tool = SumWeightTool(sum_weight_file)
            else:
                sum_weights_file = _extract_cutbook_sum_weights(
                    self.config.get_list_of_files(), sum_weight_file
                )
                self.sum_weight_tool = SumWeightTool(sum_weights_file)

        # Check for xsec file
        if self.config.xsec_file:
            self.xsec_tool = PMGXsec(self.config.xsec_file)
        else:
            self.xsec_tool = lambda event: 1

        output_keeper: dict[tuple(str, str), list[str]] = defaultdict(list)

        # Skimming
        file_groups = self.config.file_groups
        skim_event = self.skim_event
        file_campaign_map = self.config.file_campaign_map
        for campaign in file_campaign_map:
            for dsid in file_campaign_map[campaign]:
                if dsid not in file_groups and dsid != 0:
                    logger.warning(f"Cannot find DSID {dsid}")
                    continue
                filelist = file_campaign_map[campaign][dsid]
                short_name = "data" if dsid == 0 else file_groups[dsid][0]
                if show_progress:
                    filelist = tqdm(filelist, leave=False)
                for metadata, filename in filelist:
                    output_keeper[(campaign, short_name)].append(skim_event(filename))

        return output_keeper

    def merge_output(self, output_keeper):

        # Merge output keeper if it is a list
        if isinstance(output_keeper, list):
            first_output = output_keeper[0]
            for next_output in output_keeper[1:]:
                for (campaign, short_name), output_files in next_output.items():
                    first_output[(campaign, short_name)].extend(output_files)
            output_keeper = first_output

        # Merging output files with hadd, if hadd fail, set symlink
        for (campaign, short_name), output_files in output_keeper.items():
            output_filename = f"{self.config.output_directory}/{short_name}_{campaign}"
            try:
                subprocess.run(
                    ["hadd", "-f", output_filename + ".root"] + output_files,
                    check=True,
                )
            except subprocess.CalledProcessError:
                logger.warning(f"hadd failed, set symlink for {output_filename}")
                os.makedirs(output_filename, exist_ok=True)
                for src_file in output_files:
                    try:
                        os.symlink(src_file, f"{output_filename}/{Path(src_file).name}")
                    except FileExistsError:
                        pass

    def skim_event(self, filename):
        filename = Path(filename)
        output_filename = f"{self.config.output_directory}/tmp/{filename.name}"
        output_filename = f"{output_filename.removesuffix('.root')}.skimmed.root"

        # Check if output file already exists
        if Path(output_filename).exists():
            return output_filename

        with up_open(filename) as i_file, up_new(output_filename) as o_file:
            # Copy all non tree objects to output file.
            for name, item in i_file.items():
                if not isinstance(item, uproot.TTree):
                    o_file[name] = item

            if self.config.reco_tree_name not in i_file:
                return output_filename

            combined_branches = self.config.combine_branches
            filter_str = self.config.filter

            reco_tree_name = self.config.reco_tree_name

            # replace syst tag
            combined_branches = {
                k.replace("_SYS_", "_NOSYS"): v.replace("_SYS_", "_NOSYS")
                for k, v in combined_branches.items()
            }
            filter_str = filter_str.replace("_SYS_", "_NOSYS")

            drop_branches = {}
            for name, combined in combined_branches.items():
                drop_branches[name] = get_expression_variables(combined)

            i_tree = i_file[reco_tree_name]

            branch_filter = [x for x in i_tree.keys() if "NOSYS" in x]

            # Include those extra branches that are systematic independent
            extra_branches = [
                "weight_beamspot",
                "pass_el_trig",
                "pass_mu_trig",
                "mcChannelNumber",
                "eventNumber",
                "actualInteractionsPerCrossing",
                "averageInteractionsPerCrossing",
                "runNumber",
            ]

            # Check if branches are in the tree
            extra_branches = [x for x in extra_branches if x in i_tree]
            branch_filter.extend(extra_branches)

            for name in branch_filter:
                if name not in i_tree:
                    branch_filter.remove(name)

            remove_combined_branches = self.config.remove_combined

            first_chunk = True

            for events in i_tree.iterate(filter_name=branch_filter, library="np"):

                for name, combined in combined_branches.items():
                    try:
                        events[name] = ne_evaluate(combined, events)
                    except KeyError:  # okay to just ignore unavailable branches
                        pass

                mask = ne_evaluate(filter_str, events)
                for name in events:
                    events[name] = events[name][mask]

                # Removing combined branches
                if remove_combined_branches:
                    for drop_list in drop_branches.values():
                        for drop_name in drop_list:
                            if drop_name not in events:
                                continue
                            del events[drop_name]

                if first_chunk:
                    o_file[reco_tree_name] = events
                    first_chunk = False
                else:
                    o_file[reco_tree_name].extend(events)

        return output_filename
