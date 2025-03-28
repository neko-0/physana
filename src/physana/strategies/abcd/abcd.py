"""
Contains methods for serserving and creating ABCD regions
Also provide somes helper methods to retrieve the meta data of ABCD
for later use. See fake estimation in histManipulate.py
"""

import scipy.optimize
import numpy
import time
import os
import concurrent.futures
import logging
import functools
import pathlib
from copy import deepcopy
from collections import defaultdict

from ...histo import Process, Region, Histogram, Histogram2D
from ...histo.tools import Filter
from ...systematics import SystematicsBase, Systematics
from ...configs import ConfigMgr
from ...algorithm import HistMaker
from ...algorithm.histmaker import weight_from_hist
from ...algorithm.interface import refill_process
from ...serialization import Serialization
from ...plotMaker import PlotMaker

# import fnmatch
import tqdm

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)
usr_log = logging.getLogger(f"{__name__}_USER")
usr_log.setLevel(logging.INFO)

CPU_CORE = int(numpy.ceil(0.5 * os.cpu_count()))


class ABCDSel(object):
    """
    class for ABCD selections
    """

    __slots__ = ("tag", "selection", "axis")

    def __init__(self, tag=None, selection=None, axis=None):
        self.tag = tag
        self.axis = axis
        if selection:
            self.selection = selection
        else:
            self.selection = {"A": "", "B": "", "C": "", "D": ""}

    def __getitem__(self, index):
        return self.selection[index]

    def __setitem__(self, index, value):
        self.selection[index] = value

    def __add__(self, other):
        return ABCDSel.combine(self, other, axis=self.axis)

    def __radd__(self, other):
        return self.__add__(other)

    @classmethod
    def combine(cls, a, b, *, axis=None):
        if a.tag and b.tag:
            tag = f"{a.tag}-{b.tag}"
        else:
            tag = a.tag if a.tag else b.tag

        sel = {}
        for r in ["A", "B", "C", "D"]:
            first_part = a.selection[r]
            second_part = b.selection[r]
            first_part = f"{first_part}" if first_part else ""
            second_part = f"{second_part}" if second_part else ""
            if first_part and second_part:
                if axis is None:
                    sel[r] = f"({first_part}) && ({second_part})"
                elif axis == 0:  # combine along x axis
                    if r == "A" or r == "C":
                        sel[r] = f"({first_part}) && ({second_part})"
                    if r == "B" or r == "D":
                        sel[r] = f"({first_part}) || ({second_part})"
                elif axis == 1:  # combine along y axis
                    if r == "A" or r == "B":
                        sel[r] = f"({first_part}) && ({second_part})"
                    if r == "C" or r == "D":
                        sel[r] = f"({first_part}) || ({second_part})"
                else:
                    raise ValueError(f"invalid axis {axis}")
            else:
                sel[r] = first_part if first_part else second_part
        return cls(tag, sel, axis)

    @classmethod
    def Create(cls, tag, xcut, ycut, reverse_x=False, reverse_y=False):
        """
        create tag and selection

        Args:

            xcut (tuple) = tuple for splitting A and B regions.
                the first entry is the name of oboservable, and the
                second entry is the threshold/vlaue. e.g ("met", 40) using 'met'
                variable and cuts at 40

            ycut (tuple) = same as xcut but for B and C regions

            reverse_x (bool) = reversing about xcut value

            reverse_y (bool) = reversing about ycut value
        """

        if isinstance(xcut, tuple) and isinstance(ycut, tuple):
            pass
        else:
            log.critical(f"xcut and ycut need to be {type(tuple())}")
            return None

        x_pass = ">="
        x_fail = "<"
        y_pass = ">="
        y_fail = "<"

        if reverse_x:
            log.info(f"reversing about X for {tag}")
            x_pass = "<="
            x_fail = ">"
        if reverse_y:
            log.info(f"reversing about Y for {tag}")
            y_pass = "<="
            y_fail = ">"

        regions = {
            "A": f"{xcut[0]} {x_pass} {xcut[1]} && {ycut[0]} {y_pass} {ycut[1]}",
            "B": f"{xcut[0]} {x_pass} {xcut[1]} && {ycut[0]} {y_fail} {ycut[1]}",
            "C": f"{xcut[0]} {x_fail} {xcut[1]} && {ycut[0]} {y_pass} {ycut[1]}",
            "D": f"{xcut[0]} {x_fail} {xcut[1]} && {ycut[0]} {y_fail} {ycut[1]}",
        }

        return cls(tag, regions)


class ABCDCase(object):
    """
    Meta data class to hold regions name.
    """

    __slots__ = (
        "tag",
        "parent",
        "regions",
        "nevent",
        "tf_region",
        "tf_regions",
        "suffix",
    )

    def __init__(self, tag, parent):
        """
        class construtor

        Args:
            tag (str) = abcd tag

            parent (str) = name of the region for create ABCD regions.
        """
        self.tag = tag
        self.parent = parent
        self.regions = {
            "A": f"{parent}_ABCD-{tag}-rA_",
            "B": f"{parent}_ABCD-{tag}-rB_",
            "C": f"{parent}_ABCD-{tag}-rC_",
            "D": f"{parent}_ABCD-{tag}-rD_",
        }
        self.suffix = f"ABCD-{tag}"
        self.nevent = {"A": 0, "B": 0, "C": 0, "D": 0}
        # hold all possible transfer factor like regions, e.g.
        # including A/B, (A+C)/(B+D), (A+B)/(C+D), A/C, B/C
        self.tf_regions = {
            "C/D": f"ABCD-TF-{tag}-{parent}-C_div_D",
            "A/B": f"ABCD-TF-{tag}-{parent}-A_div_B",
            "A/C": f"ABCD-TF-{tag}-{parent}-A_div_C",
            "B/D": f"ABCD-TF-{tag}-{parent}-B_div_D",
            "AB/CD": f"ABCD-TF-{tag}-{parent}-AB_div_CD",
            "AC/BD": f"ABCD-TF-{tag}-{parent}-AC_div_BD",
        }
        # nominal tranfer factor region
        self.tf_region = self.tf_regions["C/D"]

    def __getitem__(self, index):
        return (self.regions[index], self.nevent[index])

    def get_signal(self):
        return self.regions["A"]


class ABCDTag(object):
    __slots__ = ("tag", "selection", "cases", "tf", "fake")

    def __init__(self, tag, abcd_sel=ABCDSel()):
        self.tag = tag
        self.selection = abcd_sel
        self.cases = []
        self.tf = {}
        self.fake = []

    @property
    def _cases_dict(self):
        return {case.parent: case for case in self.cases}

    def get_signal(self, base_region):
        try:
            return self._cases_dict[base_region].get_signal()
        except:
            return None

    def create_tf(self, proc_name):
        if proc_name in self.tf:
            log.warning(f"{proc_name} already in ABCDTag.tf")
        else:
            self.tf[proc_name] = deepcopy(self.cases)

    def create_fake(self, const_tf=True, *, pname=""):
        if pname:
            fakes_name = f"{pname}-const_tf" if const_tf else f"{pname}-binned_tf"
        else:
            fakes_name = "fakes"
        if fakes_name not in self.fake:
            self.fake.append(fakes_name)
        return fakes_name


class ABCDMetaData(object):
    """
    Keeping track of ABCD meta data
    """

    __slots__ = ("abcd_tag", "tag_list", "_has_tf", "tf_process")

    def __init__(self):
        self.abcd_tag = {}  # dict for ABCDTag objects
        self.tag_list = ()
        self._has_tf = False
        self.tf_process = {}  # keep track of process that has tf

    def __getitem__(self, index):
        return self.abcd_tag[index]

    def __setitem__(self, index, value):
        log.warning("ABCDMetaData class dose not support __setitem__")

    def __contains__(self, tag):
        return tag in self.abcd_tag

    @property
    def has_tf(self):
        return self._has_tf

    @has_tf.setter
    def has_tf(self, rhs):
        self._has_tf = rhs

    def keys(self):
        return self.abcd_tag.keys()

    def create_tag(self, tag, xcut, ycut, *, reverse_x=False, reverse_y=False):
        """
        Create ABCD tag and the corresponding seletion that define the ABCD regions

        Args:

            abcd_tag (str) = name of the ABCD study. this also serves for later reference.

            xcut (tuple) = tuple for splitting A and B regions.
                the first entry is the name of oboservable, and the
                second entry is the threshold/vlaue. e.g ("met", 40) using 'met'
                variable and cuts at 40

            ycut (tuple) = same as xcut but for B and C regions

            reverse_x (bool) = reversing about xcut value

            reverse_y (bool) = reversing about ycut value

        Return:

            no return.
        """

        if tag in self.abcd_tag:
            log.warning(f"Existing ABCD tag {tag}!")
            log.warning("Existing content will be overwrited!")

        abcd_sel = ABCDSel.Create(tag, xcut, ycut, reverse_x, reverse_y)
        self.abcd_tag[tag] = ABCDTag(tag, abcd_sel)

    def merge_tag_list(self, tag_list, *, tag_name=None, overwrite=True, axis=0):
        """
        method for merge existing abcd tag

        Args:
            tag_list (list) = list of exsiting tags to be merge
        """
        if len(tag_list) < 1:
            log.critcal("require more than 1 tag to merge")
            return tag_list[0]
        else:
            new_tag = tag_name if tag_name else "-".join(tag_list)
            if new_tag in self.abcd_tag:
                log.warning(f"Existing ABCD tag {new_tag}!")
                if not overwrite:
                    return new_tag
                else:
                    log.warning("Existing content will be overwritten!")
            abcd_sel = ABCDSel()
            abcd_sel.axis = axis
            for tag in tag_list:
                try:
                    abcd_sel += self.abcd_tag[tag].selection
                except KeyError as e:
                    raise KeyError(
                        f"Does not find tag {tag} to merge."
                        f"Please check your have reserve everything for {tag_list}."
                    ) from e
            self.abcd_tag[new_tag] = ABCDTag(new_tag, abcd_sel)
            return new_tag


# ==============================================================================

# ==============================================================================


def reserve_abcd_regions(
    config, abcd_tag, xcut, ycut, *, reverse_x=False, reverse_y=False
):
    """
    Function for reserving ABCD regions. This dose not create regions immeditely, but produce
    ABCD meta data. User requires to call create_abcd_regions with the corresponding tag to create
    Region objects.

    Args:

        config (physana.configMgr.ConfigMgr) : configMgr object.

        abcd_tag (str) = name of the ABCD study. this also serves for later reference.

        xcut (tuple of (str, float or int)) = tuple for splitting A and B regions.
            the first entry is the name of oboservable, and the
            second entry is the threshold/vlaue. e.g ("met", 40) using 'met'
            variable and cuts at 40

        ycut (tuple of (str, float or int)) = same as xcut but for B and C regions

        reverse_x (bool) = reversing about xcut value

        reverse_y (bool) = reversing about ycut value

    Return:

        no return. handle reserved regions within internal dict _abcd_region_selection

    Usage:

         >>> my_config # pre-defined ConfigMgr class object
         >>> my_config.reserve_abcd_regions("met_vs_lep1" ,("met", 25), ("lep1Signal", 1) )
    """
    _now = time.time()
    log.info(f"reserving abcd regions with {abcd_tag}")

    if "abcd" not in config.meta_data:
        config.meta_data["abcd"] = ABCDMetaData()

    config.meta_data["abcd"].create_tag(
        abcd_tag, xcut, ycut, reverse_x=reverse_x, reverse_y=reverse_y
    )
    config.reserve_branches(xcut[0])
    config.reserve_branches(ycut[0])

    log.info(f"cost {time.time()-_now:.2f}sec")


def _create_abcd_regions(config, tag, region, hist_type_filter=None, rA_only=False):
    """
    function for creating abcd regions based on the regions in the config.
    i.e. This function applies additional selection on the input region to
    create the ABCD regions.

    Args:
        config (physana.configMgr.ConfigMgr) : configMgr object

        tag (str) = abcd tag

        region (collinarw.core.Region) = parent region for create ABCD regions.

        rA_only: bool, default=False
            only enbale region A creation.

    Returns:
        ABCD regions
    """
    abcd_regions = []

    my_case = ABCDCase(tag, region.name)

    meta = config.meta_data["abcd"][tag]
    for r in ["A"] if rA_only else ["A", "B", "C", "D"]:
        r_copy = region.copy()
        r_copy.type = f"ABCD-Region{r}-{tag}"
        r_copy.name = my_case[r][0]
        r_copy.selection += f" && ({meta.selection[r]})"
        r_copy.hist_type_filter = Filter(hist_type_filter, key="type")
        abcd_regions.append(r_copy)
    return (abcd_regions, my_case)


def create_abcd_regions(
    config,
    tag_list,
    *,
    tag_name=None,
    base_region_names=None,
    use_mp=True,
    axis=0,
    filter_base_region=True,
    hist_type_filter={"reco", "tf"},
    rA_only=False,
):
    """
    Function that automatically create ABCD regions.
    If base_region_names is None, it will apply to regions with type = "abcd"

    Args:
        config: physana.configMgr.ConfigMgr
            ConfigMgr object for creating ABCD regions.

        tag_list: list(str)
            list of abcd tags

        tag_name: str, default=None
            Use for renaming the ABCD tag. if tag_name == None,
            tag_name will be constructed from "-".join(tag_list)

        base_region_names: list(str), default=None
            list of base region names for creating ABCD regions.

        use_mp: bool, default=True
            Boolean swith to enable internal multiprocessing.

        axis: int, defautl=0
            Axis used to combine different abcd tags.
            Default is 1 along y, and 0 along x.

        filter_base_region: bool, default=True
            Enabling base region filtering.
    """

    if "abcd" not in config.meta_data:
        raise KeyError("Cannot find abcd meta data. did you reserve abcd?")

    log.info(f"Creating abcd regions with {tag_list}, axis={axis}")
    if base_region_names:
        log.info(f"found {len(base_region_names)} base regions.")

    used_base_regions = set()

    _now = time.time()
    if len(tag_list) > 1:
        abcd_tag = config.meta_data["abcd"].merge_tag_list(
            tag_list, tag_name=tag_name, axis=axis
        )
    else:
        abcd_tag = tag_list[0]
        if tag_name:
            config.meta_data["abcd"].abcd_tag[tag_name] = deepcopy(
                config.meta_data["abcd"][abcd_tag]
            )
            abcd_tag = tag_name

    config.meta_data["abcd"].tag_list += (abcd_tag,)

    with tqdm.tqdm(
        desc="Creating ABCD regions",
        total=len(config.regions),
        unit="region",
        dynamic_ncols=True,
    ) as pbar:
        if use_mp:
            workers = CPU_CORE if CPU_CORE else os.cpu_count()
            pbar.desc += f"({workers} workers)"
            with concurrent.futures.ProcessPoolExecutor(workers) as pool:
                buffer = []
                for r in config.regions:
                    do_create = False
                    if base_region_names is None:
                        do_create = r.type == "abcd"
                    else:
                        do_create = r.name in set(base_region_names)
                    if do_create:
                        log.debug(f"(use_mp) create ABCD for region {r.name}")
                        _job = pool.submit(
                            _create_abcd_regions,
                            config,
                            abcd_tag,
                            r,
                            hist_type_filter,
                            rA_only,
                        )
                        buffer.append(_job)
                        used_base_regions.add(r.name)

                # update progress bar total based on the buffer
                pbar.total = len(buffer)

                for item in concurrent.futures.as_completed(buffer):
                    abcd_regions, case = item.result()
                    for r in abcd_regions:
                        if not config.has_region(r.name):
                            config.append_region(r, reserve_branches=False, aux=True)
                    config.meta_data["abcd"][abcd_tag].cases.append(case)
                    pbar.update()
        else:
            for r in config.regions:
                abcd_regions = None
                do_create = False
                if base_region_names is None:
                    do_create = r.type == "abcd"
                else:
                    do_create = r.name in set(base_region_names)
                if do_create:
                    log.debug(f"create ABCD for region {r.name}")
                    abcd_regions, case = _create_abcd_regions(
                        config, abcd_tag, r, hist_type_filter, rA_only
                    )
                    used_base_regions.add(r.name)
                if abcd_regions:
                    for r in abcd_regions:
                        if not config.has_region(r.name):
                            config.append_region(r, reserve_branches=False, aux=True)
                    config.meta_data["abcd"][abcd_tag].cases.append(case)
                pbar.update()

    if filter_base_region:
        config.region_name_filter |= used_base_regions

    log.info(f"cost {time.time()-_now:.2f} sec")


def add_region(config, name, selection, tag_list, tag_name=None, **kwargs):
    """
    wrapper of physana.configMgr.ConigMgr.add_region.
    Creating ABCD regions with given a base region.
    """
    if "abcd" not in config.meta_data:
        raise KeyError("Cannot find abcd meta data. did you reserve abcd?")
    else:
        kwargs.setdefault("study_type", "plot")
        kwargs.setdefault("corr_type", "None")
        kwargs.setdefault("reserve_branches", False)
        r = Region(
            name, kwargs["weight"], selection, kwargs["study_type"], kwargs["corr_type"]
        )
        if not config.has_region(r.name):
            config.append_region(r, kwargs["reserve_branches"])

        if len(tag_list) > 1:
            abcd_tag = config.meta_data["abcd"].merge_tag_list(
                tag_list, tag_name=tag_name, overwrite=False
            )
        else:
            abcd_tag = tag_list[0]
            if tag_name:
                config.meta_data["abcd"].abcd_tag[tag_name] = deepcopy(
                    config.meta_data["abcd"][abcd_tag]
                )
                abcd_tag = tag_name
        config.meta_data["abcd"].tag_list += (abcd_tag,)
        abcd_regions, case = _create_abcd_regions(config, abcd_tag, r)
        for r in abcd_regions:
            if not config.has_region(r.name):
                r.branch_reserved = True
                config.append_region(r, reserve_branches=False, aux=True)
        config.meta_data["abcd"][abcd_tag].cases.append(case)


def get_abcd_tags(config):
    if "abcd" in config.meta_data:
        return config.meta_data["abcd"].tag_list
    else:
        return ()


def get_selection(config, abcd_tag):
    """
    return the selections for given ABCD tag.
    """
    return config.meta_data["abcd"][abcd_tag].selection


def abcd_signal_region_name(config, base_region, abcd_tag):
    """
    base_region := base region name in the ABCD study
    abcd_tag := abcd reference tag created through create_abcd_regions
    """
    if "abcd" in config.meta_data:
        return config.meta_data["abcd"][abcd_tag].get_signal(base_region)
    else:
        raise KeyError("Cannot find abcd meta data. did you reserve abcd?")


def abcd_get_base_region(config):
    """
    Retrieve all the base regions that used in the ABCD construction
    """
    abcd_meta = config.meta_data["abcd"]
    base_regions = defaultdict(list)
    for tag in abcd_meta.abcd_tag.values():
        for case in tag.cases:
            base_regions[case.parent].append(case.regions)
    return base_regions


def abcd_get_base_region_name(config):
    return set(abcd_get_base_region(config).keys())


def set_tf(config):
    try:
        config.meta_data["abcd"].has_tf = True
    except Exception as _error:
        raise Exception("Unable to find abcd meta data") from _error


def check_tf(config):
    try:
        return config.meta_data["abcd"].has_tf
    except:
        return False


def get_tf_process(config):
    try:
        return config.meta_data["abcd"].tf_process
    except:
        return None


def _meta_data(config):
    if not config.meta_data["abcd"]:
        raise ValueError("Cannot find ABCD meta data")
    else:
        log.info("Found ABCD meta data. Check existing ABCD tags.")
        abcd_meta = config.meta_data["abcd"]
        tag_list = abcd_meta.keys()
        log.info(f"Found the following ABCD tags {tag_list}")
        return abcd_meta, tag_list


# ==============================================================================


# ==============================================================================
def _unpack_correlation_correction(name):
    correlation = []
    filter = []
    m_serial = Serialization()
    data = m_serial.from_pickle(name)
    if not isinstance(data, list):
        raise TypeError(f"Unexcpeted type{type(data)}")
    for d in data:
        if not isinstance(d, dict):
            raise TypeError(f"Cannot parse type {type(d)} in {data}")
        histogram = d["k_factor"]
        observable = d["observable"]
        region_filter = d["filter"]
        w_gen = functools.partial(weight_from_hist, obs=observable, hist=histogram)
        w_filter = region_filter
        correlation.append(w_gen)
        filter.append(w_filter)
    return correlation, filter


def _unpack_correlation_correction_list(names):
    correlation = []
    filter = []
    for name in names:
        corr, fil = unpack_correlation_correction(name)
        correlation += corr
        filter += fil
    return correlation, filter


def unpack_correlation_correction(input_corr):
    if isinstance(input_corr, list):
        return _unpack_correlation_correction_list(input_corr)
    elif isinstance(input_corr, str):
        return _unpack_correlation_correction(input_corr)
    else:
        raise TypeError(f"Cannot unpack correlation: {input_corr}, {type(input_corr)}")


def _correlation_k_factor(histo_dict, *, const_CD=False):
    """
    Compute correlation correction with given histograms from ABCD regions

    Args:
        histo_dict : {}
            histogram dictionary contains
                {
                    'hA' : core.Histogram or core.Histogram2D
                    'hB' : core.Histogram or core.Histogram2D
                    'hC' : core.Histogram or core.Histogram2D
                    'hD' : core.Histogram or core.Histogram2D
                }

        const_CD : boolean
            if const_CD == True, the ratio of C/D will be computed with integral
            instead of bin-to-bin.

    Return:
        core.Histogram objects
        k_factor, AB ratio, CD ratio
    """
    hA = histo_dict["hA"]
    hB = histo_dict["hB"]
    hC = histo_dict["hC"]
    hD = histo_dict["hD"]
    # compute ratio of A/B and C/D
    ratio_AB = hA / hB
    if const_CD:
        ratio_CD = hC.integral("all") / hD.integral("all")
    else:
        ratio_CD = hC / hD
        ratio_CD.nan_to_num()

    # compute correlation correction k-factor. i.e. k*(C/D) = A/B
    k_factor = ratio_AB / ratio_CD

    k_factor.nan_to_num()
    ratio_AB.nan_to_num()

    return k_factor, ratio_AB, ratio_CD


def derive_correlation_correction(
    config,
    base_region,
    abcd_tag,
    histograms,
    observables,
    save=None,
    *,
    region_filter="*electron*",
    corr_type="electron",
    process_name="dijets",
    abcd_observables=["met"],
    debug_plot=False,
):
    """
    Derive correlation correction for correlated ABCD observables

    Args:
        config : str or configMgr.ConfigMgr
            Config object.

        base_region : str
            Name of the base region that used in the ABCD construction

        abcd_tag : str
            Tag/name reference of a specific ABCD plane during config preparation.
            Both base_reigon and abcd_tag are used for looking up the ABCD full name.

        histograms: list(str)
            List of histogram names for lookup via region.get_observable.
            This is the actual name reference to retrieve histograms within a
            Region instance.

        observables : list(str)
            This is the Ntuple branch name corresponding to the 'histograms' arguments.
            Ususally it's the same as the name in 'histograms', unless a different
            name for 'observable' was specified during config preparation.
    """

    config = ConfigMgr.open(config)

    raw_dijets = config.get_process(process_name).copy()

    rA = abcd_signal_region_name(config, base_region, abcd_tag)
    rB = rA.replace("-rA", "-rB")
    rC = rA.replace("-rA", "-rC")
    rD = rA.replace("-rA", "-rD")

    # keep record of untouceh rA and rC
    raw_region_C = raw_dijets.get_region(rC)
    raw_region_A = raw_dijets.get_region(rA)
    raw_region_D = raw_dijets.get_region(rD)

    output = []
    phasespace_corr_obs = []

    # keep track of observables used in the correction
    name_order = ""

    dijets = config.get_process(process_name)

    for histo, obs in zip(histograms, observables):
        name_order += f"_{histo}"

        dijets = config.get_process(process_name)
        # resetting rC and rA after refilling
        dijets.get_region(rC).clear_content()
        dijets.get_region(rA).clear_content()
        dijets.get_region(rD).clear_content()
        dijets.get_region(rC).add(raw_region_C)
        dijets.get_region(rA).add(raw_region_A)
        dijets.get_region(rD).add(raw_region_D)

        dijets_obsA = raw_dijets.get_region(rA).get_observable(histo)
        dijets_obsB = dijets.get_region(rB).get_observable(histo)
        dijets_obsC = raw_dijets.get_region(rC).get_observable(histo)
        dijets_obsD = raw_dijets.get_region(rD).get_observable(histo)
        histo_dict = {
            "hA": dijets_obsA,
            "hB": dijets_obsB,
            "hC": dijets_obsC,
            "hD": dijets_obsD,
        }
        if obs in abcd_observables:
            k_factor, ratio_AB, ratio_CD = _correlation_k_factor(
                histo_dict, const_CD=True
            )
            # temperoraly make the first 0-th bin for the met.
            # in the case of other observable this might not be the case
            # need a better way to handle observable that is ABCD observable,
            # where bin-to-bin matching k-factor is not available
            k_factor.bin_content[0] = ratio_CD
        else:
            k_factor, ratio_AB, ratio_CD = _correlation_k_factor(histo_dict)

        # append to output
        _output = {}
        _output["k_factor"] = k_factor.copy()
        _output["observable"] = obs
        _output["filter"] = ("region_name", region_filter)
        output.append(_output)

        # using the phase-space correction for refilling here
        lookup = (corr_type, process_name, obs, None)
        if lookup in config.corrections:
            config.corrections[lookup].mul(k_factor)
        else:
            config.corrections.update({lookup: k_factor})
            phasespace_corr_obs.append(obs)
            config.phasespace_corr_obs = phasespace_corr_obs
        refill_process(config, [process_name])

        if debug_plot:
            plotmaker = PlotMaker(config, f"correlation_debug{name_order}")

            dijets_hA = dijets_obsA.root
            dijets_hB = dijets_obsB.root
            dijets_hC = dijets_obsC.root
            dijets_hD = dijets_obsD.root

            draw_opt = "HE" if isinstance(dijets_obsA, (Histogram)) else "colz"

            canvas = plotmaker.make_canvas()
            canvas.Divide(2, 2)
            canvas.cd(2)
            dijets_hA.SetLineColor(2)
            dijets_hA.Draw(draw_opt)

            canvas.cd(4)
            dijets_hB.SetLineColor(3)
            dijets_hB.Draw(draw_opt)

            canvas.cd(1)
            dijets_hC.SetLineColor(4)
            dijets_hC.Draw(draw_opt)

            canvas.cd(3)
            dijets_hD.SetLineColor(5)
            dijets_hD.Draw(draw_opt)

            canvas.SaveAs(f"{plotmaker.output_dir}/ABCD_{histo}.png")

            # ------------------------------
            canvas = plotmaker.make_canvas()

            canvas.Divide(1, 2)

            canvas.cd(1)
            ratio_AB_h = ratio_AB.root
            ratio_AB_h.SetLineColor(2)
            ratio_AB_h.Draw(draw_opt)

            if isinstance(ratio_CD, (Histogram, Histogram2D)):
                canvas.cd(2)
                ratio_CD_h = ratio_CD.root
                ratio_CD_h.SetLineColor(3)
                ratio_CD_h.Draw(draw_opt)

            canvas.SaveAs(f"{plotmaker.output_dir}/ABCD_ratio_{histo}.png")

            # ------------------------------
            canvas = plotmaker.make_canvas()
            canvas.cd()
            k_factor.ytitle = "Correction Factor"
            k_factor_h = k_factor.root
            k_factor_h.Draw(draw_opt)

            canvas.SaveAs(f"{plotmaker.output_dir}/ABCD_k_factor_{histo}.png")

            # ------------------------------
            canvas = plotmaker.make_canvas()
            canvas.cd()
            scaled_hB = k_factor * dijets_obsB
            scaled_AB = dijets_obsA / scaled_hB
            scaled_AB.ytitle = "Ratio of Region A/B"
            scaled_AB.nan_to_num()
            scaled_AB_h = scaled_AB.root
            scaled_AB_h.Draw(draw_opt)

            canvas.SaveAs(f"{plotmaker.output_dir}/ABCD_scaled_ratio_{histo}.png")

            # ------------------------------
            canvas = plotmaker.make_canvas()
            canvas.cd()
            norm_A = dijets_obsA / dijets_obsA.integral()
            norm_B = dijets_obsB / dijets_obsB.integral()
            ratio_norm_AB = norm_A / norm_B
            ratio_norm_AB.ytitle = "Norm. Ratio of Region A/B"
            ratio_norm_AB.nan_to_num()
            ratio_norm_AB_h = ratio_norm_AB.root
            ratio_norm_AB_h.Draw(draw_opt)

            canvas.SaveAs(f"{plotmaker.output_dir}/ABCD_norm_ratio_{histo}.png")

    if save:
        filename = str(pathlib.Path(save).resolve())
        m_serial = Serialization()
        m_serial.to_pickle(output, filename)
        config.save("test_scaled")

    return output


def transfer_factor(configMgr, process, systematic=None, all=False, remove_neg=True):
    """
    Generate transfer factor for given C and D region (ABCD method, see ABCD() ), and store
    it into the same process

    Args:
        configMgr: physana.configMgr.ConfigMgr
            An instance of class ConfigMgr.

        process : str
            name of the process for generating transfer factor.

        systematic : tuple(str)
            systematic full name for accessing systematic in a ProcessSet object.
            nominal will be return if the provide systematic is not found.

        all : bool, default = False
            compute all transfer factor likes regions. e.g A/B, A/C etc

    Returns:
        list of `core.Region` instance contains transfer factors.
    """

    usr_log.info("checking ABCD meta data")
    if not configMgr.meta_data["abcd"]:
        usr_log.info("Cannot find ABCD meta data")
        return None
    else:
        usr_log.info("Found ABCD meta data. Check existing ABCD tags.")
        abcd_meta = configMgr.meta_data["abcd"]
        tag_list = abcd_meta.keys()
        usr_log.info(f"Found the following ABCD tags {tag_list}")

    abcd_meta = configMgr.meta_data["abcd"]
    abcd_meta.has_tf = True
    my_process = configMgr.get_process_set(process).get(systematic)
    my_process.use_cache = True  # turn on cache

    if my_process.systematic is None and systematic is not None:
        my_process.systematic = Systematics(*systematic, "input source")

    output = []  # this contains tf regions
    for tag in tag_list:
        usr_log.info(f"working on {tag}")

        # abcd_meta[tag] is a ABCDTag object.
        abcd_meta[tag].create_tf(process)
        if process not in abcd_meta.tf_process:
            abcd_meta.tf_process[process] = set([tag])
        else:
            abcd_meta.tf_process[process].add(tag)

        for case in abcd_meta[tag].tf[process]:  # loop ABCDCase in tf[process]
            # usr_log.info(f"Calculating TF for {case.parent} of {process}")
            rA_name, rA_num = case["A"]
            rB_name, rB_num = case["B"]
            rC_name, rC_num = case["C"]
            rD_name, rD_num = case["D"]
            rA = my_process.get_region(rA_name)
            rB = my_process.get_region(rB_name)
            rC = my_process.get_region(rC_name)
            rD = my_process.get_region(rD_name)
            rA.use_cache = True
            rB.use_cache = True
            rC.use_cache = True
            rD.use_cache = True
            case.nevent["A"] = rA.effective_event_count
            case.nevent["B"] = rB.effective_event_count
            case.nevent["C"] = rC.effective_event_count
            case.nevent["D"] = rD.effective_event_count

            tf_name = case.tf_region
            # tf_desc = f"Transfer factor for region {case.parent} with tag {tag}"
            tf_region = Region(tf_name, "", "", "tf")
            for rC_obs in rC:
                rD_obs = rD.get_histogram(rC_obs.name)
                # rC_obs.remove_negative_bin()
                tf_obs = rC_obs / rD_obs
                if isinstance(rC_obs, Histogram):
                    tf_obs.ytitle = "Transfer Factor (C/D)"
                if remove_neg:
                    tf_obs.remove_negative_bin()
                tf_obs.nan_to_num()
                tf_region.add_histogram(tf_obs)
            output.append(tf_region)
            log.info(f"{tag} {case.parent}: {case.nevent}")

            if all:
                for tf_r_key in case.tf_regions:
                    tf_name = case.tf_regions[tf_r_key]
                    if tf_name == case.tf_region:
                        continue
                    # tf_desc = f"Transfer factor {tf_r_key} for region {case.parent} with tag {tag}"
                    tf_region = Region(tf_name, "", "", "tf")
                    for rC_obs in rC:
                        rA_obs = rA.get_histogram(rC_obs.name)
                        rB_obs = rB.get_histogram(rC_obs.name)
                        rD_obs = rD.get_histogram(rC_obs.name)
                        if tf_r_key == "A/C":
                            tf_obs = rA_obs / rC_obs
                        elif tf_r_key == "A/B":
                            tf_obs = rA_obs / rB_obs
                        elif tf_r_key == "B/D":
                            tf_obs = rB_obs / rD_obs
                        elif tf_r_key == "AB/CD":
                            tf_obs = (rA_obs + rB_obs) / (rC_obs + rD_obs)
                        elif tf_r_key == "AC/BD":
                            tf_obs = (rA_obs + rC_obs) / (rB_obs + rD_obs)

                        if isinstance(rC_obs, Histogram):
                            tf_obs.ytitle = f"Transfer Factor ({tf_r_key})"
                        if remove_neg:
                            tf_obs.remove_negative_bin()
                        tf_obs.nan_to_num()
                        tf_region.add_histogram(tf_obs)
                    output.append(tf_region)

            rA.use_cache = False
            rB.use_cache = False
            rC.use_cache = False
            rD.use_cache = False

    my_process.use_cache = False  # turn off cache

    return output


def fakes(config, const_tf=False, ext_tf=None, systematic=None):
    """
    Fake estimation based on the result from transfer fector (ABCD_TF).

    Notes:
        Need to run ABCD_TF first.

    Args:
        config (obj:ConfigMgr) : instance of configMgr that contains the ABCD result.

        const_tf (bool) : using constant transfer factor stored in ABCD meta data.

        ext_tf () : external transfer factor. either a tuple of ConfigMgr and process name,
        or just a Process. Assuming the ext_tf has the same ABCD tags and regions.

        systematic (tuple) : systematic full name

    Returns:
        return list of fake processes.
    """

    usr_log.info("checking ABCD meta data")
    if not config.meta_data["abcd"]:
        usr_log.info("Cannot find ABCD meta data")
        return None
    else:
        usr_log.info("Found ABCD meta data. Check existing ABCD tags.")
        abcd_meta = config.meta_data["abcd"]
        tag_list = abcd_meta.keys()
        usr_log.info(f"Found the following ABCD tags {tag_list}")

    if ext_tf is not None:
        if isinstance(ext_tf, tuple):
            if isinstance(ext_tf[0], ConfigMgr):
                m_pset = ext_tf[0].get_process_set(ext_tf[1])
                _ext_tf_p = m_pset.get(systematic).copy()
            else:
                raise ValueError(f"Invalid external TF {ext_tf}")
        elif isinstance(ext_tf, Process):
            _ext_tf_p = ext_tf.copy()
        else:
            raise ValueError(f"Invalid external TF {ext_tf}")
    else:
        _ext_tf_p = None

    output = []
    for tag in tag_list:
        log.debug(f"working on tag {tag}")
        for proc_name, cases in abcd_meta[tag].tf.items():
            # skip tag that dose not have abcd regions
            if not cases:
                continue

            my_process = config.get_process_set(proc_name).get(systematic)

            # create fake process & meta data
            if len(abcd_meta[tag].tf) == 1:
                fakes_name = abcd_meta[tag].create_fake(const_tf)
            else:
                fakes_name = abcd_meta[tag].create_fake(const_tf, pname=proc_name)
            fake_process = Process.fakes(fakes_name)
            fake_process.systematic = my_process.systematic

            if fake_process.systematic is None and systematic is not None:
                fake_process.systematic = Systematics(*systematic, "dummy")

            for case in cases:
                rA_name, rA_num = case["A"]
                rB_name, rB_num = case["B"]
                rC_name, rC_num = case["C"]
                rD_name, rD_num = case["D"]
                rB = my_process.get_region(rB_name).copy()
                rA = my_process.get_region(rA_name).copy()
                rA.clear()
                if const_tf:
                    log.debug(f"constant tf is used for {case.parent}")
                    if _ext_tf_p:
                        try:
                            rC_num = _ext_tf_p.get_region(rC_name).effective_event_count
                            rD_num = _ext_tf_p.get_region(rD_name).effective_event_count
                        except:
                            continue
                    if rD_num == 0:
                        tf = 0.0
                    else:
                        tf = rC_num / rD_num
                    for rB_h in rB.histograms:
                        if isinstance(rB_h, (Histogram, Histogram2D)):
                            fake_h = rB_h * tf
                            fake_h.remove_negative_bin()
                            rA.add_histogram(fake_h)
                else:
                    if _ext_tf_p:
                        try:
                            tf_region = _ext_tf_p.get_region(case.tf_region).copy()
                        except:
                            continue
                    else:
                        tf_region = my_process.get_region(case.tf_region).copy()
                    for rB_h in rB.histograms:
                        hist_name = rB_h.name
                        if isinstance(rB_h, (Histogram, Histogram2D)):
                            try:
                                tf_h = tf_region.get_histogram(hist_name)
                                fake_h = rB_h * tf_h
                            except ValueError as e:
                                log.critical(f"fake estimation encounter {e}")
                                continue
                            except KeyError as e:
                                log.critical(f"fake estimation encounter {e}")
                                continue
                            except:
                                continue
                            fake_h.remove_negative_bin()
                            rA.add_histogram(fake_h)
                fake_process.add_region(rA)
            output.append(fake_process)
            log.info(f"result is stored in {fakes_name}")
    return output


def event_level_fakes(
    config,
    tf_config,
    tf_pname,
    tf_obs,
    *,
    match_tf_process=False,
    skip_process=None,
    use_mp=True,
    step_size=None,
    systematic=None,
    correlation_correction=None,
    executor=None,
    as_completed=None,
    split_base_region=True,
    workers=None,
):
    """
    Assingning TF as weight on event level and refill the histograms.

    Args:
        config : str or physana.configMgr.ConfigMgr:
            ConfigMgr object or the file name to the ConfigMgr object.

        tf_config : physana.configMgr.ConfigMgr:
            external config contains TF.

        tf_obs : [ (tuple(str), str) ] or { str : [ (tuple(str), str) ] }:
            observables and histogram name for TF.

            the number of of observables should match the histogram dimentsion.
            e.g. (('jet1Pt'), 'jet1Pt_hist') or (('lep1Pt', 'lep1Eta'), 'pt_vs_eta')

            if it's dict, the key will be treated as region name filter. e.g.
            {'electron*', (('jet1Pt'), 'jet1Pt_hist')} will only apply to region
            name that match 'electron*'

        correlation_correction : str or list(str)
            file that contains the met-calo isolation correlation correction.

    return:
        list(Process) : list of Process objects with regions rA = tf*rB.
    """
    usr_log.info("Event Level Fake Estimation")
    usr_log.info("checking ABCD meta data in tf_config")
    if not tf_config.meta_data["abcd"]:
        usr_log.info("Cannot find ABCD meta data")
        return None
    else:
        usr_log.info("Found ABCD meta data. Check existing ABCD tags.")
        abcd_meta = tf_config.meta_data["abcd"]
        tag_list = abcd_meta.keys()
        usr_log.info(f"Found the following ABCD tags {tag_list}")

    rA_rB_map = {}
    rA_rB_type_map = {}

    histmaker = HistMaker()
    histmaker.meta_data_from_config(config)

    if systematic:
        usr_log.info(f"running with systematic -> {systematic}")

    tf_process = tf_config.get_process_set(tf_pname).get(systematic)
    # branch_list = config.reserved_branches | tf_config.reserved_branches

    output_syst_name = None
    output_syst_full_name = None
    output_syst_type = None

    do_closing = False
    # if no external executor is provided
    # construct executor locally from concurrent.futures
    if not (executor and as_completed) and use_mp:
        do_closing = True
        expect_jobs = sum([len(p.filename) for p in config.processes])
        workers = workers or min(expect_jobs, int(os.cpu_count() * 0.5))
        executor = concurrent.futures.ProcessPoolExecutor(workers)
        as_completed = concurrent.futures.as_completed
        histmaker.nthread = 1
        usr_log.info(f"created internal mp pool with {workers=}.")

    branch_list = defaultdict(set)
    prepared_fake_processes = []
    prepared_tf_processes = []
    # preparing fake processes.
    for pset in config.process_sets:
        usr_log.info(f"checking process set {pset.name} {pset.type}")
        # if matching of the tf process name is required.
        if match_tf_process and (pset.name != tf_process.name):
            continue
        # only consider 'data' and 'mc' type
        if pset.type != "mc" and pset.type != "data":
            continue

        # only use for getting region info
        if pset.type == "data":
            p = pset.get()
        else:
            p = pset.get(systematic)

        master_fake_proc_buff = p.copy(shallow=True)
        master_fake_proc_buff.clear()

        master_tf_proc_buff = master_fake_proc_buff.copy(shallow=True)
        master_tf_proc_buff.clear()

        tf_proc_buff = master_tf_proc_buff
        fake_proc_buff = master_fake_proc_buff
        if not split_base_region:
            prepared_fake_processes.append(fake_proc_buff)
            prepared_tf_processes.append(tf_proc_buff)

        if p.systematic:
            output_syst_name = p.systematic.name
            output_syst_full_name = p.systematic.full_name
            output_syst_type = p.systematic.sys_type

        for tag in tag_list:
            log.debug(f"working on tag {tag}")
            for proc_name, cases in abcd_meta[tag].tf.items():
                # skip tag that does not have abcd regions
                if not cases:
                    continue
                if proc_name != tf_process.name:
                    continue

                for case in cases:
                    rA_name, rA_num = case["A"]
                    rB_name, rB_num = case["B"]
                    rC_name, rC_num = case["C"]
                    rD_name, rD_num = case["D"]

                    # checking rA and rB
                    if rA_name in p.list_regions():
                        rA_type = p.get_region(rA_name).type
                    else:
                        log.debug(f"no {rA_name}, skip.")
                        continue
                    try:
                        rB = p.get_region(rB_name).copy()
                        rA = p.get_region(rA_name)
                    except KeyError:
                        log.debug(f"no {rB_name}, skip.")
                        continue
                    rB.clear_content()

                    # if splitting jobs by base region, create copy here.
                    if split_base_region:
                        fake_proc_buff = master_tf_proc_buff.copy()
                        tf_proc_buff = master_tf_proc_buff.copy()
                        fake_proc_buff.clear()
                        tf_proc_buff.clear()
                        prepared_fake_processes.append(fake_proc_buff)
                        prepared_tf_processes.append(tf_proc_buff)
                        fake_proc_name = fake_proc_buff.name
                        branch_list[fake_proc_name] |= rB.ntuple_branches
                        branch_list[fake_proc_name] |= fake_proc_buff.ntuple_branches
                        branch_list[fake_proc_name] |= tf_proc_buff.ntuple_branches

                    if p.name in skip_process:
                        fake_proc_buff.add_region(rA)
                    else:
                        fake_proc_buff.add_region(rB)

                    # rename tf_region name to rB name, then we can use
                    # it by name to distribute and multiply.
                    tf_region = tf_process.get_region(case.tf_region).copy()
                    tf_region.name = rB.name
                    tf_proc_buff.add_region(tf_region)

                    # maping the rB and rA name and type for later use
                    rA_rB_map[rB_name] = rA_name
                    rA_rB_type_map[rB.type] = rA_type

    # temp buffer to holding submitted jobs
    tmp_output = []
    skip_output = []
    # prepare correlation correction
    if correlation_correction:
        _calo_corr = unpack_correlation_correction(correlation_correction)
        correlation = _calo_corr[0]
        correlation_filter = _calo_corr[1]
    else:
        correlation = []
        correlation_filter = []
    # start refilling jobs
    usr_log.info(f"number of prepared jobs {len(prepared_fake_processes)}")
    assert len(prepared_fake_processes) == len(prepared_tf_processes)
    zipped_prepared = zip(prepared_fake_processes, prepared_tf_processes)
    for fake_proc_buff, tf_proc_buff in zipped_prepared:
        # apply tf on event level
        file_name = fake_proc_buff.filename

        if skip_process and fake_proc_buff.name in skip_process:
            m_ext_weight = None
            usr_log.info(f"Skip ext tf weight for {fake_proc_buff.name}")
            skip_output.append(fake_proc_buff.copy())
            continue
        else:
            m_ext_weight = {
                "process": tf_proc_buff,
                "weight_hist": tf_obs,
            }
            usr_log.info(f"apply ext weight for {fake_proc_buff.name}")

        for fname in tqdm.tqdm(file_name):
            if use_mp:
                _result = executor.submit(
                    histmaker.plevel_process,
                    fake_proc_buff,  # no need to copy since child has its own copy
                    fname,
                    branch_list=branch_list.get(fake_proc_buff.name, None),
                    ext_rweight=m_ext_weight,
                    step_size=step_size,
                    weight_generators=correlation,
                    weight_generators_filters=correlation_filter,
                )
            else:
                _result = histmaker.plevel_process(
                    fake_proc_buff.copy(),  # require copy avoid double merge later
                    fname,
                    branch_list=branch_list.get(fake_proc_buff.name, None),
                    ext_rweight=m_ext_weight,
                    step_size=step_size,
                    weight_generators=correlation,
                    weight_generators_filters=correlation_filter,
                )
            tmp_output.append(_result)
        log.info(f"finished submition of {fake_proc_buff.name}")
        # remove from the prepared list
        # prepared_fake_processes.remove(fake_proc_buff)
        # prepared_tf_processes.remove(tf_proc_buff)

    usr_log.info(f"Finalizing fakes with {tf_pname}")
    output = []  # store output Process objects

    if use_mp:
        ready_for_process = as_completed(tmp_output)
    else:
        ready_for_process = tmp_output

    usr_log.info("Retrieving jobs and preparing post processing.")
    for _fake in ready_for_process:
        if use_mp:
            filled_fake = _fake.result()  # no copy is needed from child processes.
            # replace rB name with rA name
            for r in filled_fake.regions:
                r.type = rA_rB_type_map[r.type]
                r.name = rA_rB_map[r.name]
                for h in r.histograms:
                    h.remove_negative_bin()
            output_fake = filled_fake  # just do name reference
        else:
            filled_fake = _fake
            output_fake = filled_fake.copy(shallow=True)
            output_fake.regions = []
            # replace rB name with rA name, append to the output_fake process instance.
            for r in filled_fake.regions:
                c_r = r.copy()
                c_r.type = rA_rB_type_map[r.type]
                c_r.name = rA_rB_map[r.name]
                for h in c_r.histograms:
                    h.remove_negative_bin()
                    output_fake.add_region(c_r)

        if output_fake.systematic is None:
            if systematic is not None:
                if output_syst_name:
                    _syst = SystematicsBase(
                        output_syst_name,
                        output_syst_full_name,
                        "dummy",
                        output_syst_type,
                    )
                else:
                    _syst = Systematics(*systematic, "input source")
                output_fake.systematic = _syst

        output.append(output_fake)

    # closing internal executor
    if do_closing:
        usr_log.info("closing internal mp pool.")
        executor.shutdown()

    output += skip_output

    return output


def abcd_system_equations(p, *const):
    """
    data[A] = mu * signal[A] + Fake[A]
    data[B] = mu * signal[B] + Fake[B]
    data[C] = mu * signal[C] + Fake[C]
    data[D] = mu * signal[D] + Fake[D]
    Fake[A] = (Fake[C]/Fake[D]) * Fake[B]
    where mu is the correction/normalization factor to the signal
    """
    mu, Fa, Fb, Fc, Fd = p
    data_a, data_b, data_c, data_d, signal_a, signal_b, signal_c, signal_d = const

    eq1 = mu * signal_a + Fa - data_a
    eq2 = mu * signal_b + Fb - data_b
    eq3 = mu * signal_c + Fc - data_c
    eq4 = mu * signal_d + Fd - data_d
    eq5 = Fa - (Fc / Fd) * Fb

    return (
        eq1,
        eq2,
        eq3,
        eq4,
        eq5,
        # 1 - (Fa >= 0 and Fb >= 0 and Fc >= 0 and Fd >= 0 and mu > 0),
    )


def abcd_system_equations2(p, *const):
    """
    data[A] = mu * signal[A] + Fake[A]
    data[B] = signal[B] + Fake[B]
    data[C] = mu * signal[C] + Fake[C]
    data[D] = signal[D] + Fake[D]
    Fake[A] = (Fake[C]/Fake[D]) * Fake[B]
    where mu is the correction/normalization factor to the signal
    """
    mu, Fa, Fb, Fc, Fd = p
    data_a, data_b, data_c, data_d, signal_a, signal_b, signal_c, signal_d = const

    eq1 = mu * signal_a + Fa - data_a
    eq2 = signal_b + Fb - data_b
    eq3 = mu * signal_c + Fc - data_c
    eq4 = signal_d + Fd - data_d
    eq5 = Fa - (Fc / Fd) * Fb
    # eq5 = 1 - (Fa >= 0 and Fb >= 0 and Fc >= 0 and Fd >= 0 and mu > 0)

    return (eq1, eq2, eq3, eq4, eq5)


def abcd_system_equations3(p, *const):
    """
    data[A] = mu * signal[A] + Fake[A]
    data[B] = signal[B] + Fake[B]
    data[C] = mu * signal[C] + Fake[C]
    data[D] = signal[D] + Fake[D]
    Fake[A] = (Fake[C]/Fake[D]) * Fake[B]
    where mu is the correction/normalization factor to the signal
    """
    mu, Fa, Fb, Fc, Fd = p
    data_a, data_b, data_c, data_d, signal_a, signal_b, signal_c, signal_d = const

    eq1 = signal_a + Fa - data_a
    eq2 = signal_b + Fb - data_b
    eq3 = signal_c + Fc - data_c
    eq4 = signal_d + Fd - data_d
    eq5 = Fa - (Fc / Fd) * Fb
    # eq5 = 1 - (((Fa >= 0 and Fb >= 0) and (Fc >= 0 and Fd >= 0)) and mu > 0)
    print(eq5)

    return (eq1, eq2, eq3, eq4, eq5)


def correction_factor(config, data, signal, bkgd, observable=None):
    """
    config (configMgr.ConfigMgr) : config file.

    data (str) : name of data.

    bkgd (list(str)) : name of background processes

    """
    usr_log.info(f"calculating abcd correction factor to signal: {signal}")
    abcd_meta, tag_list = _meta_data(config)

    data_p = config.get_process(data)
    signal_p = config.get_process(signal)
    bkgd_p = [config.get_process(b) for b in bkgd]

    bkgd_subtract = data_p.copy()
    for bk in bkgd_p:
        bkgd_subtract.sub(bk)
    bkgd_subtract.name = "background_subtract"

    output = {}
    for tag in tag_list:
        usr_log.info(f"working on {tag}")
        # abcd_meta[tag] is a ABCDTag object.
        for case in abcd_meta[tag].cases:
            rA, _ = case["A"]
            rB, _ = case["B"]
            rC, _ = case["C"]
            rD, _ = case["D"]

            if observable:
                signal_a = signal_p.get_region(rA).get_observable(observable)
                signal_b = signal_p.get_region(rB).get_observable(observable)
                signal_c = signal_p.get_region(rC).get_observable(observable)
                signal_d = signal_p.get_region(rD).get_observable(observable)

                bkgd_sub_a = bkgd_subtract.get_region(rA).get_observable(observable)
                bkgd_sub_b = bkgd_subtract.get_region(rB).get_observable(observable)
                bkgd_sub_c = bkgd_subtract.get_region(rC).get_observable(observable)
                bkgd_sub_d = bkgd_subtract.get_region(rD).get_observable(observable)

            else:
                signal_a = signal_p.get_region(rA).effective_event_count
                signal_b = signal_p.get_region(rB).effective_event_count
                signal_c = signal_p.get_region(rC).effective_event_count
                signal_d = signal_p.get_region(rD).effective_event_count

                bkgd_sub_a = bkgd_subtract.get_region(rA).effective_event_count
                bkgd_sub_b = bkgd_subtract.get_region(rB).effective_event_count
                bkgd_sub_c = bkgd_subtract.get_region(rC).effective_event_count
                bkgd_sub_d = bkgd_subtract.get_region(rD).effective_event_count

            if observable:
                correction_histo = signal_a.copy()
                correction_histo.clear_content()
                nbin = len(correction_histo.bin_content)
                for i in range(nbin):
                    """
                    a = signal_a[i] * signal_d[i] - signal_c[i] * signal_b[i]
                    b = (
                        bkgd_sub_c[i] * signal_b[i]
                        + bkgd_sub_b[i] * signal_c[i]
                        - bkgd_sub_a[i] * signal_d[i]
                        - bkgd_sub_d[i] * signal_a[i]
                    )
                    c = bkgd_sub_a[i] * bkgd_sub_d[i] - bkgd_sub_c[i] * bkgd_sub_b[i]
                    roots = numpy.roots([a, b, c])  # root of quadratic equation
                    mu = roots[numpy.isreal(roots)]
                    pos_mu = mu[mu > 0]
                    valid_value = []
                    # print(f"{(signal,case.parent,tag)} real solution {mu}")
                    for v in pos_mu:
                        check_a = bkgd_sub_a[i] - v * signal_a[i]
                        check_b = bkgd_sub_b[i] - v * signal_b[i]
                        check_c = bkgd_sub_c[i] - v * signal_c[i]
                        check_d = bkgd_sub_d[i] - v * signal_d[i]
                        if check_b >= 0 and check_a >= 0:
                            valid_value.append(v)
                    """
                    # print(f"{(signal,case.parent,tag)} : bin{i}, {mu}")
                    print(f"Na {bkgd_sub_a[i]}")
                    print(f"Nb {bkgd_sub_b[i]}")
                    print(f"Nc {bkgd_sub_c[i]}")
                    print(f"Nd {bkgd_sub_d[i]}")
                    print(f"Wa {signal_a[i]}")
                    print(f"Wb {signal_b[i]}")
                    print(f"Wc {signal_c[i]}")
                    print(f"Wd {signal_d[i]}")
                    # if len(valid_value) == 1:
                    #    correction_histo[i] = valid_value[0]
                    # print(f"{(signal,case.parent,tag)} : {valid_value[0]}")
                    # else:
                    # print(f"more than 1 solution : {valid_value}")
                    #    correction_histo[i] = 1.0
                result = correction_histo
            else:
                a = signal_a * signal_d - signal_c * signal_b
                b = (
                    bkgd_sub_c * signal_b
                    + bkgd_sub_b * signal_c
                    - bkgd_sub_a * signal_d
                    - bkgd_sub_d * signal_a
                )
                c = bkgd_sub_a * bkgd_sub_d - bkgd_sub_c * bkgd_sub_b
                roots = numpy.roots([a, b, c])  # root of quadratic equation
                mu = roots[numpy.isreal(roots)]
                pos_mu = mu[mu >= 0]
                # usr_log.info(f"{signal} {tag} {case.parent}: {mu}")
                print(f"Na {bkgd_sub_a}")
                print(f"Nb {bkgd_sub_b}")
                print(f"Nc {bkgd_sub_c}")
                print(f"Nd {bkgd_sub_d}")
                print(f"Wa {signal_a}")
                print(f"Wb {signal_b}")
                print(f"Wc {signal_c}")
                print(f"Wd {signal_d}")
                # print(mu)
                # input()

                Na = bkgd_sub_a
                Nb = bkgd_sub_b
                Nc = bkgd_sub_c
                Nd = bkgd_sub_d
                Wa = signal_a
                Wb = signal_b
                Wc = signal_c
                Wd = signal_d

                const = (Na, Nb, Nc, Nd, Wa, Wb, Wc, Wd)

                _ratio = ((Nc - Wc) / (Nd - Wd)) * (Nb - Wb)
                _ratio = _ratio if _ratio > 0 else 0
                # print(_ratio)
                # res =  scipy.optimize.fsolve(equations, (1, Na-Wa, Nb-Wb, Nc-Wc, Nd-Wd))
                res = scipy.optimize.fsolve(
                    abcd_system_equations2,
                    (1, Na - Wa, Nb - Wb, Nc - Wc, Nd - Wd),
                    args=const,
                )
                print(f"{(signal, case.parent, tag)} : {res}")
                # mu, Fa, Fb, Fc, Fd
                '''
                for v in numpy.arange(0,2, 0.01):
                    f_a = bkgd_sub_a - v*signal_a
                    f_b = bkgd_sub_b - v*signal_b
                    f_c = bkgd_sub_c - v*signal_c
                    f_d = bkgd_sub_d - v*signal_d
                    #print(f"{v} a: {f_a}")
                    #print(f"{v} b: {f_b}")
                    #print(f"{v} c: {f_c}")
                    #print(f"{v} d: {f_d}")
                    print(f"{v} {f_a - (f_c/f_d)*f_b}")
                '''
                try:
                    result = pos_mu[0]
                except:
                    result = 1.0
            output[(signal, case.parent, tag)] = result
    return output
