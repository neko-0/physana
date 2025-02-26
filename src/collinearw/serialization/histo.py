import json
import yaml
import uproot
import numpy as np
import ast
import copy
import collections
import logging
from pathlib import Path

from .base import SerializationBase
from ..histo import Histogram
from ..systematics.syst_band import SystematicsBand

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SerialHistogram(SerializationBase):
    content = {"bin_content", "sumW2", "bins", "backend"}
    style = {
        "color",
        "alpha",
        "fillstyle",
        "linestyle",
        "linewidth",
        "markersize",
        "markerstyle",
        "xtitle",
        "ytitle",
    }

    def to_pickle(self, hist, *args, noparent=True, **kwargs):
        if noparent:
            super().to_pickle(hist.copy(shallow=True, noparent=True), *args, **kwargs)
        else:
            super().to_pickle(hist, *args, **kwargs)

    # ==========================================================================
    # Json Serialization of Histogram
    # ==========================================================================
    def to_json(self, hist, ofilename, *, extra_info=None):
        """
        Convert Histogram(Base) object to json format.
        """
        output_data = {}
        output_data["hist_class"] = str(type(hist).__name__)
        # need to provide set of basic argument to construct object
        basic_info = hist.basic_info()
        # check if there numpy type
        for name, value in basic_info.items():
            if isinstance(value, np.integer):
                basic_info[name] = value.item()
            elif isinstance(value, np.floating):
                basic_info[name] = value.item()
            elif isinstance(value, np.ndarray):
                basic_info[name] = value.tolist()
        output_data["basic_info"] = basic_info
        # bin content and core parts for histogram
        for name in self.content:
            if name in {"bin_content", "sumW2", "bins"}:
                output_data[name] = getattr(hist, name).tolist()
            else:
                output_data[name] = getattr(hist, name)
        if extra_info is not None:
            if "style" in extra_info:
                output_data["style"] = {}
                for name in self.style:
                    if not hasattr(hist, name):
                        continue
                    output_data["style"][name] = getattr(hist, name)
            if "structure" in extra_info:
                output_data["structure"] = getattr(hist, "full_name")
            # this is the core part which can be used to recover the python object
            if "systematic" in extra_info:
                output_data["systematic"] = hist.systematic
                output_data["systematic_band"] = {}
                for name, syst in hist.systematic_band.items():
                    flatten_syst = copy.deepcopy(syst.flatten())
                    flatten_syst["shape"] = str(flatten_syst.pop("shape"))
                    components = flatten_syst["components"]
                    for comp_syst in list(components["up"]):
                        np_array = components["up"].pop(comp_syst)
                        components["up"][str(comp_syst)] = np.array(np_array).tolist()
                    for comp_syst in list(components["down"]):
                        np_array = components["down"].pop(comp_syst)
                        components["down"][str(comp_syst)] = np.array(np_array).tolist()
                    output_data["systematic_band"][name] = flatten_syst
            # this part writes the overall up/down of each systematic band
            if "systematic-extra" in extra_info:
                output_data["systematic-extra"] = {}
                summary_up_down = {}
                for name, syst in hist.systematic_band.items():
                    summary_up_down[name] = {
                        "up": syst.up.tolist(),
                        "down": syst.down.tolist(),
                    }
                output_data["systematic-extra"]["up_down"] = summary_up_down

        with open(ofilename, "w") as f:
            json.dump(output_data, f)

    def from_json(self, ifilename):
        """
        parsing json back to python object

        example minimum requried format :
        {
            "hist_class": "Histogram",
            "basic_info": {"xmax": 20, "nbin": 1, "xmin": 0, "name": "nTruthBJet30"},
            "sumW2": [0.0, 10000.0, 0.0], "bin_content": [0.0, 1000.0, 0.0],
            "ytitle": "Number of events",
            "name": "nTruthBJet30",
            "xtitle": "Particle level Inclusive observable [number of b-jets (p_{T} > 30 GeV)]",
            "style": {
                "markerstyle": 8,
                "linewidth": 1,
                "markersize": 1.0,
                "color": 6533,
                "alpha": null,
                "fillstyle": 1001,
                "linestyle": 1
            },
            "structure": "/unfold_realthang_fake-EL/nominal/electron_inclusive_truth/nTruthBJet30",
            "systematic": null,
            "systematic_band": {
                "Jet": {
                    "type": "experimental",
                    "name": "Jet",
                    "shape": "(3,)",
                    "components": {
                        "up": {
                            "('JET_GroupedNP_1', 'symmetrize_up_down', 'up')": [0.0, 0.1, 0.0],
                            "('JET_GroupedNP_2', 'symmetrize_up_down', 'up')": [0.0, 0.1, 0.0]
                        },
                        "down": {
                            "('JET_GroupedNP_1', 'symmetrize_up_down', 'down')": [0.0, 0.1, 0.0],
                            "('JET_GroupedNP_2', 'symmetrize_up_down', 'down')": [0.0, 0.2, 0.0]
                        }
                    }
                }
            }
        }
        """
        with open(ifilename, "r") as f:
            idata = json.load(f)
        # constructe object
        hist = getattr(Histogram, idata["hist_class"])(**idata["basic_info"])
        # load in information saved in the json file
        for name in self.content:
            if name in {"bin_content", "sumW2", "bins"}:
                value = np.array(idata[name])
            else:
                value = idata[name]
            setattr(hist, name, value)
        for name, value in idata.get("style", {}).items():
            if not hasattr(hist, name):
                continue
            setattr(hist, name, value)
        systematic_band = idata.get("systematic_band", {})
        for syst_name, syst in systematic_band.items():
            syst["shape"] = ast.literal_eval(syst.pop("shape"))
            components = syst["components"]
            for comp_syst in list(components["up"]):
                try:
                    parsed_name = ast.literal_eval(comp_syst)
                except ValueError:
                    logger.warning(f"cannot use ast.literal_evel on {comp_syst}")
                    parsed_name = comp_syst
                value = np.array(components["up"].pop(comp_syst))
                value = [x if x is not None else 0 for x in value]
                components["up"][parsed_name] = np.array(value)
            for comp_syst in list(components["down"]):
                try:
                    parsed_name = ast.literal_eval(comp_syst)
                except ValueError:
                    logger.warning(f"cannot use ast.literal_evel on {comp_syst}")
                    parsed_name = comp_syst
                value = components["down"].pop(comp_syst)
                value = [x if x is not None else 0 for x in value]
                components["down"][parsed_name] = np.array(value)
            unflatten_band = SystematicsBand.loads(syst)
            hist.update_systematic_band(unflatten_band)

        return hist

    # ==========================================================================
    # Text Serialization of Histogram
    # ==========================================================================
    def _to_text_style_1(self, histo, output=None, corr=None, symmetrize=True):
        """
        Text format use for custom averaging tool of the following format:
            ErrorType Percent
            BinLabers   Bin1    Bin2    Bin3
            Binning     0.0     100.0   200.0   300.0
            Nominal     123     456     789
            StatData    0.01    0.02    0.03
            Jet_syst    0.01    0.02    0.03
            StatCorrData 3
            1.0 0.1 0.2
            0.1 1.0 0.0
            0.2 0.0 1.0
        """
        content = histo.bin_content[1:].astype(str)
        bin_index = range(1, len(content) + 1)
        bin_edge = [x for x in histo.bins.astype(str)] + [str(histo.bins[-1] * 2)]
        bin_labels = " ".join([f'Bin{i}' for i in bin_index])
        binning = " ".join([x for x in bin_edge])
        nominal = " ".join([x for x in content])
        stats = histo.statistical_error(ratio=True)['up']  # up/down are the same
        stats = " ".join([x for x in stats[1:].astype(str)])
        nbin = len(histo.bins)
        if corr is None:
            try:
                corr = histo.correlation_matrix()[1:]
            except AttributeError:
                corr = np.identity(nbin + 1)[1:]
        if output is None:
            output = "./"
        syst_list = []
        if histo.systematic_band is not None:
            for name, value in histo.systematic_band.items():
                if symmetrize:
                    m_data = [x for x in value.average()[1:].astype(str)]
                    syst_list.append(f"{name} {' '.join(m_data)}")
                else:
                    m_data_up = [x for x in value.up[1:].astype(str)]
                    m_data_dn = [x for x in (-1.0 * value.down[1:]).astype(str)]
                    syst_list.append(f"{name}_up {' '.join(m_data_up)}")
                    syst_list.append(f"{name}_down {' '.join(m_data_dn)}")
        fpath = Path(f"{output}/{histo.name}.txt")
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w+") as ofile:
            ofile.write("ErrorType Percent \n")
            ofile.write(f"BinLabers {bin_labels} \n")
            ofile.write(f"Binning {binning} \n")
            ofile.write(f"Nominal {nominal} \n")
            ofile.write(f"StatData {stats} \n")
            for syst_line in syst_list:
                ofile.write(f"{syst_line} \n")
            ofile.write(f"StatCorrData {nbin} \n")
            for row in corr.astype(str):
                ofile.write(f"{' '.join([x for x in row[1:]])} \n")

    def _from_text_style_1(self, txt_file, histo):
        """
        parsing text with the following format:
            BinLabels
            Binning    0        100      3000
            Nominal    11805.2  15557
            ErrorType  Percent
            Stat       0.0241   0.0138
            Jet        0.0008   0.0208
            Lepton     0.0051   0.0194
            Total      0.0491   0.0638
        """
        parsed_input = {}
        with open(txt_file, "r") as f:
            for line in f.readlines()[1:]:
                split_line = line.split()
                if "ErrorType" in split_line:
                    continue
                if "Total" in split_line:
                    continue
                if "Binning" in split_line:
                    data = np.array([float(x) for x in split_line[1:-1]])
                    parsed_input["nbin"] = data
                    continue
                data = np.array([0.0] + [float(x) for x in split_line[1:]])
                if "Nominal" in split_line:
                    parsed_input["bin_content"] = data
                else:
                    parsed_input[split_line[0]] = data
        histo.bin_content = parsed_input.pop("bin_content")
        histo.sumW2 = (parsed_input.pop("Stat") * histo.bin_content) ** 2
        histo.nbin = parsed_input.pop("nbin")

        # systematic
        for name, content in parsed_input.items():
            syst_args = {
                "type": "averaged",
                "name": name,
                "shape": content.shape,
                "components": {
                    "up": {(name, name, "up"): content},
                    "down": {(name, name, "down"): content},
                },
            }
            syst_band = SystematicsBand.loads(syst_args)
            histo.update_systematic_band(syst_band)

        return histo

    def to_text(self, histo, *, style=1, **kwargs):
        if style == 1:
            return self._to_text_style_1(histo, **kwargs)
        else:
            raise ValueError(f"Invalid text style {style}")

    def from_text(self, txt_file, histo, *, style=1, **kwargs):
        if style == 1:
            return self._from_text_style_1(txt_file, histo, **kwargs)
        else:
            raise ValueError(f"Invalid text style {style}")

    def to_combiner_yaml(
        self,
        output,
        histograms,
        bootstrap_file,
        avg=None,
        rescale=False,
        statsCovMatrix=None,
        general=None,
        do_components=True,
        decorrelate=None,
        decorrelate_all=False,
        exclude_components=None,
        ranking=True,
        set_exclude_zero=True,
        use_average=True,
        impact_type=1,
        symmetrize=True,
        use_fullname=False,
        start_index=1,
        end_index=-1,
        max_index=None,
        constraint=1,
        constraint_dict=None,
        generate_np_correlation=False,
        select_list=None,
        removal_list=None,
        sym_list=None,
        exclude_syst_group=None,
        statOnly=False,
        sym_type=1,
        filter_zero=False,
    ):
        """
        Generate yaml file for the Combiner package:
        https://gitlab.cern.ch/tdado/combiner

        Args:
            output (str or PosixPath):
                path for the yaml file.

            histograms (dict{str:collinearw.core.Histogram}):
                dictionary of histograms. The key is used to locate the tree in
                the bootstrap root file.

            bootstrap_file (str):
                path to the bootstrap root file.
        """

        temp_hist = next(iter(histograms.values()))
        nbins = len(temp_hist.bin_content)
        hist_name = temp_hist.name
        end_b = end_index
        max_b = max_index or nbins - 1
        pois = [f"bin{x}" for x in range(start_index, max_b)]

        if decorrelate is None:
            decorrelate = []
        decorrelated_syst = set(decorrelate)

        if constraint_dict is None:
            constraint_dict = {}

        # prepare general block
        general_block = {
            "general": {
                "outputPath": "./",
                "minos": True,
                "statOnly": statOnly,
                "debug": False,
                # "CorrelatedNPs": True,
                # "inversionPrecision": 1e-5,
                # "fitStrategy": 2,
                # "removeNonConstrainedCovarianceTerms": False,
                "threads": 8,
                "parameterRange": [{"parameter": x, "min": 0, "max": 10} for x in pois],
            }
        }
        if general:
            general_block["general"].update(general)

        if ranking:
            general_block["general"]["ranking"] = True

        # prepare mesurements block
        measurements_name = [{"name": f"Meas_{i}"} for i in histograms]
        measurements = {"measurements": measurements_name}

        meta_data = {}

        if not rescale or avg is None:
            avg = np.sum([h.bin_content for h in histograms.values()], axis=0)
            avg /= len(histograms)
            avg = np.nan_to_num(avg)
            if not rescale:
                # update parameters ranges
                temp_diff = np.abs(avg - temp_hist.bin_content)
                new_range = []
                for i, x in enumerate(pois, start=start_index):
                    new_min = float(avg[i] - temp_diff[i] * 3)
                    new_max = float(avg[i] + temp_diff[i] * 3)
                    # new_min = float(np.min([h.bin_content[i] for h in histograms.values()]))
                    # new_max = float(np.max([h.bin_content[i] for h in histograms.values()]))
                    new_range.append({"parameter": x, "min": new_min, "max": new_max}),
                general_block["general"]["parameterRange"] = new_range
                avg = np.ones(nbins)

        meta_data["avg"] = avg[start_index:end_b]

        up_checker = {"up", "symmetrize_up", "max", "std_up"}
        dn_checker = {"down", "symmetrize_down", "min", "std_down"}

        # prepare block for each measurement
        meas_data = {}
        # template style 1
        for key, hist in histograms.items():
            meta_data[key] = {}
            hist_dict = {}
            hist_dict["poiNames"] = pois[:]
            hist_dict["bootstrapReplicaFile"] = bootstrap_file
            hist_dict["bootstrapReplicaTree"] = f"Meas_{key.replace('/', '_')}"
            hist_dict["bootstrapReplicaVariables"] = [
                {"parameter": f"bin{x}", "branchName": f"bin{x}"}
                for x in range(start_index, max_b)
            ]
            values = np.nan_to_num(hist.bin_content / avg)
            hist_dict["values"] = values[start_index:end_b].tolist()
            # prepare systematic
            hist_dict["systematics"] = []
            zero_impact = set()
            if hist.systematic_band is None:
                syst_band = {}
            elif do_components:
                syst_band = {}
                for bandname, band in hist.systematic_band.items():
                    if bandname in exclude_syst_group:
                        continue
                    for comp_band in band.components_as_bands(filter_zero=filter_zero):
                        if isinstance(comp_band.name, tuple):
                            if use_fullname:
                                _name = ",".join(comp_band.name)
                            else:
                                _name = comp_band.name[0]
                            _name_0 = comp_band.name[0]
                            if removal_list and _name_0 in removal_list:
                                continue
                            if select_list and _name_0 not in select_list:
                                continue
                            if comp_band.name[2]:
                                idn_check = comp_band.name[2]
                            else:
                                idn_check = comp_band.name[1]
                            if idn_check in up_checker:
                                _idn = "up"
                            elif idn_check in dn_checker:
                                _idn = "dn"
                            else:
                                _idn = "nosym"
                            comp_band_name = f"{_name}:{_idn}"
                            exc_name_check = comp_band.name[0]
                        else:
                            if select_list and comp_band.name not in select_list:
                                continue
                            comp_band_name = f"{comp_band.name}:nosym"
                            exc_name_check = comp_band.name
                        # temperoraly exclusion method
                        if exclude_components:
                            if exc_name_check in exclude_components[key]:
                                if not set_exclude_zero:
                                    continue
                                zero_impact.add(comp_band.name)
                        if decorrelate_all or bandname in decorrelated_syst:
                            syst_name = f"{bandname}:{key}:{comp_band_name}"
                        else:
                            syst_name = f"{bandname}::{comp_band_name}"
                        # check for EL/MU channel specific removal
                        if removal_list and syst_name in removal_list:
                            continue
                        syst_band[syst_name] = comp_band
            else:
                syst_band = hist.systematic_band
                if exclude_syst_group:
                    for _name in exclude_syst_group:
                        syst_band.pop(_name)
                if decorrelate_all or decorrelated_syst:
                    for _name in list(syst_band.keys()):
                        if _name in removal_list:
                            syst_band.pop(_name)
                            continue
                        if decorrelate_all or _name in decorrelated_syst:
                            new_name = f"{_name}:{key}:{_name}:nosym"
                        else:
                            new_name = f"{_name}::{_name}:nosym"
                        syst_band[new_name] = syst_band.pop(_name)
                else:
                    for _name in list(syst_band.keys()):
                        new_name = f"{_name}::{_name}:nosym"
                        syst_band[new_name] = syst_band.pop(_name)
            if symmetrize:
                sym_group = collections.defaultdict(list)
                for name, _band in syst_band.items():
                    group, m_key, comp_name, idn = name.split(":")
                    if idn != "dn" and idn != "up":
                        if not do_components:  # testing with average
                            # if "ttbar" in group:
                            #     breakpoint()
                            abs_up = np.abs(_band.up)
                            abs_dn = np.abs(_band.down)
                            imp = np.where(abs_up > abs_dn, abs_up, abs_dn)
                            # imp = np.abs(imp)
                            imp *= np.where(abs_up > abs_dn, 1, -1)
                        elif group in sym_list or comp_name in sym_list:
                            imp = _band.up  # assume dn is the same
                        elif np.all(np.abs(_band.up - _band.down) < 1e-5):
                            imp = _band.up
                        else:
                            imp = _band.up - _band.down
                            # imp = _band.down - _band.up
                        _band.up = imp
                        _band.down = imp
                        continue
                    sym_group[(group, m_key, key, comp_name)].append(_band)
                for name, up_dn in sym_group.items():
                    assert len(up_dn) == 2
                    group, m_key, _, comp_name = name
                    syst_up, syst_dn = up_dn
                    syst_real_up = syst_up.up - syst_up.down
                    syst_real_dn = syst_dn.up - syst_dn.down
                    up_name = f"{group}:{m_key}:{comp_name}:up"
                    dn_name = f"{group}:{m_key}:{comp_name}:dn"
                    new_name = f"{group}:{m_key}:{comp_name}:sym"
                    old_up = syst_band.pop(up_name)
                    syst_band.pop(dn_name)
                    if sym_type == 1:
                        # sign = np.where(np.abs(syst_real_up) < np.abs(syst_real_dn), syst_real_up, syst_real_dn)
                        sign_mask = np.abs(syst_real_up) >= np.abs(syst_real_dn)
                        # sign_mask = np.abs(syst_real_up) <= np.abs(syst_real_dn)
                        imp = np.where(sign_mask, syst_real_up, syst_real_dn)
                    else:
                        sign_mask = np.abs(syst_real_up) >= np.abs(syst_real_dn)
                        _avg = (np.abs(syst_real_up) + np.abs(syst_real_dn)) * 0.5
                        imp = np.where(sign_mask, _avg, -1.0 * _avg)
                    old_up.up = imp
                    old_up.down = imp
                    if imp.sum() == 0:
                        continue
                    syst_band[new_name] = old_up
            meta_data[key]["nom"] = hist.bin_content[start_index:max_b]
            meta_data[key]["nom/avg"] = values[start_index:max_b]
            meta_data[key]["syst_up"] = {}
            meta_data[key]["syst_dn"] = {}
            meta_data[key]["impact_up"] = {}
            meta_data[key]["impact_dn"] = {}
            meta_data[key]["impact_up/avg"] = {}
            meta_data[key]["impact_dn/avg"] = {}
            for syst_name, syst in syst_band.items():
                '''
                we want to recalcuated the impact with respect to the
                pre-calculated average (or any scale factors used in rescaling
                all of the channels), namely (syst - avg) / avg
                start with ((syst-nominal)/nominal * nominal + nominal - avg) / avg
                (syst-nominal)/nominal is given by syst.average()
                this reduces to
                = (syst.average()*nominal + nominal - avg) / avg
                = ((syst.average()+1)*nominal - avg) / avg
                = (syst.average()+1)*nominal/avg - 1.0
                Alt:
                = ((syst-nominal)/nominal * nominal) / avg
                Note:
                up_band = (syst - nominal) / nominal, up_banb[up_band<0] = 0
                dn_band = (syst - nominal) / nominal, dn_band[dn_band>0] = 0
                so up_band and dn_band is always positive (or absolute values).
                syst.average() is just (up_band + dn_band) / 2
                '''
                # handle uder/overflow impact
                # impact = (syst.average() + 1.0) * values - 1.0
                # impact = (syst.average()) * values * np.where(values<avg, 1, -1)
                _, _, comp_name, _ = syst_name.split(":")
                _const = constraint_dict.get(comp_name, constraint)

                if use_average:
                    if syst.name in zero_impact:
                        impact = np.zeros(values.shape)
                    else:
                        if impact_type == 1:
                            impact = syst.average() * values
                        elif impact_type == 2:
                            sign = np.where((values * avg) < avg, -1, 1)
                            impact = syst.average() * values * sign
                        elif impact_type == 3:
                            _nom = hist.bin_content
                            _syst = syst.average() * _nom + _nom
                            sign = np.where(_syst < avg, -1, 1)
                            impact = syst.average() * values * sign
                        elif impact_type == 4:
                            _up_diff = syst.up * values
                            _dn_diff = syst.down * values * -1.0
                            impact = _up_diff + _dn_diff
                        elif impact_type == 5:
                            impact = syst.up * values
                        elif impact_type == 6:
                            impact = np.abs(syst.up * values)
                        else:
                            raise ValueError(f"invalid impact type {impact_type}")
                    syst_dict = {
                        # Combiner/Roofit doesn't like space in name
                        "parameter": f"{syst_name.replace(' ', '_')}:avg",
                        "impact": impact[start_index:max_b].tolist(),
                        "pull": 0.0,
                        "constraint": _const,
                    }
                    hist_dict["bootstrapReplicaVariables"] = [
                        {"parameter": f"bin{x}", "branchName": f"bin{x}"}
                        for x in range(start_index, max_b)
                    ]

                    hist_dict["bootstrapReplicaVariables"]
                    hist_dict["systematics"].append(syst_dict)
                    meta_data[key]["syst_up"][syst_name] = (
                        1.5 * hist.bin_content[start_index:max_b]
                    )
                    meta_data[key]["syst_dn"][syst_name] = (
                        0.5 * hist.bin_content[start_index:max_b]
                    )
                    meta_data[key]["impact_up"][syst_name] = (
                        0.5 * hist.bin_content[start_index:max_b]
                    )
                    meta_data[key]["impact_dn"][syst_name] = (
                        -0.5 * hist.bin_content[start_index:max_b]
                    )
                    meta_data[key]["impact_up/avg"][syst_name] = impact
                    meta_data[key]["impact_dn/avg"][syst_name] = impact * -1.0
                else:
                    if syst.name in zero_impact:
                        impact_up = np.zeros(values.shape)
                        impact_dn = np.zeros(values.shape)
                    else:
                        impact_up = syst.up * values
                        impact_dn = syst.down * values * -1.0
                    syst_dict = {
                        # Combiner/Roofit doesn't like space in name
                        "parameter": f"{syst_name.replace(' ', '_')}:up",
                        "impact": impact_up[start_index:max_b].tolist(),
                        "pull": 0.0,
                        "constraint": 1.0,
                    }
                    hist_dict["systematics"].append(syst_dict)
                    syst_dict = {
                        # Combiner/Roofit doesn't like space in name
                        "parameter": f"{syst_name.replace(' ', '_')}:dn",
                        "impact": impact_dn[start_index:max_b].tolist(),
                        "pull": 0.0,
                        "constraint": 1.0,
                    }
                    hist_dict["systematics"].append(syst_dict)
            # prepare correlation between systematics
            if generate_np_correlation:
                syst_correlation = []
                n_systband = len(syst_band) if use_average else len(syst_band) * 2
                for i_syst in range(1, n_systband + 1):
                    corr_row = [0] * i_syst
                    corr_row[-1] = 1
                    syst_correlation.append({"row": corr_row})
                if use_average:
                    np_name = [f"{x}:avg" for x in list(syst_band.keys())]
                else:
                    np_name = [f"{x}:up" for x in list(syst_band.keys())]
                    np_name += [f"{x}:dn" for x in list(syst_band.keys())]
                hist_dict["nuisanceParameters"] = {
                    "names": np_name,
                    "correlationMatrix": syst_correlation,
                }
            if statsCovMatrix is not None:
                if isinstance(statsCovMatrix[key], str):
                    with open(statsCovMatrix[key], "rb") as f:
                        icov = np.load(f)
                else:
                    icov = statsCovMatrix[key]
                full_cov_check = np.all(np.linalg.eigvals(icov) > 0)
                cov_check = np.all(
                    np.linalg.eigvals(icov[start_index:-1, start_index:-1]) > 0
                )
                if not full_cov_check and not cov_check:
                    logger.warning(f"Cov. is NOT POS. Def: {hist_name}")
                cov_m = np.tril(icov)
                rows = []
                for b in range(start_index, max_b):  # range(1, cov_m.shape[0] - 1)
                    # rows.append({"row": cov_m[b][: b + 1][1:-1].tolist()})
                    rows.append({"row": cov_m[b][1:-1][:b].tolist()})
                hist_dict["statCovMatrix"] = rows
                # hist_dict["stats_error"] = diag_ele.tolist()
            meas_data[f"Meas_{key}"] = hist_dict

        data = [general_block, measurements, meas_data]

        fpath = Path(output)
        fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(fpath, "w+") as ofile:
            for datum in data:
                ofile.write(yaml.dump(datum, default_flow_style=None))
                ofile.write("\n")

        # with open("meta.txt", "w+") as ofile:
        #     for key in meta_data:
        #         ofile.write(f"{hist.bin_content}")

    def from_combiner_fit(
        self,
        histogram,
        fit_result_path,
        fit_stats="Parameters_statOnly.txt",
        fit_result="Parameters.txt",
        ranking=True,
        start_index=1,
        max_index=None,
    ):
        """
        Parsing the Combiner fit result back to the Histogram object

        Args:
            histogram (core.Histogram):
                Instance of core.Histogram for storing the Combiner results.
                The histogram bin content should be the scaling used for rescaling
                the input for Combiner. e.g. if the average is used for scaling,
                the histogram bin content should be the average.

            fit_result_path (str):
                directory path to the Combiner fit results.

        Return:
            histogram object & dict of pull of NP
        """
        nbins = len(histogram.bin_content)
        # max_b = max_index or nbins - 1
        max_b = max_index or nbins
        if nbins == 3 and max_index is None:
            max_b = nbins - 1

        isyst_path = f"{fit_result_path}/{fit_result}"
        istats_path = f"{fit_result_path}/{fit_stats}"
        fit_syst = np.loadtxt(isyst_path, dtype=str, delimiter=" ", usecols=range(4))
        fit_stats = np.loadtxt(istats_path, dtype=str, delimiter=" ", usecols=range(4))
        rank_files = {}
        rank_fit = {"up": {}, "dn": {}}
        single_syst = False
        if ranking:
            for i in range(start_index, max_b):
                rank_path = f"{fit_result_path}/Ranking_bin{i}.txt"
                rank_files[f"bin{i}"] = np.loadtxt(
                    rank_path, dtype=str, delimiter=" ", usecols=range(5)
                )
            try:
                names = rank_files[f'bin{start_index}'][:, 0]
            except IndexError:
                names = [rank_files[f'bin{start_index}'][0]]
                single_syst = True
            for name in names:
                rank_fit["up"][name] = np.zeros(nbins)
                rank_fit["dn"][name] = np.zeros(nbins)
        up_syst = np.zeros(nbins)
        dn_syst = np.zeros(nbins)
        # histogram.bin_content[[0, -1]] = 0.0
        # histogram.sumW2[[0, -1]] = 0.0
        histogram.bin_content[0] = 0.0
        histogram.sumW2[0] = 0.0
        for i in range(start_index, max_b):
            loc = np.where(fit_syst == f"bin{i}")[0]
            stats_loc = np.where(fit_stats == f"bin{i}")[0]
            fitted = float(fit_syst[loc][0, 1])  # fitted mu
            fitted_dn = float(fit_syst[loc][0, 2])  # mu_minus - mu
            fitted_up = float(fit_syst[loc][0, 3])  # mu_plus - mu
            if fit_stats.ndim == 1:
                # fitted_stats = float(fit_stats[1])
                stats_dn = float(fit_stats[2])
                stats_up = float(fit_stats[3])
            else:
                # fitted_stats = float(fit_stats[stats_loc][0, 1])
                stats_dn = float(fit_stats[stats_loc][0, 2])
                stats_up = float(fit_stats[stats_loc][0, 3])
            # stats_avg = (stats_up - stats_dn) * 0.5 * fitted_stats
            stats_avg = (stats_up - stats_dn) * 0.5
            histogram.bin_content[i] *= fitted
            histogram.sumW2[i] = stats_avg * stats_avg
            dn_syst[i] = fitted_dn / fitted
            up_syst[i] = fitted_up / fitted
            if not rank_files:
                continue
            for name in rank_fit['up']:
                if single_syst:
                    if histogram.name == "DeltaRTruthLepJetClosest100":
                        print(f"bin{i} {rank_files[f'bin{i}'][1:]}")
                    _up = float(rank_files[f"bin{i}"][3])
                    _dn = float(rank_files[f"bin{i}"][4])
                else:
                    loc = np.where(rank_files[f"bin{i}"] == name)[0]
                    _up = float(rank_files[f"bin{i}"][loc][0, 3])
                    _dn = float(rank_files[f"bin{i}"][loc][0, 4])
                rank_fit['up'][name][i] = abs(_up / fitted)
                rank_fit['dn'][name][i] = abs(_dn / fitted)
        # histogram._systematic_band = None  # removing existing systematic
        hist_bands = histogram.systematic_band
        nouse_syst = list(hist_bands.keys())
        for _band in hist_bands.values():
            _band.clear()
        band = SystematicsBand("total", "Combiner", up_syst.shape)
        band.up = abs(up_syst)
        band.down = abs(dn_syst)
        # band.add_component("up", "Combiner", up_syst)
        # band.add_component("down", "Combiner", dn_syst)
        if rank_files:
            for name in rank_fit["up"]:
                # format after split group:channel:component:idn:avgtype
                group, _, comp_name, idn, up_dn = name.split(":")
                # band.add_component("up", name, rank_fit["up"][name])
                # band.add_component("down", name, rank_fit["dn"][name])
                if group in hist_bands:
                    if idn == "up" or up_dn == "up":
                        hist_bands[group].add_component(
                            "up", comp_name, rank_fit["up"][name]
                        )
                    elif idn == "dn" or up_dn == "dn":
                        hist_bands[group].add_component(
                            "down", comp_name, rank_fit["dn"][name]
                        )
                    elif idn == "sym" or up_dn == "avg":
                        hist_bands[group].add_component(
                            "up", f"{group}/{comp_name}", rank_fit["up"][name]
                        )
                        hist_bands[group].add_component(
                            "down", f"{group}/{comp_name}", rank_fit["dn"][name]
                        )
                    else:
                        raise ValueError(f"cannot resolve {up_dn}")
                    try:
                        nouse_syst.remove(group)
                    except ValueError:
                        pass
            if nouse_syst:
                logger.info(f"No use systematics: {nouse_syst}")
                for _nouse in nouse_syst:
                    del hist_bands[_nouse]
            for _b in hist_bands.values():
                band.update_sub_bands(_b, copy=False)
        histogram.update_systematic_band(band)

        # processing fitted NP
        np_pulls = {}
        for line in fit_syst:
            if "bin" in line[0]:
                continue
            group, ch, comp_name, idn, up_dn = line[0].split(":")
            if ch is None:
                ch = "avg"
            if ch not in np_pulls:
                np_pulls[ch] = {}
            np_pulls[ch][comp_name] = line[1:].astype(np.float64)

        # dict of ranking (integrated all bins)
        int_rank = {"up": {}, "dn": {}, "int": 0.0}
        if ranking:
            for name in rank_fit["up"]:
                group, ch, comp_name, idn, up_dn = name.split(":")
                _up = rank_fit["up"][name] * histogram.bin_content
                _dn = rank_fit["dn"][name] * histogram.bin_content
                int_rank["up"][comp_name] = _up.sum()
                int_rank["dn"][comp_name] = _dn.sum()
            int_rank["int"] = histogram.bin_content.sum()

        return histogram, np_pulls, int_rank

    def to_bootstrap_root_file(
        self, output, histograms, scale_factor=False, merge_overflow=False
    ):
        with uproot.recreate(output) as f:
            for key, hist in histograms.items():
                # ttree filling
                replica_data = {}
                last_bin = len(hist.bin_content)
                if merge_overflow:
                    last_bin -= 1
                    hist.replica[:, -2] += hist.replica[:, -1]
                # NOTE: the underflow bin might have problem with combiner when it's 0
                for x in range(0, last_bin):
                    replica_data[f"bin{x}"] = hist.replica[:, x]
                    if scale_factor:  # scale by its own nominal bin
                        replica_data[f"bin{x}"] /= hist.bin_content[x]
                f[f"Meas_{key}"] = replica_data

    def to_dict(self, histo):
        histo_dict = {}
        histo_dict["name"] = histo.name  # should be unique identifier
        # metadata infomation need for recontructing the histogram object
        histo_dict["metadata"] = {
            "nbin": histo.nbin,
            "xmin": float(histo.xmin),
            "xmax": float(histo.xmax),
            "xtitle": histo.xtitle,
            "ytitle": histo.ytitle,
            "backend": histo.backend,
        }
        # bin edges and content of the histograms
        histo_dict["data"] = {
            "bin_content": histo.bin_content.tolist(),
            "sumW2": histo.sumW2.tolist(),
            "bins": histo.bins.tolist(),
        }
        if histo.systematic_band:
            syst_band = {}
            for name, band in histo.systematic_band.items():
                syst_band[name] = {"up": band.up.tolist(), "down": band.down.tolist()}
                if band.sub_bands:
                    syst_band[name]["sub_bands"] = {}
                    for sub_name, sub_band in band.sub_bands.items():
                        syst_band[name]["sub_bands"][sub_name] = {
                            "up": sub_band.up.tolist(),
                            "down": sub_band.down.tolist(),
                        }
                else:
                    syst_band[name]["sub_bands"] = None
            histo_dict["systematic"] = syst_band
        else:
            histo_dict["systematic"] = None

        return histo_dict

    def from_mcfm_output(self, input):
        with open(input) as f:
            lines = f.readlines()
        data = {}
        data["name"] = lines[0].split()[1]
        data["underflow"] = [float(x) for x in lines[1].split()[2:]]
        data["overflow"] = [float(x) for x in lines[2].split()[2:]]
        # lines[3].split()
        # ['#', 'sum', '623.42772', '1.9081501']
        # lines[4].split()
        # ['#', 'xmin', 'xmax', 'cross', 'numerror']
        bin_edges = []
        bin_cross = []
        bin_error = []
        for line in lines[5:]:
            ldata = line.split()
            # bin_edges += [float(x) for x in ldata[:2]]
            bin_edges.append(float(ldata[0]))
            bin_cross.append(float(ldata[2]))
            bin_error.append(float(ldata[3]))
        bin_edges.append(float(lines[-1].split()[1]))
        histo = Histogram.variable_bin(data["name"], bin_edges, data["name"])
        histo.bin_content[1:-1] = bin_cross
        histo.sumW2[1:-1] = np.array(bin_error) ** 2
        histo.bin_content[[0, -1]] = [data["underflow"][0], data["overflow"][0]]
        histo.sumW2[[0, -1]] = [data["underflow"][0] ** 2, data["overflow"][0] ** 2]
        return histo
