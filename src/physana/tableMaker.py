'''
class for making tables
'''

from tabulate import tabulate
from pathlib import Path
import logging
import pandas
import numpy as np
import copy

from .configs import ConfigMgr
from .histo import Histogram
from .strategies.unfolding.utils import get_xsec_uncert

logging.basicConfig()
logger = logging.getLogger(__name__)


class SystUPDN:
    def __init__(self, up, down):
        self.up = up
        self.down = down


class TableMaker(object):
    def __init__(self, output_dir=".", save_tag="tables"):
        self.output_dir = f"{output_dir}/{save_tag}/"
        self.syst_percentage = True

    def getYieldAndError(
        self,
        hist,
        format="latex",
        underflow=False,
        syst=None,
        symmtrize=True,
        include_stats=True,
        precision=".0f",
    ):
        syst = syst or hist.total_band(include_stats=include_stats)
        if underflow:
            hist_content = hist.bin_content
            sumW2 = hist.sumW2
            if syst:
                syst_up = syst.up
                syst_dn = syst.down
        else:
            hist_content = hist.bin_content[1:]
            sumW2 = hist.sumW2[1:]
            if syst:
                syst_up = syst.up[1:]
                syst_dn = syst.down[1:]
        integral = np.sum(hist_content)
        error = np.sqrt(np.sum(sumW2))
        if syst:
            syst_up = np.sum(syst_up * hist_content)
            syst_dn = np.sum(syst_dn * hist_content)
        if format.lower() == "latex":
            if syst:
                if symmtrize:
                    avg_syst = (syst_up + syst_dn) * 0.5
                    if self.syst_percentage:
                        syst_part = (
                            f' \\pm {avg_syst:{precision}} [{avg_syst/integral:.2f}]'
                        )
                    else:
                        syst_part = f' \\pm {avg_syst:{precision}}'
                else:
                    syst_part = f"\\substack{{+{syst_up:{precision}} \\\\ -{syst_dn:{precision}}}}"
            else:
                syst_part = None
            if syst_part is None:
                val = f'${integral:{precision}}\\pm {error:{precision}}$'
            elif not include_stats:
                val = f'${integral:{precision}}{syst_part} \\pm {error:{precision}}$'
            else:
                val = f'${integral:{precision}}{syst_part}$'
        elif format.lower() == "unicode":
            pm = u'\u00B1'
            val = [f'{integral:{precision}}', f'{error:{precision}}']
            if syst:
                val += [f'{syst_up:{precision}}', f'{syst_dn:{precision}}']
            val = f"{pm}".join(val)
        else:
            val = (round(integral, 2), round(error, 2))
            if syst:
                val = (*val, round(syst_up, 2), round(syst_dn, 2))
        return val

    def makeTables(
        self,
        configMgr,
        regions=None,
        processes=None,
        processNameMap=None,
        regionNameMap=None,
        excludeProcesses=None,
        obs=None,
        signal="",
        systematic=None,
    ):
        theaders = ["Process"]
        prow_bkgd_sum = ['Total SM background']
        prow_sum = ['Total SM']
        prow_data = ['Observed data']
        table = []
        saveName = []
        sumHists = {}
        bkgdSumHist = {}
        sumSyst = {}

        if obs is None:
            obs = configMgr.histograms[0].name
        if processNameMap is None:
            processNameMap = {}
        if regionNameMap is None:
            regionNameMap = {}

        # loop through both regions ans aux_regions
        region_list = configMgr.regions + configMgr.aux_regions
        # User selected regions
        if regions:
            name_dict = {x.name: x for x in region_list if x.name in regions}
            region_list = []
            for name in regions:
                x = name_dict.get(name, None)
                if x:
                    region_list.append(x)

        # First do all mc processes
        for p in configMgr.processes:
            if processes and p.name not in processes:
                continue

            if excludeProcesses and p.name in excludeProcesses:
                continue

            pname = processNameMap.get(p.name, p.name)
            prow = [pname]

            for r in region_list:
                # Regions should have at least one histogram
                # doesn't matter which one we take
                if len(configMgr.histograms) == 0:
                    continue

                # Name to be displayed at the top of the table
                if not r.name in saveName:
                    table_region_name = regionNameMap.get(r.name, r.name)
                    saveName.append(r.name)
                    theaders.append(table_region_name)

                # Integrate histogram and include overflow bins
                print(f"Process: {p.name} and region {r.name}")
                hist = configMgr.get(p.name).get(systematic).get(r.name).get(obs)
                hist.nan_to_num()

                # Add region, and keep track of total region yields
                if p.type == "mc" or p.type == "fakes":
                    if r.name in sumHists:
                        sumHists[r.name].add(hist)
                    else:
                        sumHists[r.name] = hist.copy(noparent=True)
                    prow.append(self.getYieldAndError(hist))
                    if p.name == signal:
                        continue
                    if r.name in bkgdSumHist:
                        bkgdSumHist[r.name].add(hist)
                    else:
                        bkgdSumHist[r.name] = hist.copy(noparent=True)
                    if sumSyst.get(r.name, None) is None:
                        totb = hist.total_band(include_stats=True)
                        if totb is not None:
                            syst_up = (totb.up * hist.bin_content) ** 2
                            syst_dn = (totb.down * hist.bin_content) ** 2
                            sumSyst[r.name] = SystUPDN(syst_up, syst_dn)
                    else:
                        totb = hist.total_band(include_stats=True)
                        if totb is not None:
                            syst_up = (totb.up * hist.bin_content) ** 2
                            syst_dn = (totb.down * hist.bin_content) ** 2
                            sumSyst[r.name].up += syst_up
                            sumSyst[r.name].down += syst_dn
                else:
                    prow_data.append(self.getYieldAndError(hist))

            if p.type == "mc" or p.type == "fakes":
                table.append(prow)

        prow_sum_temp = []
        for x, h in sumHists.items():
            if x not in sumSyst:
                prow_sum_temp.append(self.getYieldAndError(h, syst=None))
                continue
            sumSyst[x].up = np.nan_to_num(np.sqrt(sumSyst[x].up) / h.bin_content)
            sumSyst[x].down = np.nan_to_num(np.sqrt(sumSyst[x].down) / h.bin_content)
            prow_sum_temp.append(self.getYieldAndError(h, syst=sumSyst[x]))
        prow_sum += prow_sum_temp

        prow_bkgd_sum += [self.getYieldAndError(h) for h in bkgdSumHist.values()]

        table.append(prow_bkgd_sum)
        table.append(prow_sum)
        table.append(prow_data)

        output = Path(f"{self.output_dir}/Table_{'_'.join(saveName)}.tex")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w+") as f:
            f.write(
                tabulate(table, headers=theaders, tablefmt="latex_raw", floatfmt=".%2f")
            )

    # ==========================================================================
    # ==========================================================================
    def ABCDTable(
        self,
        config,
        *,
        processNameMap={},
        regionNameMap={},
        tagNameMap={},
        type="data_mc",
        yield_format='unicode',
    ):
        """
        Create pandas dataframe for ABCD study.
        It only considers MC and data processes,
        and the fake estimation will compute independently in here (non-bin).
        No need to pass the config object to histmanipulate for TF and fakes.

        Args:
            config := configMgr object or path to the configMgr file.

        Optional Args:

            processNameMap := dict for renaming the processes within the config.

            regionNameMap := similar to processNameMap, but it's for region rename

            abcd_tag := abcd tag inside the config object. use for lookup ABCD regions

        Return:

            dict of pandas.dataframe

        Notes:

            the dataframe layout:

            process | region-tag 1  | region-tag 2
            -----------------------------------------
                    | A | B | C | D | A | B | C | D |
            -----------------------------------------
            data    | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
            -----------------------------------------
        """

        if isinstance(config, ConfigMgr):
            pass
        else:
            config = ConfigMgr.open(config)

        # internal function for division and multiplication
        def _divide(v1, v2):
            value = v1[0] / v2[0] if v2[0] != 0 else 0
            error = (
                np.sqrt(v1[1] / v2[0] ** 2 + v2[1] / v2[0] ** 4) if v2[0] != 0 else 0
            )
            return (value, error)

        def _multiply(v1, v2):
            value = v1[0] * v2[0]
            error = np.sqrt((v1[1] * v2[0]) ** 2 + (v1[0] * v2[1]) ** 2)
            return (value, error)

        def _sum(v1, v2):
            value = v1[0] + v2[0]
            error = np.sqrt(v1[1] ** 2 + v2[1] ** 2)
            return (value, error)

        def _v_divide(v1_list, v2_list):
            return [_divide(v1, v2) for v1, v2 in zip(v1_list, v2_list)]

        def _v_multiply(v1_list, v2_list):
            return [_multiply(v1, v2) for v1, v2 in zip(v1_list, v2_list)]

        def _v_sum(v1_list, v2_list):
            return [_sum(v1, v2) for v1, v2 in zip(v1_list, v2_list)]

        _round = lambda x: (round(x[0], 2), round(x[1], 2))
        _v_round = lambda x: [_round(_x) for _x in x]

        abcd_meta = config.meta_data["abcd"]
        df_dict = {}
        for tag in abcd_meta.keys():
            # loop through ABCD base level region
            for case in abcd_meta[tag].cases:
                has_fake = False
                has_data = False
                all_yield = {}

                all_yield["mc"] = {}

                # header for columns and rows
                abcd_col = ["A", "B", "C", "D"]
                region_col = []

                # check name map. This reformat the name in the config.
                base_region = regionNameMap.get(case.parent, case.parent)
                tag_name = tagNameMap.get(tag, tag)

                region_col.append(f"{base_region} : {tag_name}")

                for p in config.processes:
                    pname = processNameMap.get(p.name, p.name)

                    # check process type.
                    if p.type == "data" or pname == "data":
                        has_data = True
                    elif p.type == "fakes":
                        has_fake = True
                    else:
                        pass

                    region_yield = []
                    for r in ["A", "B", "C", "D"]:
                        rName = case.regions[r]

                        # check if region exists
                        try:
                            _region = config.get_process(p.name).get_region(rName)
                        except KeyError:
                            logger.warning(f"cannot find region {rName}")
                            break

                        # retrieve yields
                        _hist = _region.get_histogram(config.histograms[0].name).root
                        m_yield = self.getYieldAndError(_hist, "number")
                        if p.type == "fakes":
                            # in the case of fakes, fill BCD regions with zero.
                            m_fakes = [m_yield, (0, 0), (0, 0), (0, 0)]
                            all_yield[pname] = m_fakes
                            break
                        else:
                            region_yield.append(m_yield)

                    if region_yield == [] or len(region_yield) != 4:
                        continue

                    if p.type == "fakes":
                        pass
                    elif p.type == "data" or p.name == "data":
                        all_yield["data"] = region_yield
                    else:
                        all_yield["mc"][pname] = region_yield

                if has_data and type.lower() == "data_mc":
                    # calculate total MC yields
                    mc_sum = [(0, 0), (0, 0), (0, 0), (0, 0)]
                    for mc in all_yield["mc"].values():
                        mc_sum = _v_sum(mc_sum, mc)

                    data_yield = all_yield["data"]
                    # calcuate on-the-fly data and MCs comparison
                    all_yield["Data/Pred"] = _v_round(_v_divide(data_yield, mc_sum))

                    sub_data = []
                    for i, (_d, _m) in enumerate(zip(data_yield, mc_sum)):
                        value = round(_d[0] - _m[0], 2)
                        error = round(np.sqrt(_d[1] ** 2 + _m[1] ** 2), 2)
                        sub_data.append((value, error))
                    all_yield["Data-MCs"] = sub_data

                    if not has_fake:
                        # fakes
                        tf = _divide(sub_data[2], sub_data[3])
                        fakes = _multiply(tf, sub_data[1])
                        all_yield["fakes"] = [_round(fakes), (0, 0), (0, 0), (0, 0)]

                        # data / Pred+fakes
                        pred_fake = _sum(fakes, mc_sum[0])
                        data_pred_fake = _divide(
                            copy.deepcopy(data_yield[0]), pred_fake
                        )
                        all_yield["Data/(Pred+Fakes)"] = [
                            _round(data_pred_fake),
                            (0, 0),
                            (0, 0),
                            (0, 0),
                        ]

                elif type.lower() == "closure":
                    abcd_col.append("Fakes")
                    abcd_col.append("Fakes/MC")
                    for item in all_yield["mc"].values():
                        Av = item[0]
                        Bv = item[1]
                        Cv = item[2]
                        Dv = item[3]
                        tf = _divide(Cv, Dv)
                        fakes = _multiply(tf, Bv)
                        item.append(_round(fakes))
                        ratio = _divide(fakes, Av)
                        item.append(_round(ratio))
                else:
                    pass

                mux = pandas.MultiIndex.from_product([region_col, abcd_col])

                process_row = []
                data = []
                if "data" in all_yield:
                    process_row.append("data")
                    data.append(all_yield["data"])
                if "mc" in all_yield:
                    for name, value in all_yield["mc"].items():
                        process_row.append(name)
                        data.append(value)
                if "fakes" in all_yield:
                    process_row.append("fakes")
                    data.append(all_yield["fakes"])
                if "Data-MCs" in all_yield:
                    process_row.append("Data-MCs")
                    data.append(all_yield["Data-MCs"])
                if "Data/Pred" in all_yield:
                    process_row.append("Data/Pred")
                    data.append(all_yield["Data/Pred"])

                df = pandas.DataFrame(data, columns=mux, index=process_row)

                if yield_format == "unicode":
                    for i, row in df.iterrows():
                        for k, value in enumerate(row):
                            pm = u'\u00B1'
                            row[k] = f"{value[0]}{pm}{value[1]}"
                elif yield_format == "latex":
                    for i, row in df.iterrows():
                        for k, value in enumerate(row):
                            row[k] = f"${value[0]}\\pm{value[1]}$"
                else:
                    pass

                if (case.parent, tag) not in df_dict:
                    df_dict[(case.parent, tag)] = df

        return df_dict


def hist_to_dataframe(config, data, signal, bgks):
    """
    Convert set of histograms from process/region to pandas.DataFrame
    """
    output = {}

    config = ConfigMgr.open(config)

    data_proc = config.get_process(data)
    for region in data_proc.regions:
        output[region.name] = {}
        for hist in region.histograms:
            if isinstance(hist, Histogram):
                hist_df = pandas.DataFrame()
                bins = hist.bins
                hist_df["bin edge"] = pandas.Series(bins)
                hist_df[data] = pandas.Series(hist.bin_content[1:])
                for bgk in [signal] + bgks:
                    try:
                        content = config.get(
                            f"{bgk}//nominal//{region.name}//{hist.name}"
                        ).bin_content[1:]
                    except KeyError:
                        content = np.zeros(bins.shape)
                    hist_df[bgk] = pandas.Series(content)

                hist_df["signal+bkgs"] = hist_df[signal] + hist_df[bgks].sum(axis=1)
                hist_df["bgks sum"] = hist_df[bgks].sum(axis=1)
                hist_df["bgks/signal"] = hist_df["bgks sum"] / hist_df[signal]
                hist_df["data/total"] = hist_df[data] / hist_df["total"]

                for bgk in bgks:
                    hist_df[f"{bgk} norm to bkgs"] = hist_df[bgk] / hist_df["bgks sum"]

                hist_df["(data-bkg_sum)/total"] = hist_df["data"] - hist_df["bgks sum"]
                hist_df["(data-bkg_sum)/total"] /= hist_df["total"]

                output[region.name][hist.name] = hist_df


def make_histo_systematic_table(
    histo,
    header="values",
    exclude=None,
    include=None,
    combine=None,
    lumi=140.0,
    lumiunc=0.0,
):
    """
    Calculate the total cross section from a histogram, and compute the systematic
    contribution.
    """
    if histo.systematic_band is None:
        raise ValueError(f"histogram {histo.name} does not have systematic band.")

    if exclude is None:
        exclude = []
    if combine is None:
        combine = {}
    if include is None:
        include = list(combine.keys()) + ["total"]

    bands = copy.deepcopy(histo.systematic_band)

    for bname in list(bands.keys()):
        if exclude and bname in exclude:
            bands.pop(bname, None)

    for grp_name, grp in combine.items():
        if len(grp) < 1:
            logger.warning(f"cannot combine band from {grp_name}")
            continue
        temp_bands = [bands.pop(x, None) for x in grp]
        temp_bands = [x for x in temp_bands if x is not None]
        if not temp_bands:  # check if empty
            continue
        # replace grouped systematic band
        first_band = copy.deepcopy(temp_bands[0])  # use the first one to setup
        first_band.name = grp_name  # rename to the user grouped name
        for _band in temp_bands[1:]:  # combine the rest
            first_band.combine(_band)
        bands.update({grp_name: first_band})

    # update the grouping if total band already exist
    if "total" not in bands:
        total_band = histo.total_band(
            include_stats=True,
            exclude_names=exclude,
            ext_band=bands,
        )
        bands[total_band.name] = total_band
    else:
        for grp_name, grp in combine.items():
            for gname in grp:
                try:
                    bands["total"].remove_sub_band(gname)
                except KeyError:
                    logger.warning(f"total has no sub-band {gname} !!!")
            bands["total"].update_sub_bands(bands[grp_name])

    xsec, stat_error, _, _ = get_xsec_uncert(
        histo,
        lumi,
        lumiunc,
        exclude_names=exclude,
    )

    row_name = ["Cross section [fb]", "Statistical"]
    row_values = [round(xsec, 2), round(stat_error / xsec * 100, 3)]

    if include and "lumi" not in include:
        include = ["lumi"] + include

    # note that the band is (syst. - nom.) / nom.
    for bname, band in bands.items():
        if include and bname not in include:
            continue
        up_band = band.up
        dn_band = band.down
        avg_band = np.nan_to_num((up_band + dn_band) * 0.5)
        frac = np.sum(avg_band * histo.bin_content) / np.sum(histo.bin_content)
        row_name.append(bname)
        row_values.append(round(frac * 100, 3))

    syst_df = pandas.DataFrame()
    syst_df["category"] = row_name
    syst_df[header] = row_values

    return syst_df


def make_histo_systematic_table_bins(histo, *args, **kwargs):
    for i in range(len(histo.bin_content)):
        c_histo = histo.copy(noparent=True)
        ith_values = c_histo[i]
        c_histo.bin_content = np.zeros(len(histo.bin_content))
        c_histo.bin_content[i] = ith_values
        yield make_histo_systematic_table(c_histo, *args, **kwargs)
