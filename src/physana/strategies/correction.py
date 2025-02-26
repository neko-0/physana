'''
Handle correction and it's derivation.
The simultaneous iterative correction involves ABCD methods.
'''

import numpy as np
import warnings
import logging
import copy
import os
from tabulate import tabulate

try:
    import ROOT

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    ROOT.gROOT.SetBatch(True)
except ImportError:
    warnings.warn("Cannot import ROOT module!")

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
log = logging.getLogger(__name__)
usr_log = logging.getLogger(f"{__name__}_USER")
usr_log.setLevel(logging.INFO)


class PhaseSpace:
    """
    class for storing phase-space definition for derivation of correction factors.
    """

    def __init__(self, type, region, signal):
        """
        Args:
            type (str) : name of correction type

            region (list(str)) : name of the control region

            signal (list(str)) : signal process name

        """
        if len(region) != len(signal):
            raise RuntimeError("not enough variables or equations!")
        self.type = type
        self.signal = signal
        self.region = region
        self.bkg = {key: None for key in region}
        self.signal_region_map = {sig: r for sig, r in zip(signal, region)}

    def set_background(self, region, bkg):
        """
        Setting the background processes for a given region.
        """
        if region not in self.bkg:
            raise KeyError(f"cannot find {region} in {self.bkg.keys()}")
        self.bkg.update({region: bkg})


class Correction:
    """
    Class for correction derivation.
    """

    def __init__(self):
        self.corrections = []

    def add_correction(self, *args, **kwargs):
        """
        Defines a phase-space region where corrections will be derived
        Arguments follow PhaseSpace.__init__ constructor
        """
        self.corrections.append(PhaseSpace(*args, **kwargs))

    def set_background(self, region, bkg):
        """
        Wrapper for PhaseSpace.set_background.
        It does automatic region name check to assign backgrounds.
        """
        for corr in self.corrections:
            if region in corr.region:
                corr.set_background(region, bkg)
                break

    def _DeriveCorrection(self, configMgr):
        """
        Derive correction for normalization of processes based upon a set of regions

        data_region_1 = mu_bkg_1*N_bkg_1 + mu_bkg_2*N_bkg_2 + ... mu_bkg_n*N_bkg_n + other-backgrounds
        ...
        data_region_n = ...

        """

        cConfigMgr = configMgr.copy()

        base_output_path = os.path.join(configMgr.out_path, "phasespace-corr")
        try:
            os.makedirs(base_output_path)
        except FileExistsError:
            pass

        corrections = {}

        # Loop over all correction types
        for c in self.corrections:
            fProcesses = []
            fRegions = []
            for pr in self.corrections[c]:
                fProcesses.append(pr)
                fRegions.append(self.corrections[c][pr])

            usr_log.info(f"Working on correction set named {c}")

            # Loop over histograms

            corrections[c] = {}

            for histogram in configMgr.histograms:
                # Clone histogram to store the result
                usr_log.info(f"Working on histogram named {histogram.name}")

                cHists = {}
                cHists_buff = {}
                for p in fProcesses:
                    sf = copy.copy(
                        configMgr.get_process(p)
                        .get_region(fRegions[0])
                        .get_histogram(histogram.name)
                        .root
                    )
                    sf.SetLineColor(configMgr.get_process(p).color)
                    sf.SetLineWidth(2)
                    sf.SetTitle(f'correction-{p}-{histogram.name}')
                    sf.Reset()

                    cHists[p] = sf
                    cHists_buff[p] = []

                for b in range(1, histogram.nbin + 1):
                    A = []
                    B = []

                    usr_log.info(f"Working on bin number {b}")

                    for region in fRegions:
                        # Now do the calculation
                        data = 0.0
                        bkg = 0.0
                        usr_log.info(
                            f"Calculating data - background yields in region {region}"
                        )
                        for bp in configMgr.processes:
                            if bp.name in fProcesses:
                                continue
                            hist = (
                                configMgr.get_process(bp.name)
                                .get_region(region)
                                .get_histogram(histogram.name)
                                .root
                            )

                            y = hist.GetBinContent(b)
                            usr_log.info(
                                f"Found process {bp.name} with total yield {y}"
                            )
                            if bp.type == "data":
                                data += y
                            else:
                                bkg += y

                        data_subtracted = data - bkg
                        usr_log.info(f"Data subtracted yield {data_subtracted}")

                        if data_subtracted == 0:
                            data_subtracted = 1e-9
                        B.append(data_subtracted)

                        # Processes that we want to correct
                        usr_log.info("Extracting yields to float")
                        process_integrals = []
                        for p in fProcesses:
                            hist = (
                                configMgr.get_process(p)
                                .get_region(region)
                                .get_histogram(histogram.name)
                                .root
                            )

                            integral = hist.GetBinContent(b)
                            usr_log.info(
                                f"Found process {p} in region {region} with total yield {integral}"
                            )
                            if integral < 0.01:
                                integral = 1e-9

                            process_integrals.append(integral)
                        A.append(process_integrals)

                    X2 = np.linalg.solve(A, B)

                    for i, p in enumerate(fProcesses):
                        usr_log.info(f"Saving process {p} normalization factor {X2[i]}")
                        corr_factor = 1.0
                        if len(X2) >= i:
                            corr_factor = X2[i]

                        cHists[p].SetBinContent(b, corr_factor)
                        cHists_buff[p].append(corr_factor)

                corrections[c][histogram.name] = cHists

                input(cHists_buff)

                can = ROOT.TCanvas()
                ROOT.gStyle.SetOptStat(0)
                cmd = 'e0'
                # TODO: Legend, and move this into PlotMaker?
                for _hist in cHists:
                    cHists[_hist].Draw(cmd)
                    cmd = 'e0 same'
                can.SaveAs(
                    f'{base_output_path}/CorrectionFactors_{c}-{histogram.name}.pdf'
                )

        cConfigMgr.corrections = corrections
        return cConfigMgr

    def DeriveCorrection(self, configMgr):
        """
        See dervie_correction

        Depreciated:

        Derive correction for normalization of processes based upon a set of regions

        data_region_1 = mu_bkg_1*N_bkg_1 + mu_bkg_2*N_bkg_2 + ... mu_bkg_n*N_bkg_n + other-backgrounds
        ...
        data_region_n = ...

        """

        cConfigMgr = configMgr.copy()

        base_output_path = os.path.join(configMgr.out_path, "phasespace-corr")
        try:
            os.makedirs(base_output_path)
        except FileExistsError:
            pass

        corrections = {}

        # Loop over all correction types
        for c in self.corrections:
            fProcesses = []
            fRegions = []
            for pr in self.corrections[c]:
                fProcesses.append(pr)
                fRegions.append(self.corrections[c][pr])

            usr_log.info(f"Working on correction set named {c}")

            # Loop over histograms

            corrections[c] = {}

            # note this is only 1D histogram
            histogram = [(h.name, h) for h in configMgr.histograms]

            for hist_name, h in histogram:
                usr_log.info(f"Working on histogram named {hist_name}")

                cHists = {}
                for p in fProcesses:
                    sf = h.copy()
                    cHists[p] = sf

                A = []
                B = []
                for region in fRegions:
                    data = 0.0
                    bkg = 0.0

                    for bp in configMgr.processes:
                        if bp.name in fProcesses:
                            continue

                        hist = bp.get_region(region).get_histogram(hist_name).copy()

                        if bp.type == "data":
                            data = data + hist
                        else:
                            bkg = bkg + hist

                    data_subtracted = data - bkg
                    data_subtracted.bin_content[data_subtracted.bin_content == 0] = 1e-9

                    usr_log.info(
                        f"Data subtracted yield\n {data_subtracted.bin_content}"
                    )

                    B.append(data_subtracted.bin_content)

                    # Processes that we want to correct
                    usr_log.info("Extracting yields to float")
                    process_integrals = []
                    for p in fProcesses:
                        m_p = configMgr.get_process(p)
                        hist = m_p.get_region(region).get_histogram(hist_name).copy()
                        hist.bin_content[hist.bin_content < 0.01] = 1e-9
                        process_integrals.append(hist.bin_content)
                        usr_log.info(
                            f"Found process {p} in region {region} with total yield\n {process_integrals[-1]}"
                        )
                    A.append(np.transpose(process_integrals))

                usr_log.info("Arranging elements")
                A = np.stack(A, axis=1)
                B = np.array(B).transpose()

                X2 = np.linalg.solve(A, B)
                X2 = X2.T

                for i, p in enumerate(fProcesses):
                    cHists[p].bin_content = X2[i]

                corrections[c][hist_name] = cHists

                can = ROOT.TCanvas()
                leg = ROOT.TLegend()
                ROOT.gStyle.SetOptStat(0)
                cmd = 'e0'
                # TODO: Legend, and move this into PlotMaker?
                buff = []
                for p, hist in cHists.items():
                    hr = hist.root
                    hr.SetTitle(f"correction-{p}-{hist_name}")
                    hr.SetLineColor(configMgr.get_process(p).color)
                    hr.SetLineWidth(2)
                    hr.Draw(cmd)
                    buff.append(hr)
                    leg.AddEntry(hr, p)
                    cmd = 'e0 same'
                leg.Draw()
                can.SaveAs(f'{base_output_path}/CorrectionFactors_{c}-{hist_name}.pdf')

        cConfigMgr.corrections = corrections
        return cConfigMgr

    def derive_phasespace_correction(
        self,
        config,
        histograms=[],
        *,
        data_name="data",
        verbose=True,
        signal_threshold=0.01,
        err_prop=True,
        systematic=None,
    ):
        """
        method that derive phase-space correction simultaneously
        for signal and backgrounds.

        Args:
            config : ConfigMgr
                An instance of ConfigMgr class

            histograms : list(str), optional
                a list that contains 1D histogram names for correction derivation.
                if the list if empty, the names from ConfigMgr.histograms will be used.

            verbose : boolean, optional
                display the detail of the system of equations and it's solution.

            signal_threshold : float, optional
                setting the bin content to zero if below the threshold.

            err_prop: boolean, optional
                propagation of error in the solution to the system of equations.

        Returns:
            dict
                return a dictionary of correction factors. the key of the dict
                follow (correction type, signal name, histogram name)
        """
        histograms = histograms or [h.name for h in config.histograms]
        data_process = config.get_process_set(data_name).get(systematic)
        output = {}
        for corr in self.corrections:
            usr_log.info(f"working on correction type {corr.type}")
            usr_log.info(f"control regions: {corr.region}")
            usr_log.info(f"signal processes: {corr.signal}")
            if systematic:
                usr_log.info(f"with systematic: {systematic}")
            for hist_name in histograms:
                usr_log.info(f"working on histogram {hist_name}")
                A = []
                B = []
                control_region_name = []
                for control in corr.region:
                    usr_log.info(f"generating equation for {control}")
                    control_region_name.append(control)
                    signal = [
                        config.get_process_set(s).get(systematic).get_region(control)
                        for s in corr.signal
                    ]
                    coeff = []
                    for sp in signal:
                        hist = sp.get_observable(hist_name).copy()
                        hist.bin_content[hist.bin_content < signal_threshold] = (
                            0.0  # 1e-9
                        )
                        hist.nan_to_num(posinf=0, neginf=0)
                        coeff.append(hist.bin_content)

                    A.append(np.transpose(coeff))

                    sub_data = (
                        data_process.get_region(control)
                        .get_observable(hist_name)
                        .copy()
                    )
                    if corr.bkg[control] is None:
                        for p in config.processes:
                            if p.name not in corr.signal and p.name != data_name:
                                sub_data.sub(
                                    p.get_region(control).get_observable(hist_name)
                                )
                                usr_log.info(f"subtracting bkg {p.name}")
                    else:
                        for bkg in corr.bkg[control]:
                            bkg_hist = (
                                config.get_process_set(bkg)
                                .get(systematic)
                                .get_region(control)
                                .get_observable(hist_name)
                            )
                            sub_data.sub(bkg_hist)
                            usr_log.info(f"subtracting bkg {bkg}")
                    # sub_data.bin_content[sub_data.bin_content <= 0] = 1e-9
                    sub_data.nan_to_num(posinf=0, neginf=0)
                    B.append(sub_data.bin_content)

                    tot_yield = ""
                    for c, s in zip(coeff, corr.signal):
                        tot_yield += f"\n{s}--{c}\n"
                    tot_yield += f"\ndata subtracted--{sub_data.bin_content}\n"
                    usr_log.info(f"total yield {tot_yield}")

                usr_log.info("Arranging elements")
                A = np.stack(A, axis=1)
                B = np.array(B).transpose()

                try:
                    X2 = np.linalg.solve(A, B)
                except np.linalg.LinAlgError as e:
                    if "Singular matrix" in str(e):
                        log.critical("Singluar matrix. use alternative method")
                        x_bin = []
                        for i, (b, a) in enumerate(zip(B, A)):
                            try:
                                x = np.linalg.solve(a, b)
                            except np.linalg.LinAlgError as e2:
                                if "Singular matrix" in str(e2):
                                    tab = tabulate(
                                        a,
                                        headers=["control"] + corr.signal,
                                        showindex=b,
                                    )
                                    log.critical(
                                        f"\n!{'='*10}\nSingluear on {hist_name}:{i}\n {tab}\n{'='*10}!"
                                    )
                                    # x, residuals, rank, s = np.linalg.lstsq(A,b)
                                    x = np.linalg.lstsq(a, b, rcond=None)[0]
                                else:
                                    raise
                            x_bin.append(x)
                        X2 = np.array(x_bin)
                        # import pdb; pdb.set_trace()
                    else:
                        raise

                # testing error propagation
                # detail: https://cds.cern.ch/record/400631/files/9909031.pdf
                # variable map for paper: B = x, f = b, epsilon = a
                # assuming each element is independent
                err = None
                if err_prop:
                    err = []
                    # reading of system of equaiton for each bin
                    for i, (b, a, x) in enumerate(zip(B, A, X2)):
                        if verbose:
                            tab = tabulate(
                                a, headers=["control"] + corr.signal, showindex=b
                            )
                            usr_log.info(f"\n\n{hist_name} bin {i} \n {tab}\n {x}\n")

                        try:
                            inv_a = np.linalg.inv(a)
                        except np.linalg.LinAlgError:
                            inv_a = np.linalg.pinv(a)

                        # this is an approximation
                        '''
                        sig_b = np.sqrt(b)
                        sig_a = np.sqrt(a)
                        sig_inv_a_square = np.square(inv_a @ sig_a @ inv_a)
                        first_term = inv_a ** 2 @ sig_b ** 2
                        second_term = sig_inv_a_square @ (b ** 2)
                        x_err = first_term + second_term
                        print(f" aprro = {x_err}")
                        err.append(x_err)
                        '''

                        # better version?
                        # assuming each of element of b vector is iid,
                        # then the covariance matrix will just be diag(sig_b)
                        sig_a = np.sqrt(a)
                        first_term = inv_a**2 @ b
                        sig_inv_a_square = (inv_a**2) @ (sig_a**2) @ (inv_a**2)
                        second_term = sig_inv_a_square @ (b**2)
                        x_err = first_term + second_term
                        err.append(x_err)

                if verbose:
                    usr_log.info(f"control region names:{control_region_name}")
                    for i, (b, a, x) in enumerate(zip(B, A, X2)):
                        tab = tabulate(
                            a, headers=["control"] + corr.signal, showindex=b
                        )
                        m_err = f"+- {err[i]}" if err is not None else ""
                        usr_log.info(f"\n\n{hist_name} bin {i} \n {tab}\n {x}{m_err}\n")

                if err is not None:
                    err = np.array(err).T
                X2 = X2.T

                for i, signal_name in enumerate(corr.signal):
                    tmp_h = (
                        config.get_process_set(signal_name)
                        .get(systematic)[0]
                        .get_histogram(hist_name)
                        .copy(noparent=True)
                    )
                    key = (corr.type, signal_name, hist_name, systematic)
                    tmp_h.bin_content = np.nan_to_num(X2[i], posinf=0, neginf=0)
                    tmp_h.bin_content[tmp_h.bin_content < 0] = 0.0  # 1e-9
                    if err is not None:
                        tmp_h.sumW2 = np.nan_to_num(err[i], posinf=0, neginf=0)
                    output[key] = tmp_h
        return output

    def ApplyCorrections(self, _configMgr):
        """
        Apply MC process corrections to configMgr
        """
        return self._ApplyCorrections(_configMgr.corrections, _configMgr)

    def ApplyCorrectionsExtConfig(self, _configMgr, _cconfigMgr):
        """
        Apply MC process corrections derived from one configMgr
        to a different configMgr
        """
        return self._ApplyCorrections(_cconfigMgr.corrections, _configMgr)

    def _ApplyCorrections(self, corrections, _configMgr):
        """
        Master ApplyCorrections method called internally
        """

        configMgr = _configMgr.copy()

        fProcess = []
        for t in corrections:
            for p in corrections[t]:
                fProcess.append(p)

        # Apply corrections to each process
        for p in configMgr.processes:
            if p.name not in fProcess:
                continue

            usr_log.info(f"Applying correction to process {p.name}")
            for r in p.regions:
                if r.corr_type not in corrections:
                    usr_log.info(
                        f"Did not find your correction {r.corr_type} in corrrection list. Won't apply!"
                    )
                    continue

                for h in r.histograms:
                    h.mul(corrections[r.corr_type][p.name])

        return configMgr
