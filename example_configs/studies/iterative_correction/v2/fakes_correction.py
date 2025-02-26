from collinearw import ConfigMgr, HistManipulate, PlotMaker, Histogram, HistMaker
from collinearw import run_HistMaker
from collinearw import histManipulate
from collinearw import run_PlotMaker
from collinearw.strategies import abcd
import os
import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)
log = logging.getLogger(__name__)


def plot_region(plotmaker, outdir, region):
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass
    for h in region.histograms:
        canvas = plotmaker.make_canvas()
        canvas.cd()
        hr = h.to_root
        hr.Draw("h")
        canvas.SaveAs(f"{outdir}/{h.name}.png")


def plot_correction_hist(plotmaker, outdir, histlist):
    try:
        os.makedirs(outdir)
    except FileExistsError:
        pass

    total_hist = histlist[0].copy()
    buff = []

    canvas = plotmaker.make_canvas()
    canvas.cd()
    legend = plotmaker.make_legend(0.55, 0.55, 0.85, 0.85)
    color = 2
    for i, h in enumerate(histlist, start=1):
        hr = h.to_root
        hr.SetLineColor(color)
        buff.append(hr)
        legend.AddEntry(hr, f"iteration:{i}")
        if i == 1:
            hr.Draw("h")
        else:
            hr.Draw("same")
            total_hist.mul(h)
        color += 1
    legend.Draw()
    canvas.SaveAs(f"{outdir}/overlap_{total_hist.name}.png")

    canvas = plotmaker.make_canvas()
    canvas.cd()
    hr = total_hist.to_root
    hr.Draw("h")
    canvas.SaveAs(f"{outdir}/total_{h.name}.png")


def run_iteration(signal="wjets", skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"]):

    # run usual ABCD fake before iteration correction

    pre_iterate = True

    src_path = pathlib.Path(os.path.realpath(__file__)).parent
    run2 = ConfigMgr.open(f"{src_path}/run2.pkl")

    ofile_name = f"run2_{signal}_fakes_pre_iteration"

    if pre_iterate:

        # subtract all MC process from  data
        bkgd_subtract = HistManipulate.Subtract_MC(
            run2,
            "data",
            "subtracted_data",
            skip_processes=skip,
        )

        # compute ABCD transfer factor
        tf = histManipulate.run_ABCD_TF(bkgd_subtract, "subtracted_data")

        # redo tree filling on region B get fakes in region A
        fake = histManipulate.run_ABCD_Fakes_EventLevel(
            run2,
            tf,
            "subtracted_data",
            {
                "muon*": (("abs(lep1Eta)", "lep1Pt"), "abs(eta)_vs_lepPt_mu"),
                "electron*": (("abs(lep1Eta)", "lep1Pt"), "abs(eta)_vs_lepPt_e"),
            },
            skip_process=skip,
            use_mp=True,
        )

        # do stack plotting
        run_PlotMaker.run_stack(
            fake,
            f"Wj_AB212108_v3p3_{signal}_pre_iterate",
            data="data",
            mcs=[signal, "zjets", "ttbar", "singletop", "diboson", "fakes"],
            low_yrange=(0.5, 1.7),
            logy=True,
            workers=16,
            low_ytitle="Data/Pred",
        )

    # ==============================================================================
    # this is the beginnig of iteration step to correct wjets

    # observable to use for correction
    correction_obs = "nJet25"

    # holding the correction in each iteration
    el_correction_buffer = []
    mu_correction_buffer = []
    el_vali_correction_buffer = []
    mu_vali_correction_buffer = []

    plotmaker = PlotMaker(run2, "plots")

    # open the result we got from above
    run2_fakes = ConfigMgr.open(f"{run2.out_path}/{ofile_name}.pkl")

    # getting the signal region for electron and muon
    corr_region_name_el = abcd.abcd_signal_region_name(
        run2_fakes, "electron", "electron_track_iso-pid"
    )
    corr_region_name_mu = abcd.abcd_signal_region_name(
        run2_fakes, "muon", "muon_official_iso-pid"
    )

    # getting the validation region
    corr_vali_region_name_el = abcd.abcd_signal_region_name(
        run2_fakes, "electron_backToBack", "electron_track_iso-pid"
    )
    corr_vali_region_name_mu = abcd.abcd_signal_region_name(
        run2_fakes, "muon_backToBack", "muon_official_iso-pid"
    )

    for i in range(1, 10):

        # perform subtraction: subtracted = data - fakes - diboson - singletop - ttbar - zjets
        # note no wjets subtraction here
        bkgd_subtract_iter = HistManipulate.Subtract_MC(
            run2_fakes,
            "data",
            f"subtracted_data_iter{i}",
            skip_processes=[signal] + skip,
        )

        # finding subtracted / wjets
        # this will be the scale factor for correction wjets
        correction_iter = HistManipulate.Divide_Process(
            bkgd_subtract_iter.get_process(f"subtracted_data_iter{i}"),
            bkgd_subtract_iter.get_process(signal),
            f"correction_iter{i}",
        )

        # getting region we want for caculating the correction factor
        el_region = correction_iter.get_region(corr_region_name_el)
        mu_region = correction_iter.get_region(corr_region_name_mu)

        el_vali_region = correction_iter.get_region(corr_vali_region_name_el)
        mu_vali_region = correction_iter.get_region(corr_vali_region_name_mu)

        # plots all the correction factor
        plot_region(plotmaker, f"{src_path}/el_iter{i}", el_region)
        plot_region(plotmaker, f"{src_path}/mu_iter{i}", mu_region)

        plot_region(plotmaker, f"{src_path}/el_vali_iter{i}", el_vali_region)
        plot_region(plotmaker, f"{src_path}/mu_vali_iter{i}", mu_vali_region)

        # make copy of the configMgr
        run2_iter = run2_fakes.copy()

        print(f"Applying SF iter{i} ====================================")
        """
        for r in run2_iter.get_process("wjets").regions:
            for h in r.histograms:
                h.mul(corr_region_iter1.get_observable(h.name))
        """
        # run2_iter.clear_process()
        # run2_iter.prepared = False

        # updating the correction meta data
        el_corr_meta = run2_iter.corrections["electron"]["nJet25"]
        if signal in el_corr_meta:
            # if previous correction exist, multiply the latest correction on that
            # e.g. corr2 * corr1 * wjets
            el_corr_meta[signal].mul(el_region.get_observable("nJet25"))
        else:
            el_corr_meta[signal] = el_region.get_observable("nJet25").copy()
        el_correction_buffer.append(el_region.get_observable("nJet25").copy())

        # electron validation region
        el_vali_corr_meta = run2_iter.corrections["electron_validation"]["nJet25"]
        if signal in el_vali_corr_meta:
            el_vali_corr_meta[signal].mul(el_vali_region.get_observable("nJet25"))
        else:
            el_vali_corr_meta[signal] = el_vali_region.get_observable("nJet25").copy()
        el_vali_correction_buffer.append(el_vali_region.get_observable("nJet25").copy())

        # muon signal region
        mu_corr_meta = run2_iter.corrections["muon"]["nJet25"]
        if signal in mu_corr_meta:
            mu_corr_meta[signal].mul(mu_region.get_observable("nJet25"))
        else:
            mu_corr_meta[signal] = mu_region.get_observable("nJet25").copy()
        mu_correction_buffer.append(mu_region.get_observable("nJet25").copy())

        # muon validation region
        mu_vali_corr_meta = run2_iter.corrections["muon_validation"]["nJet25"]
        if signal in mu_vali_corr_meta:
            mu_vali_corr_meta[signal].mul(mu_vali_region.get_observable("nJet25"))
        else:
            mu_vali_corr_meta[signal] = mu_vali_region.get_observable("nJet25").copy()
        mu_vali_correction_buffer.append(mu_vali_region.get_observable("nJet25").copy())

        # clear the content of wjets process, and then we refill it with
        # correction apply on each event
        wjets_p = run2_iter.get_process(signal)
        wjets_p.clear_content()

        # create histmaker object, and use the process level funtion to refill
        histmaker = HistMaker()
        histmaker._corrections = run2_iter.corrections
        branch_list = run2_iter.reserved_branches
        for f in wjets_p.filename:
            histmaker.plevel_process(
                f,
                wjets_p,
                branch_list,
            )

        print(f"Updatin correction iter{i} ====================================")

        # perform subtracion before fake estimation
        # note here, the subtraction is : data-corr*wjets-zjets-ttbar-singletop-diboson
        # no fake subtracion in here.
        bkgd_subtract = HistManipulate.Subtract_MC(
            run2_iter,
            "data",
            f"subtracted_data_iter{i}",
            skip_processes=skip + ["fakes"],
        )

        # compute the transfer factor
        tf = histManipulate.run_ABCD_TF(bkgd_subtract, f"subtracted_data_iter{i}")

        # update the output file name,
        # save the configMgr in each iteration step for later debugging.
        ofile_name = f"run2_{signal}_fakes_iter{i}"

        # compute the fakes as usual, but now the wjets is corrected
        fake_iter = histManipulate.run_ABCD_Fakes_EventLevel(
            run2_iter,
            tf,
            f"subtracted_data_iter{i}",
            {
                "muon*": (("abs(lep1Eta)", "lep1Pt"), "abs(eta)_vs_lepPt_mu"),
                "electron*": (("abs(lep1Eta)", "lep1Pt"), "abs(eta)_vs_lepPt_e"),
            },
            skip_process=skip + ["fakes"],
            use_mp=True,
            force_refill=False,
        )

        # do plotting
        run_PlotMaker.run_stack(
            fake_iter,
            f"Wj_AB212108_v3p3_{signal}_iter{i}",
            data="data",
            mcs=[signal, "zjets", "ttbar", "singletop", "diboson", "fakes"],
            low_yrange=(0.5, 1.7),
            logy=True,
            workers=16,
            low_ytitle="Data/Pred",
        )

        # update the configMgr
        run2_fakes = fake_iter

    # plotting the accumulated correction factor
    plot_correction_hist(
        plotmaker, f"{src_path}/{signal}_el_total_corr", el_correction_buffer
    )
    plot_correction_hist(
        plotmaker, f"{src_path}/{signal}_mu_total_corr", mu_correction_buffer
    )
    plot_correction_hist(
        plotmaker, f"{src_path}/{signal}_el_vali_total_corr", el_vali_correction_buffer
    )
    plot_correction_hist(
        plotmaker, f"{src_path}/{signal}_mu_vali_total_corr", mu_vali_correction_buffer
    )


if __name__ == "__main__":
    run_iteration("wjets", ["wjets_mg", "wjets_FxFx", "wjets_powheg"])
    # run_iteration("wjets_mg", ["wjets", "wjets_FxFx", "wjets_powheg"])
    # run_iteration("wjets_FxFx", ["wjets", "wjets_mg", "wjets_powheg"])
    # run_iteration("wjets_powheg", ["wjets", "wjets_FxFx", "wjets_mg"])
