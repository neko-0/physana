doEventLevel = True
doCorr = True

if __name__ == "__main__":
    from collinearw import ConfigMgr
    from collinearw import run_HistManipulate
    import os
    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.CRITICAL)

    run2 = ConfigMgr.open("./output/unfoldTest_v3_fullrun2_sys/run2.pkl")
    if doEventLevel:
        skip_processes = [
            process.name
            for process in run2.processes
            if process.process_type not in ["data", "signal", "bkg"]
        ]

        tf_params = {
            "*muon*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_muon"),
            "*electron*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_electron"),
        }

        basedir = os.path.dirname(os.path.realpath(__file__))
        corr_path = f"{basedir}/../../../data/Wj_AB212108_v3/ControlRegions/wjets_Sherpa_signal_correction.pkl"

        fake_config = run_HistManipulate.run_abcd_fakes_estimation(
            run2,
            tf_params,
            correction=corr_path if doCorr else None,
            skip_process=skip_processes,
        )

    else:
        fake_config = run_HistManipulate.run_ABCD_Fakes(run2, False, None, ext_tf=None)

    fake_config.save('fakes.pkl')

from collinearw import run_PlotMaker

run2_fakes = ConfigMgr.open("./output/unfoldTest_v3_fullrun2/fakes.pkl")
run_PlotMaker.run_stack(
    run2_fakes,
    "fakes",
    data="data",
    mcs=["wjets", "zjets", "ttbar", "singletop", "diboson", "fakes"],
    low_yrange=(0.5, 1.7),
    logy=True,
    workers=16,
    # rname_filter=["*rA*"],
    # check_region=True,
    low_ytitle="Data/Pred",
)
