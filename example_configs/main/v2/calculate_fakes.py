doEventLevel = True

if __name__ == "__main__":
    from collinearw import ConfigMgr
    from collinearw import HistManipulate
    from collinearw import run_HistManipulate

    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.CRITICAL)

    run2 = ConfigMgr.open("./output/unfoldTest_v2_fullrun2/run2.pkl")
    if doEventLevel:
        skip_processes = [
            process.name
            for process in run2.processes
            if process.process_type not in ["data", "signal", "bkg"]
        ]

        sub_config = HistManipulate.Subtract_MC(
            run2, "data", "subtracted_data", skip_processes=skip_processes
        )
        tf_config = run_HistManipulate.run_ABCD_TF(
            sub_config, "subtracted_data", oname=None
        )
        run_HistManipulate.run_ABCD_Fakes_EventLevel(
            run2,
            tf_config,
            "subtracted_data",
            {
                "*muon*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_muon"),
                "*electron*": (("abs(lep1Eta)", "lep1Pt"), "eta_vs_lepPt_electron"),
            },
            "fakes",
            skip_process=skip_processes,
        )

    else:
        fake_config = run_HistManipulate.run_ABCD_Fakes(run2, False, None, ext_tf=None)
        fake_config.save('fakes.pkl')

from collinearw import run_PlotMaker

run2_fakes = ConfigMgr.open("./output/unfoldTest_v2_fullrun2/fakes.pkl")
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
