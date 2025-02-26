import collinearw


def explore_band():
    config = collinearw.ConfigMgr.open("band_unfold.pkl")

    plotmaker = collinearw.PlotMaker(config, "mc16a_only_unfold_bands_v2")

    filter = "*reco_ABCD*rA*"
    process_sets = (
        config.list_processes()
    )  # ["unfold_realthang_fake-MU", "unfold_realthang_fake-EL"]
    print(process_sets)
    for proc in process_sets:
        m_proc = config.get_process(proc)
        plotmaker.plot_relative_systematics(
            m_proc,
            region_filter=filter,
            yrange=(0.001, 100),
            output_dir="mc16a_only_unfold_bands_v2",
        )


def explore_syst():

    config = collinearw.ConfigMgr.open("band_run2_2211.pkl")

    group = {}
    group.update(
        config.generate_systematic_group("JET-JER-up", ("*JET*JER*", "*JET*up*", ""))
    )

    syst_list = []
    for sub_group in group.values():
        syst_list += sub_group

    plotmaker = collinearw.PlotMaker(config)
    for proc in config.process_sets:
        plotmaker.plot_process_set(
            proc,
            region_filter="*reco_ABCD*rA*",
            systematic_list=syst_list,
            output_dir="mc16a_only",
        )


if __name__ == "__main__":
    explore_band()
    # explore_syst()
