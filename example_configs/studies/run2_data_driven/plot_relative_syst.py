
import collinearw
from collinearw import run_PlotMaker, ConfigMgr

def main():

    config = ConfigMgr.open(f"band_unfold.pkl")

    theory_excluding = [
        "B-tagging",
        "Lepton",
        "Jet",
        "ttbar",
        "zjets",
        "diboson",
        "singletop",
        "wjets-fxfx",
        "dijets",
    ]

    data_excluding = ["diboson"]

    plotting_config = [
        (
            ["unfold_realthang_fake-EL"],
            {"exclude_band":data_excluding, "yrange":(-0.5,0.5)},
        ),
        (
            ["unfold_realthang_fake-MU"],
            {"exclude_band":data_excluding, "yrange":(-0.5,0.5)},
        ),
        (
            ["wjets_2211"],
            {"exclude_band":theory_excluding, "yrange":(-1.5,1.5)},
        ),
        (
            ["wjets_FxFx"],
            {"exclude_band":theory_excluding, "yrange":(-1.5,1.5)},
        ),
        (
            ["wjets"],
            {"exclude_band":theory_excluding, "yrange":(-1.5,1.5)},
        ),
    ]

    for process_list, plot_config in plotting_config:
        for phasespace in ["inclusive", "collinear"]:
            run_PlotMaker.plot_pset_relative_systematics(
                config,
                process_list,
                region_filter=f"*{phasespace}*truth*",
                #show_components=True,
                show_details=False,
                output_dir="relative_syst_unfolded_Apr28_2022",
                include_band = ["total"],
                **plot_config,
            )

if __name__ == "__main__":
    main()
