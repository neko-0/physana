import pathlib
import collinearw
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend

def main():

    config_file = "run2_2211_nominal.pkl"
    config = collinearw.ConfigMgr.open(config_file)

    signal_process = "wjets_2211"

    el_regions = config.get_process(signal_process).list_regions("*electron_inclusive_reco*rA*")
    #el_regions += config.get_process(signal_process).list_regions("electron_inclusive_truth")

    mu_regions = config.get_process(signal_process).list_regions("*muon_inclusive_reco*rA*")
    #mu_regions += config.get_process(signal_process).list_regions("muon_inclusive_truth")

    histogram_list = [
        "dR_dPhi_reco",
        "dR_dPhi_truth",
    ]

    print(f"{el_regions=}")
    print(f"{mu_regions=}")

    for el_r, mu_r in zip(el_regions, mu_regions):
        for hist in histogram_list:
            base_path = pathlib.Path(f"{config.out_path}/plots2D/{hist}").resolve()

            el_hist = config.get(f"{signal_process}//nominal//{el_r}//{hist}")
            mu_hist = config.get(f"{signal_process}//nominal//{mu_r}//{hist}")
            #sum_hist = el_hist + mu_hist
            #sum_root_hist = sum_hist.root
            el_root_hist = el_hist.root
            mu_root_hist = mu_hist.root
            RootBackend.apply_process_styles(el_root_hist, config.get_process(signal_process))
            RootBackend.apply_process_styles(mu_root_hist, config.get_process(signal_process))

            el_base_path = base_path.joinpath("_el")
            mu_base_path = base_path.joinpath("_mu")

            unfolding.plot.plot_response(el_root_hist, output_path=el_base_path.with_suffix(".pdf"), logz=False, add_blue=True)
            unfolding.plot.plot_response(mu_root_hist, output_path=mu_base_path.with_suffix(".pdf"), logz=False, add_blue=True)

if __name__ == "__main__":
    main()
