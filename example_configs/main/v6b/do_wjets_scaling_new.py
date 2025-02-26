from collinearw import ConfigMgr, run_HistMaker
from collinearw.histManipulate import Subtract_MC
from collinearw import run_PlotMaker
from collinearw.strategies.unfolding.utils import merge_migration_process
from collinearw.strategies import unfolding
from collinearw.backends import RootBackend
import numpy as np
from pathlib import Path

from do_unfold_finalize import single_thread

if __name__ == "__main__":

    config = ConfigMgr.open("merged_run2_nominal.pkl")

    prompt = ConfigMgr.open("merged_run2_signal.pkl")
    # breakpoint()
    # config.remove_process_set("wjets_2211")
    # config.append_process(prompt.get_process("wjets_2211"))

    skip = [
        "wjets_2211",
        "wjets_FxFx",
        "wjets_2211_EXPASSEW",
        "wjets",
        "singletop_Wt",
        "singletop_stchan",
        "singletop_Wt_DS",
    ]
    subtracted_config = Subtract_MC(config, "data", "subtracted_data", skip_processes=skip)

    # subtracted_config.remove_process_set("wjets_2211")
    # subtracted_config.append_process(prompt.get_process("wjets_2211"))

    scale_config_name = unfolding.bias.basic_rescaling(
        config,
        scale_process=subtracted_config.get_process("subtracted_data"),
        prefix="basic_bias",
    )

    truth_prior = "wjets_2211"
    signal_meas = "wjets_2211"
    signal_res = "wjets_2211"
    # truth_prior = prompt.get_process("wjets_2211")
    # signal_meas = prompt.get_process("wjets_2211")
    # signal_res = prompt.get_process("wjets_2211")

    # test_config = ConfigMgr.open(scale_config_name)
    # for region in test_config.get_process("wjets_2211"):
    #     if region.type not in {"reco_match"}:
    #         continue
    #     for hist in region:
    #         if hist.hist_type != "2d":
    #             continue
    #         hist.bin_content = hist.bin_content * 1/5.0

    single_thread(
        # test_config,
        scale_config_name,
        "unfolding_bias_basic.pkl",
        include_fakes=False,
        max_n_unfolds=2,
        exact_niter=True,
        save_all_iteration=True,
        truth_prior=truth_prior,
        signal_meas=signal_meas,
        signal_res=signal_res,
        # saveYoda=True,
    )
