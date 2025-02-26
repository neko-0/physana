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
    fxfx = ConfigMgr.open("merged_run2_signal.pkl")

    prompt = ConfigMgr.open("merged_run2_signal.pkl")

    # config.remove_process_set("wjets_2211")
    # config.append_process(prompt.get_process("wjets_2211"))
    config.append_process(fxfx.get_process("wjets_FxFx"))

    scale_config_name = unfolding.bias.cross_rescaling(
        config,
        signal="wjets_2211",
        alt_signal="wjets_FxFx",
        prefix="hidden_bias",
        reco_to_truth=False,
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
        "unfolding_bias_hidden.pkl",
        include_fakes=False,
        max_n_unfolds=2,
        exact_niter=True,
        save_all_iteration=True,
        truth_prior=truth_prior,
        signal_meas=signal_meas,
        signal_res=signal_res,
    )
