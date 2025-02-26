from collinearw.core import SystematicBand
from collinearw import ConfigMgr
import numpy as np

import pathlib
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.CRITICAL)

log = logging.getLogger(__name__)

def create_generator_unfolding_uncertainty(iconfig, syst_config, oconfig=None, bases=None, symmetrize=True):

    if oconfig is None:
        oconfig = iconfig

    scaled_truth_name = "wjets_scaled"
    scaled_unfold_prefix = "unfold_alt_validate_closure_wjets_scaled"

    if bases is None:
        bases = ['realthang_fake-MU', 'realthang_fake-EL']

    for base in bases:
        unfolded_name = f'unfold_{base}'
        base_process = oconfig.get_process(unfolded_name)
        scaled_truth_process = syst_config.get(f"{scaled_truth_name}//nominal")
        scaled_unfold_path = f'{scaled_unfold_prefix}{base.replace("realthang", "")}'
        scaled_unfold_process = syst_config.get(f"{scaled_unfold_path}//nominal")

        for region in base_process:
            scaled_truth_region = scaled_truth_process.get(region.name)
            scaled_unfold_region = scaled_unfold_process.get(region.name)
            for observable in region:
                scaled_truth_h = scaled_truth_region.get(observable.name)
                scaled_unfold_h = scaled_unfold_region.get(observable.name)
                diff = scaled_truth_h.bin_content - scaled_unfold_h.bin_content
                if symmetrize:
                    up = dn = np.nan_to_num(np.abs(diff) / scaled_truth_h.bin_content)
                else:
                    up = np.where(diff >= 0, diff, 0) / scaled_truth_h.bin_content
                    dn = np.where(diff <= 0, diff, 0) / scaled_truth_h.bin_content
                    up = np.nan_to_num(np.abs(up))
                    dn = np.nan_to_num(np.abs(dn))
                shape = observable.bin_content.shape
                band = SystematicBand(f"unfold_basic", "unfold", shape)
                band.add_component("up", "unfold", up)
                band.add_component("down", "unfold", dn)
                observable.update_systematic_band(band)

def create_hidden_uncertainty(
    iconfig,
    syst_config_list,
    hiddens,
    oconfig=None,
    bases=None,
    symmetrize=True
):

    if oconfig is None:
        oconfig = iconfig

    scaled_truth_name = "wjets_scaled"
    scaled_unfold_prefix = "unfold_alt_validate_closure_wjets_scaled"

    if bases is None:
        bases = ['realthang_fake-MU', 'realthang_fake-EL']

    for base in bases:
        unfolded_name = f'unfold_{base}'
        base_process = oconfig.get_process(unfolded_name)

        for region in base_process:
            scaled_unfold_path = f'{scaled_unfold_prefix}{base.replace("realthang", "")}'
            for observable in region:
                if observable.name == "DeltaRLepJetClosest100" and "collinear" in region.name:
                    continue
                up_cache = []
                dn_cache = []
                for hidden, syst_config in zip(hiddens, syst_config_list):
                    scaled_truth_process = syst_config.get(f"{scaled_truth_name}//nominal")
                    scaled_unfold_process = syst_config.get(f"{scaled_unfold_path}//nominal")
                    scaled_truth_region = scaled_truth_process.get(region.name)
                    scaled_unfold_region = scaled_unfold_process.get(region.name)
                    scaled_truth_h = scaled_truth_region.get(observable.name)
                    scaled_unfold_h = scaled_unfold_region.get(observable.name)

                    diff = scaled_truth_h.bin_content - scaled_unfold_h.bin_content
                    if symmetrize:
                        up = dn = np.nan_to_num(np.abs(diff) / scaled_truth_h.bin_content)
                    else:
                        up = np.where(diff >= 0, diff, 0) / scaled_truth_h.bin_content
                        dn = np.where(diff <= 0, diff, 0) / scaled_truth_h.bin_content
                        up = np.nan_to_num(np.abs(up))
                        dn = np.nan_to_num(np.abs(dn))
                    up_cache.append(up)
                    dn_cache.append(dn)
                    shape = observable.bin_content.shape
                    band = SystematicBand(f"nouse_{hidden}", "unfold", shape)
                    band.add_component("up", f"unfold", up)
                    band.add_component("down", f"unfold", dn)
                    observable.update_systematic_band(band)
                try:
                    up = np.max(up_cache, axis=0)
                    dn = np.max(dn_cache, axis=0)
                except ValueError:
                    breakpoint()
                shape = observable.bin_content.shape
                band = SystematicBand(f"unfold_hidden", "unfold", shape)
                band.add_component("up", f"unfold", up)
                band.add_component("down", f"unfold", dn)
                observable.update_systematic_band(band)

if __name__ == "__main__":
    # iconfig = "band_unfolded_June_30_merged_run2_full_syst_June15.pkl"
    # iconfig = ConfigMgr.open(iconfig)
    # # syst_config = "unfolded_truth_match_scaled_run2_2211.pkl"
    # # syst_config = ConfigMgr.open(syst_config)
    # # create_generator_unfolding_uncertainty(iconfig, syst_config)
    # # iconfig.save("band_unfold_March2023_basic_unfold_uncertainty.pkl")
    #
    # hiddens = [
    #     "DeltaRLepJetClosest100",
    #     "Ht30",
    #     "jet1Pt",
    #     # "jet2Pt",
    #     "nJet30",
    #     "wPt",
    #     # "mjj",
    # ]
    # syst_config_list = [ConfigMgr.open(f"unfolded_June_15_scaled_hidden_{x}_run2_2211.pkl") for x in hiddens]
    # create_hidden_uncertainty(iconfig, syst_config_list, hiddens)
    #
    # iconfig.save("band_unfolded_June_30_basic_unfold_uncertainty.pkl")

    iconfig = "band_unfolded_Oct_v3.pkl"
    iconfig = ConfigMgr.open(iconfig)

    # prompt = ConfigMgr.open("merged_run2_signal.pkl")
    # for r in iconfig.get_process("wjets_2211"):
    #     if r.type != "truth":
    #         print(r.name)
    #         continue
    #     for h in r:
    #         h.bin_content = prompt.get_process("wjets_2211").get(r.name).get(h.name).bin_content


    basic_config = ConfigMgr.open("unfolding_bias_basic.pkl")
    create_generator_unfolding_uncertainty(iconfig, basic_config)

    hidden_config = ConfigMgr.open("unfolding_bias_hidden.pkl")
    create_hidden_uncertainty(iconfig, [hidden_config], ["hiddens"])

    iconfig.save("band_unfolded_Oct_unfold_uncertainty_v3.pkl")
