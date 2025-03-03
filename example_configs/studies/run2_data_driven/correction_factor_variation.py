"""
load and generate background correction factor variation as systematics
"""
import collinearw


def generate_correction_variation(
    config:str,
    process_name:str,
    el_variation:str,
    mu_variation:str,
):
    """
    load correction factor variation (mean,min,max) to refill process and
    generate sysemtatic.
    """

    config = collinearw.ConfigMgr.open(config)
    nominal_process = config.get_process(process_name)

    corr_obs = "nJet30"

    # create copy of the config for inherenting meta data
    c_config = config.copy(shallow=True)
    c_config.clear_process_set()
    nominal_process.clear_content()
    c_config.append_process(nominal_process)

    up_proc = nominal_process.copy()
    up_proc.clear_content()
    down_proc = nominal_process.copy()
    down_proc.clear_content()

    up_syst = collinearw.core.Systematics("correction_factor_up", "NoSys", "", "CR")
    up_proc.systematic = up_syst
    down_syst = collinearw.core.Systematics("correction_factor_down", "NoSys", "", "CR")
    down_proc.systematic = down_syst

    c_config.append_process(up_proc, mode="merge")
    c_config.append_process(down_proc, mode="merge")

    el_var = collinearw.serialization.Serialization().from_pickle(el_variation)
    mu_var = collinearw.serialization.Serialization().from_pickle(mu_variation)

    # (correction_type, process, corr_obs, systematic)
    mean_el_lookup = ("electron", "dijets", corr_obs, None)
    up_el_lookup = ("electron", "dijets", corr_obs, up_syst.full_name)
    down_el_lookup = ("electron", "dijets", corr_obs, down_syst.full_name)

    mean_mu_lookup = ("muon", "dijets", corr_obs, None)
    up_mu_lookup = ("muon", "dijets", corr_obs, up_syst.full_name)
    down_mu_lookup = ("muon", "dijets", corr_obs, down_syst.full_name)

    c_config.corrections.clear_database()
    c_config.corrections.clear_files()
    c_config.corrections[mean_el_lookup] = el_var["mean"]
    c_config.corrections[down_el_lookup] = el_var["min"]
    c_config.corrections[up_el_lookup] = el_var["max"]

    c_config.corrections[mean_mu_lookup] = mu_var["mean"]
    c_config.corrections[down_mu_lookup] = mu_var["min"]
    c_config.corrections[up_mu_lookup] = mu_var["max"]

    c_config.corrections.save_current_correction("current_dijets_corr_var.shelf")
    c_config.corrections.clear_database()
    c_config.corrections.clear_files()
    c_config.corrections.add_correction_file("current_dijets_corr_var.shelf")

    c_config.phasespace_apply_nominal = False

    # breakpoint()

    refilled_config = collinearw.run_HistMaker.run_HistMaker(
        c_config,
        #split_type="process",
        #merge_buffer_size=50,
        #submission_buffer=50,
        #executor=client,
        #as_completed=dask_as_completed,
        #r_copy=False,
        #copy=False,
    )

    refilled_config.save("dijets_variation.pkl")

if __name__ == "__main__":
    el_var = "../dijets_CR_study/iterative_sf/dijets_el_correction_variation.pkl"
    mu_var = "../dijets_CR_study/iterative_sf/dijets_mu_correction_variation.pkl"

    generate_correction_variation("prune_run2_2211.pkl", "dijets", el_var, mu_var)
