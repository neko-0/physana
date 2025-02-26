from pathlib import Path
from tqdm import tqdm
from collinearw import ConfigMgr
from collinearw.strategies import unfolding
from collinearw.histManipulate import Subtract_MC
from collinearw.utils import all_redirected


def unfold_bootstrap(ifile, ofile):
    config = ConfigMgr.open(ifile)
    # metadata = unfolding.metadata
    process_sets = config.process_sets
    # signals = [x for x in process_sets if x.process_type == "signal"]
    bkgs = [x for x in process_sets if x.process_type == "bkg"]
    data = [x for x in process_sets if x.process_type == "data"]
    # alt_signals = [x for x in process_sets if x.process_type == "signal_alt"]
    assert len(data) == 1
    data = data[0]

    subtracted_data_name = f"{data.name}_subtracted_bkg_fakes"
    skip_processes = [p.name for p in process_sets if p not in bkgs]
    subtracted_config = Subtract_MC(
        config,
        data_name=data.name,
        rename=subtracted_data_name,
        skip_processes=skip_processes,
        systematics=None,
    )
    subtracted_data = subtracted_config.get_process(subtracted_data_name)

    # preparing region names with (reco, truth, match)
    regions = [
        (
            "electron_collinear_reco_ABCD-fake-EL-rA_",
            "electron_collinear_truth",
            "electron_collinear_truth_reco_matching",
        ),
        (
            "muon_collinear_reco_ABCD-fake-MU-rA_",
            "muon_collinear_truth",
            "muon_collinear_truth_reco_matching",
        ),
        # (
        #     "electron_inclusive_reco_ABCD-fake-EL-rA_",
        #     "electron_inclusive_truth",
        #     "electron_inclusive_truth_reco_matching",
        # ),
        # (
        #     "muon_inclusive_reco_ABCD-fake-MU-rA_",
        #     "muon_inclusive_truth",
        #     "muon_inclusive_truth_reco_matching",
        # ),
    ]

    # prepare observable names with (reco, truth, resonse)
    observables = [
        ("Ht30", "HtTruth30", "response_matrix_Ht30"),
        ("wPt", "wTruthPt", "response_matrix_wPt"),
        ("nJet30", "nTruthJet30", "response_matrix_nJet30"),
        ("jet1Pt", "jet1TruthPt", "response_matrix_jet1Pt"),
        (
            "DeltaRLepJetClosest100",
            "DeltaRTruthLepJetClosest100",
            "response_matrix_DeltaRLepJetClosest100",
        ),
    ]

    # make copy of a process to store unfolded output
    unfold_process = subtracted_data.copy()
    unfold_process.clear_content()
    unfold_process.name = "unfold_realthang"

    signal_process = subtracted_config.get_process("wjets_2211")

    bayes_unfold = unfolding.core.bayes_unfold

    with all_redirected(Path("bootstrap_unfold_log.txt").open("w+")):
        for (reco_r, truth_r, match_r) in regions:
            data_reco_r = subtracted_data.get(reco_r)
            signal_reco_r = signal_process.get(reco_r)
            signal_truth_r = signal_process.get(truth_r)
            signal_match_r = signal_process.get(match_r)
            unfold_truth_r = unfold_process.get(truth_r)
            for (reco_h, truth_h, res_h) in observables:
                data_reco_h = data_reco_r.get(reco_h)
                signal_reco_h = signal_reco_r.get(reco_h)
                signal_truth_h = signal_truth_r.get(truth_h)
                signal_match_h = signal_match_r.get(res_h)
                unfold_truth_h = unfold_truth_r.get(truth_h)
                nreplica = subtracted_data.get(reco_r).get(reco_h).nreplica
                for i in range(nreplica):
                    signal_match_h.xtitle = data_reco_h.xtitle
                    signal_match_h.ytitle = data_reco_h.ytitle
                    hUnfoldable = data_reco_h.get_replica_root(i)
                    hMeas = signal_reco_h.get_replica_root(i)
                    hTruth = signal_truth_h.get_replica_root(i)
                    hRes = signal_match_h.get_replica_root(i)
                    content, ww = bayes_unfold(hUnfoldable, hMeas, hTruth, hRes, 2)
                    unfold_truth_h.replica[i] = content
                    unfold_truth_h.replica_sumw2[i] = ww
                # nominal
                hMeas = signal_reco_h.root
                hTruth = signal_truth_h.root
                signal_match_h.xtitle = hMeas.GetName()
                signal_match_h.ytitle = hTruth.GetName()
                hRes = signal_match_h.root
                content, ww = bayes_unfold(hUnfoldable, hMeas, hTruth, hRes, 2)
                unfold_truth_h.bin_content = content
                unfold_truth_h.sumW2 = ww

    # regions = metadata.regions(subtracted_config)
    # observables = metadata.observables(subtracted_config)
    subtracted_config.append_process(unfold_process)

    return subtracted_config.save(ofile)


if __name__ == "__main__":

    import glob

    input_fs = glob.glob("Jan2023_replica_run2_*.*.pkl")
    output_fs = [f"replica_unfolded_{f}" for f in input_fs]

    batch_mode = False

    if batch_mode:
        # import dask
        from dask_jobqueue import SLURMCluster
        from dask.distributed import Client
        from dask.distributed import as_completed as dask_as_completed

        workers = 45
        cores = 12
        account = "shared"
        cluster = SLURMCluster(
            queue=account,
            walltime="05:00:00",
            project="collinear_Wjets",
            nanny=False,
            cores=1,
            processes=1,
            memory="16GB",
            job_extra=[
                f"--account={account}",
                f"--partition={account}",
                f"--cpus-per-task={cores}",
            ],
            # cores=cores,
            # memory="128GB",
            # job_extra=[f'--account={account}', f'--partition={account}'],
            local_directory="dask_unfold_output",
            log_directory="dask_unfold_logs",
            n_workers=workers,
            death_timeout=36000,
        )
        client = Client(cluster)
        client.get_versions(check=True)
        futures = []
        with cluster, Client(cluster) as client:
            client.get_versions(check=True)
            for input_f, output_f in tqdm(zip(input_fs, output_fs)):
                future = client.submit(unfold_bootstrap, input_f, output_f)
                futures.append(future)
            for future in tqdm(dask_as_completed(futures)):
                _ = future.result()
    else:
        input_fs = [input_fs[0]]
        output_fs = [output_fs[0]]

        for input_f, output_f in tqdm(zip(input_fs, output_fs)):
            unfold_bootstrap(
                input_f,
                output_f,
                # systematic_names=[None],
                # saveYoda=True,
                # include_fakes=False,
                # correct_hMeas=False,
                # subtract_notmatch=False,
                # overflow=True,
                # store_vfakes=False,
                # max_n_unfolds=20,
            )
