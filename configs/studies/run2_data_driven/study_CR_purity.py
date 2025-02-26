import collinearw


dijets_CR_variation = {
    "nominal" : "(lep1Topoetcone20/lep1Pt<0.06 && ptcone20_TightTTVALooseCone_pt1000/lep1Pt<0.06 && lep1Signal==0)",
    "relax_track_or_lep1Signal" : "(lep1Topoetcone20/lep1Pt<0.06 && (ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 || lep1Signal==0))",
    "relax_calo_or_lep1Signal" : "(ptcone20_TightTTVALooseCone_pt1000/lep1Pt<0.06 && (lep1Topoetcone20/lep1Pt>=0.06 || lep1Signal==0))",
    "relax_track_or_calo" : "(lep1Signal==1 && (ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 || lep1Topoetcone20/lep1Pt>=0.06))",
    "relax_track_or_calo_or_lep1Siganl" : "(ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 || lep1Topoetcone20/lep1Pt>=0.06 || lep1Signal==0)",
    "relax_track_and_calo_or_lep1Siganl" : "((ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 && lep1Topoetcone20/lep1Pt>=0.06) || lep1Signal==0)",
    "relax_track_and_calo_and_lep1Siganl" : "(ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 && lep1Topoetcone20/lep1Pt>=0.06 && lep1Signal==0)",
    "relax_track_or_calo_and_lep1Siganl" : "(ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 || (lep1Topoetcone20/lep1Pt>=0.06 && lep1Signal==0))",
    "relax_track_and_lep1Siganl_or_calo" : "((ptcone20_TightTTVALooseCone_pt1000/lep1Pt>=0.06 && lep1Signal==0) || lep1Topoetcone20/lep1Pt>=0.06)",
}

dijets_Mu_CR_variation = {
    "nominal" : "(lep1Signal==1 && IsoLoose_FixedRad == 0)",
    "relax_iso_or_lep1Signal" : "(lep1Signal==0 && IsoLoose_FixedRad == 1)",
    "relax_iso_and_lep1Signal" : "(lep1Signal==0 && IsoLoose_FixedRad == 0)",
}

name_mapping_el = {
    "nominal" : "Pass track+calo iso, inverted (PiD | d0Z0)",
    "relax_track_or_lep1Signal" : "Inverted (track iso | PiD | d0Z0)",
    "relax_calo_or_lep1Signal" : "Inverted (calo iso | PiD | d0Z0)",
    "relax_track_or_calo" : "Pass (PiD+d0Z0), inverted (track | calo iso)",
    "relax_track_or_calo_or_lep1Siganl" : "Inverted (calo | track | PiD | d0Z0)",
    "relax_track_and_calo_or_lep1Siganl" : "Inverted ((track+calo) | Pid | d0Z0)",
    "relax_track_and_calo_and_lep1Siganl" : "Inverted (track+calo+PiD+d0Z0)",
    "relax_track_or_calo_and_lep1Siganl" : "Inverted (track | (calo+PiD+d0Z0))",
    "relax_track_and_lep1Siganl_or_calo" : "Inverted ((track+PiD+d0Z0) | calo)",
}

name_mapping_mu = {
    "nominal" : "Pass d0z0, inverted (track+calo iso)",
    "relax_iso_or_lep1Signal" : "Pass iso, inverted (d0Z0)",
    "relax_iso_and_lep1Signal" : "Inverted (d0Z0 & track+calo iso)",
}


def generate_background_purity_table(config):
    config = collinearw.ConfigMgr.open(config)
    obs = "nBJet30"
    signal = "wjets_2211"
    data = "data"
    mc_list = [
        "wjets_2211",
        "zjets_2211",
        "ttbar",
        "singletop",
        "diboson_powheg",
        "wjets_2211_tau",
        "vgamma",
        "dijets",
    ]
    el_regions = config.get_process("dijets").list_regions("*electron*rA*")
    mu_regions = config.get_process("dijets").list_regions("*muon*rA*")
    study_CR = [
        #("dijets", [(f"dijetsCR_Ele_{var}", name_mapping_el[var]) for var in dijets_CR_variation]),
        #("dijets", [(f"dijetsCR_Mu_{var}", name_mapping_mu[var]) for var in dijets_Mu_CR_variation]),
        ("dijets", [(r,r) for r in el_regions]),
        ("dijets_mean", [(r,r) for r in el_regions]),
        ("dijets_up", [(r,r) for r in el_regions]),
        ("dijets_down", [(r,r) for r in el_regions]),
        ("dijets", [(r,r) for r in mu_regions]),
        ("dijets_mean", [(r,r) for r in mu_regions]),
        ("dijets_up", [(r,r) for r in mu_regions]),
        ("dijets_down", [(r,r) for r in mu_regions]),
    ]
    processes = [config.get_process(mc) for mc in mc_list]

    for cr_process, CRs in study_CR:
        header_row = [
            "Control Region Selection",
            f"{data} yield",
            f"{cr_process} yield",
            f"{cr_process} purity",
            f"{signal} yield",
            f"{signal} contamination",
        ]
        print(' & '.join(header_row))
        for (cr, nickname) in CRs:
            try:
                data_yield = config.get_process(data).get_region(cr).effective_event_count
                total_mc = sum([process.get_region(cr).effective_event_count for process in processes])
                cr_process_yield = config.get_process(cr_process).get_region(cr).effective_event_count
                signal_yield = config.get_process(signal).get_region(cr).effective_event_count
            except KeyError:
                print(f"encounter error {nickname} {cr}")
                continue
            output = [
                nickname,
                f"{data_yield:.2f}",
                f"{cr_process_yield:.2f}",
                f"{cr_process_yield/total_mc:.2%}",
                f"{signal_yield:.2f}",
                f"{signal_yield/total_mc:.2%}",
            ]
            output = ' & '.join(output)
            output = output.replace("%", "\\%")
            print(output)

if __name__ == "__main__":
    dijets_config = collinearw.ConfigMgr.open("dijets_variation.pkl")
    nominal_config = collinearw.ConfigMgr.open("reco_band.pkl")
    dijets_set = dijets_config.get_process_set("dijets")
    dijets_nominal = dijets_set.nominal
    dijets_nominal.name = "dijets_mean"
    dijets_up, dijets_down = dijets_set.systematics
    dijets_up.name, dijets_down.name = "dijets_up", "dijets_down"
    dijets_up.systematic, dijets_down.systematic = None, None
    nominal_config.append_process(dijets_nominal)
    nominal_config.append_process(dijets_up)
    nominal_config.append_process(dijets_down)
    generate_background_purity_table(nominal_config)

    collinearw.run_PlotMaker.plot_processes(
        nominal_config,
        "plot_dijets_variation",
        process=["dijets", "dijets_mean", "dijets_down", "dijets_up"],
        rname_filter=["*rA*"],
        logy=False,
        yrange=(0,5000),
    )
