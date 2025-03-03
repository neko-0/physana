import collinearw
from collinearw import run_PlotMaker, ConfigMgr
from collinearw.strategies.systematics.core import compute_quadrature_sum, compute_systematics
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

def compute_band(ifile):
    config = ConfigMgr.open(ifile)

    group = {}
    group.update( config.generate_systematic_group("JET-GroupedNP-up", ("*JET_GroupedNP*", "*JET*up*", "")) )
    group.update( config.generate_systematic_group("JET-GroupedNP-down", ("*JET_GroupedNP*", "*JET*down*", "")) )

    group.update( config.generate_systematic_group("JET-JER-up", ("*JET*JER*", "*JET*up*", "")) )
    group.update( config.generate_systematic_group("JET-JER-down", ("*JET*JER*", "*JET*down*", "")) )

    group.update( config.generate_systematic_group("JET-EtaIntercalibration-NonClosure-up", ("*JET*Eta*", "*JET*up*", "")) )
    group.update( config.generate_systematic_group("JET-EtaIntercalibration-NonClosure-down", ("*JET*Eta*", "*JET*down*", "")) )

    group.update( config.generate_systematic_group("Lepton-EL-Weight-up", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*up*")) )
    group.update( config.generate_systematic_group("Lepton-EL-Weight-down", ("*leptonWeight*EL*", "NoSys", "*leptonWeight*down*")) )
    group.update( config.generate_systematic_group("Lepton-MU-Weight-up", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*up*")) )
    group.update( config.generate_systematic_group("Lepton-MU-Weight-down", ("*leptonWeight*MUON*", "NoSys", "*leptonWeight*down*")) )

    group.update( config.generate_systematic_group("JVT-up", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*up*")) )
    group.update( config.generate_systematic_group("JVT-down", ("*jvtWeight_JET_JvtEfficiency*", "NoSys", "*down*")) )

    group.update( config.generate_systematic_group("Trig-EL-up", ("*trigWeight_EL*_*", "NoSys", "*up*")) )
    group.update( config.generate_systematic_group("Trig-EL-down", ("*trigWeight_EL*", "NoSys", "*down*")) )

    group.update( config.generate_systematic_group("Trig-MUON-up", ("*trigWeight_MUON*_*", "NoSys", "*up*")) )
    group.update( config.generate_systematic_group("Trig-MUON-down", ("*trigWeight_MUON*", "NoSys", "*down*")) )

    group.update( config.generate_systematic_group("bTagWeight-up", ("*bTagWeight*", "NoSys", "*up*")) )
    group.update( config.generate_systematic_group("bTagWeight-down", ("*bTagWeight*", "NoSys", "*down*")) )

    # computing min max of scaling uncertainties
    compute_systematics(config, "wjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "wjets_2211_NNPDF30nnlo_hessian", "stdev")
    compute_systematics(config, "zjets_2211_MUR_MUF_Scale", "min_max")
    compute_systematics(config, "zjets_2211_NNPDF30nnlo_hessian", "stdev")
    compute_systematics(config, "ttbar_ren_fac_scale", "min_max")
    compute_systematics(config, "ttbar_ISR_scale", "min_max")
    compute_systematics(config, "ttbar_FSR_scale", "min_max")
    compute_systematics(config, "ttbar_NNPDF30_PDF", "stdev")
    compute_systematics(config, "singletop_ren_fac_scale", "min_max")

    fakes_theory = {}

    ttbar_theory = {}
    ttbar_theory.update( config.generate_systematic_group("ttbar-ren_fac_scale-up", ("*ttbar*ren_fac_scale*", "min_max", "max")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-ren_fac_scale-down", ("*ttbar*ren_fac_scale*", "min_max", "min")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-ISR_scale-up", ("*ttbar*ISR_scale*", "min_max", "max")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-ISR_scale-down", ("*ttbar*ISR_scale*", "min_max", "min")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-FSR_scale-up", ("*ttbar*FSR_scale*", "min_max", "max")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-FSR_scale-down", ("*ttbar*FSR_scale*", "min_max", "min")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-NNPDF30_PDF-up", ("*ttbar*NNPDF30_PDF*", "stdev", "std_up")) )
    ttbar_theory.update( config.generate_systematic_group("ttbar-NNPDF30_PDF-down", ("*ttbar*NNPDF30_PDF*", "stdev", "std_down")) )

    singletop_theory = {}
    singletop_theory.update( config.generate_systematic_group("singletop-ren_fac_scale-up", ("*singletop_ren_fac_scale*", "min_max", "max")) )
    singletop_theory.update( config.generate_systematic_group("singletop-ren_fac_scale-down", ("*singletop_ren_fac_scale*", "min_max", "min")) )

    zjets_theory = {}
    zjets_theory.update( config.generate_systematic_group("zjets_2211-MUR_MUF-up", ("*zjets*", "min_max", "max")) )
    zjets_theory.update( config.generate_systematic_group("zjets_2211-MUR_MUF-down", ("*zjets*", "min_max", "min")) )
    zjets_theory.update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian-up", ("*zjets*", "stdev", "std_up")) )
    zjets_theory.update( config.generate_systematic_group("zjets_2211-NNPDF30nnlo_hessian-down", ("*zjets*", "stdev", "std_down")) )

    wjets_theory = {}
    wjets_theory.update( config.generate_systematic_group("wjets_2211-MUR_MUF-up", ("*wjets*", "min_max", "max")) )
    wjets_theory.update( config.generate_systematic_group("wjets_2211-MUR_MUF-down", ("*wjets*", "min_max", "min")) )
    wjets_theory.update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian-up", ("*wjets*", "stdev", "std_up")) )
    wjets_theory.update( config.generate_systematic_group("wjets_2211-NNPDF30nnlo_hessian-down", ("*wjets*", "stdev", "std_down")) )

    fakes_theory.update(ttbar_theory)
    fakes_theory.update(singletop_theory)
    fakes_theory.update(wjets_theory)
    fakes_theory.update(zjets_theory)

    theories = {
        "wjets" : wjets_theory,
        "zjets" : zjets_theory,
        "ttbar" : ttbar_theory,
        "singletop" : singletop_theory,
        "fakes" : fakes_theory,
    }

    use_ratio = True

    for process_name in config.list_processes():
        for name, syst_list in group.items():
            print(f"Doing process {process_name}, group {name}")
            if "-up" in name:
                name = name.replace("-up", "")
                config = compute_quadrature_sum(name, "experimental", config, process_name, syst_list, "up", copy=False, use_ratio=use_ratio)
            else:
                name = name.replace("-down", "")
                config = compute_quadrature_sum(name, "experimental", config, process_name, syst_list, "down", copy=False, use_ratio=use_ratio)

        if "wjets" in process_name:
            theory = theories["wjets"]
        elif "zjets" in process_name:
            theory = theories["zjets"]
        elif "ttbar" in process_name:
            theory = theories["ttbar"]
        elif "singletop" in process_name:
            theory = theories["singletop"]
        elif "fakes" in process_name:
            theory = theories["fakes"]
        else:
            continue

        for name, syst_list in theory.items():
            print(f"Doing process {process_name}, group {name}")
            if "wjets" in process_name or "zjets" in process_name:
                theory_legend_name = "W/Z+jets theory"
            else:
                theory_legend_name = "bkgd theory"
            if "-up" in name:
                name = name.replace("-up", "")
                config = compute_quadrature_sum(name, theory_legend_name, config, process_name, syst_list, "up", copy=False, use_ratio=use_ratio)
            else:
                name = name.replace("-down", "")
                config = compute_quadrature_sum(name, theory_legend_name, config, process_name, syst_list, "down", copy=False, use_ratio=use_ratio)

    config.save(f"band_{ifile}.pkl")
