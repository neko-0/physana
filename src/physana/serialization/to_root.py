import warnings

FOUND_ROOT = True
try:
    import ROOT
except ImportError:
    warnings.warn("Cannot import ROOT module!")
    FOUND_ROOT = False


def dump_config_histograms(config, filename):
    if not FOUND_ROOT:
        warnings.warn("ROOT module is not found!")

    tfile = ROOT.TFile.Open(filename, "RECREATE")
    tfile.cd()

    with ROOT.TFile.Open(filename, "RECREATE") as tfile:
        tfile.cd()
        for process_set in config.process_sets:
            for process in process_set:
                if process.systematics:
                    tag = process.systematics.tag
                else:
                    tag = "NOSYS"

                histo_gen = (h for region in process for h in region)
                for histo in histo_gen:
                    name = "_".join([process.name, histo.parent.name, histo.name, tag])
                    root_histo = histo.root
                    root_histo.SetName(name)
                    root_histo.Write()
