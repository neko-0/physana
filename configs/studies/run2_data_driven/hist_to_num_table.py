import collinearw
import pandas as pd
import numpy as np
import pathlib

filename = "fakes_2211_nominal.pkl"

config = collinearw.ConfigMgr.open(filename)

data = "data"
signal = "wjets_2211"
mcs = ["ttbar", "zjets_2211", "singletop", "diboson_powheg", "wjets_2211_tau", "fakes"]

output = {}
data = config.get_process(data)
for region in data.regions:
    output[region.name] = {}
    for hist in region.histograms:
        if isinstance(hist, collinearw.Histogram):
            hist_df = pd.DataFrame()
            bins = hist.bins
            hist_df["bin edge"] = pd.Series(bins)
            hist_df[data.name] = pd.Series(hist.bin_content[1:])
            for mc in [signal]+mcs:
                try:
                    content = config.get(f"{mc}//nominal//{region.name}//{hist.name}").bin_content[1:]
                except KeyError:
                    content = np.zeros(bins.shape)
                hist_df[mc] = pd.Series(content)
            hist_df["total"] = hist_df[signal] + hist_df[mcs].sum(axis=1)
            hist_df["bgks sum"] = hist_df[mcs].sum(axis=1)
            hist_df["bgks/signal"] = hist_df["bgks sum"] / hist_df[signal]
            hist_df["data/total"] = hist_df[data.name]/hist_df["total"]

            for mc in mcs:
                hist_df[f"{mc} norm to bkgs"] = hist_df[mc] / hist_df["bgks sum"]

            hist_df["(data-bkg_sum)/total"] = (hist_df["data"]-hist_df["bgks sum"]) / hist_df["total"]

            output[region.name][hist.name] = hist_df

output["ratio"] = {}
output["ratio"]["jet1Pt"] = output["muon_collinear_reco_ABCD-fake-MU-rA_"]["jet1Pt"] / output["electron_collinear_reco_ABCD-fake-EL-rA_"]["jet1Pt"]

for _r in output:
    for _obs in output[_r]:
        ofile = pathlib.Path(f"hist_table/{_r}/{_obs}.html")
        ofile.parent.mkdir(parents=True, exist_ok=True)
        with open(ofile, "w") as f:
            f.write(output[_r][_obs].to_html())
