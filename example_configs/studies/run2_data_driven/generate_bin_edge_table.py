import collinearw
import pandas as pd
import numpy as np
import pathlib


def main():

    config_file = "run2_2211_nominal.pkl"
    config = collinearw.ConfigMgr.open(config_file)

    hist_selector = {"DeltaRLepJetClosest100", "mjj", "wPt", "jet1Pt", "Ht30", "nJet30"}

    selected_hist = [h for h in config.histograms if h.name in hist_selector]

    bin_edge_df = pd.DataFrame()
    bin_edge_df["Observable"] = pd.Series([h.xtitle for h in selected_hist])
    bin_edge_df["Binning"] = pd.Series([ f"[{','.join(map(str, np.round(h.bins,1)))}]" for h in selected_hist ])

    bin_edge_df = bin_edge_df.set_index(["Observable"])

    with pd.option_context("max_colwidth", 1000):
        print(
            bin_edge_df.to_latex(
                multicolumn=True,
                header=True,
                index_names=True,
                index=True,
                column_format='p{0.5\linewidth} | p{0.5\linewidth}',
             )
        )

if __name__ == "__main__":
    main()
