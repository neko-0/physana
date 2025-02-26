import collinearw

def main():

    config = collinearw.ConfigMgr.open("band_unfold.pkl")

    collinearw.run_PlotMaker.plot_purity(config, plot_response=False)


if __name__ == "__main__":
    main()
