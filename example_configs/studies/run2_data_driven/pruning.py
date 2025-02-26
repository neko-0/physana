import collinearw


def main():

    config_file = "run2_2211.pkl"
    config = collinearw.ConfigMgr.open(config_file)
    print("start pruning")
    # clean up BCD regions after fakes
    # simply lookup region with pattern *ABCD*rB* etc.
    for pset in config.process_sets:
        for p in pset:
            for region in ["rB", "rC", "rD"]:
                for r in p.list_regions(f"*ABCD*{region}*"):
                    p.remove_region(r)
            # clean up transfer factor histogram
            for rname in p.list_regions("*ABCD*rA*"):
                _region = p.get_region(rname)
                for h in _region.histograms:
                    if h.type == "tf":
                        _region.remove_histogram(h)

    config.save("prune_run2_2211.pkl")

if __name__ == "__main__":
    main()
