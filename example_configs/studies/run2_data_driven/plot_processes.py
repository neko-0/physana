from collinearw import ConfigMgr, run_PlotMaker, histManipulate, PlotMaker
from collinearw.serialization import Serialization
from collinearw.histMaker import weight_from_hist
from collinearw.strategies.systematics.core import compute_quadrature_sum, compute_systematics, compute_process_set_systematics

import functools
import os
import logging
import dask
from dask_jobqueue import SLURMCluster
from dask.distributed import Client
from dask.distributed import as_completed as dask_as_completed
from compute_band import compute_band


logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


doEventLevel = True
doCorr = True


def main():

    run2 = ConfigMgr.open("run2_2211.pkl")

    process_filter = run2.list_processes()
    process_filter.remove("diboson_powheg")

    run_PlotMaker.plot_pset_systematics(
        run2,
        use_mp=False,
        process_filter=process_filter,
    )


if __name__ == "__main__":
    main()
