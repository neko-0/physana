from ...histMaker import HistMaker, histogram_eval

# from .rootbackend import replica_weights
from .npbackend import replica_weights, replica_weights_nb

# import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import logging

log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)


def from_array_wrapper(hist, *args, **kwargs):
    hist.from_array(*args, **kwargs)
    return hist


class BootstrapHistMaker(HistMaker):
    """
    Same HistMaker but with custom histogram loop
    """

    def __init__(self, *args, bootstrap_backend='numpy', init_seed=None, **kwargs):
        log.debug("Using BootstrapHistMaker")
        super().__init__(*args, **kwargs)
        self.bootstrap_backend = bootstrap_backend
        self.tot_replica_w = None
        self.do_total_replica = True
        self._current_batch = -1
        self.use_mp = False
        self._cache = []
        if self.use_mp:
            self._executor = ProcessPoolExecutor(10)
        else:
            self._executor = None
        self.multithread_hist_fill = False
        self.step_size = 1_000_000
        self.init_seed = init_seed or time.time()

    """
    def process_ttree_resolve(self, *args, **kwargs):
        ttree, sys_name, sys_weight = super().process_ttree_resolve(*args, **kwargs)
        if ttree is None:
            return ttree, sys_name, sys_weight
        if self.do_total_replica:
            self.tot_replica_w = replica_weights(ttree.num_entries, 10, tolist=False)
        return ttree, sys_name, sys_weight
    """

    def _finalise(self):
        self._current_batch = -1
        self.tot_replica_w = None
        if self._executor is not None:
            self._executor.shutdown()
        self._cache = []

    def _histogram_loop(self, histograms, mask, event, weights, sumW2):
        # currently only support ROOT version
        # start_t = time.perf_counter()
        log.debug("using bootstrap looping method")
        # initializing random seed
        # current_process = histograms[0].parent.parent
        # seed = f"{current_process.name}_{current_process.systematic}"
        # seed = int.from_bytes(seed.encode(), "little")
        seed = int(self.init_seed + self._current_batch)
        n_repli = histograms[0].nreplica  # assume all replica are same size
        if self._current_batch != self._tree_ibatch:
            self._current_batch = self._tree_ibatch
            if self.do_total_replica and self.bootstrap_backend == "root":
                self.tot_replica_w = replica_weights(
                    len(mask), n_repli, tolist=False, seed=seed
                )
            elif self.do_total_replica and self.bootstrap_backend == "numpy":
                self.tot_replica_w = replica_weights_nb(len(mask), n_repli, seed=seed)
        if not self.do_total_replica:
            replica_w = replica_weights(len(weights[mask]), n_repli, seed=seed)
            for hist in histograms:
                hist.replica_w = replica_w
        else:
            if self.bootstrap_backend == "root":
                m_w = self.tot_replica_w[mask].ravel()  # only use by ROOT version
            else:
                m_w = self.tot_replica_w[mask]
            for hist in histograms:
                hist.replica_w = m_w
                if not self.use_mp:
                    continue
                fdata = histogram_eval(event, mask, *hist.observable)
                # hist_w = weights[mask]
                ft = self._executor.submit(
                    from_array_wrapper, hist, *fdata, weights[mask], sumW2[mask]
                )
                self._cache.append(ft)
            # breakpoint()
            if not self.use_mp:
                # call parent histogram loop
                super()._histogram_loop(histograms, mask, event, weights, sumW2)
                return
            for ft in as_completed(self._cache):
                filled_hist = ft.result()
                for hist in histograms:
                    if hist.name != filled_hist.name:
                        continue
                    hist.bootstrap = filled_hist.bootstrap
                    hist.replica_w = None
                    break

        # print(f"After hist loop {time.perf_counter()-start_t}s")
