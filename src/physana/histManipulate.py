import numpy as np
import fnmatch
import multiprocessing as mp
import concurrent.futures
import os
import numpy
import time
import sys
import logging

from . import run_PlotMaker
from .histo import Region
from .systematics import SystematicsBase
from .configs import ConfigMgr
from .strategies.abcd import abcd
from .strategies.correction import Correction
from .histmaker.interface import run_HistMaker
from .histmaker.interface import refill_process
from .serialization import Serialization


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


# ==============================================================================


class HistManipulate:
    def __init__(self):
        self.class_name = "_HistManipulate_"
        self.purpose = "_"

    def Divide(self, configMgr, regionA, regionB):
        for p in configMgr.processes:
            A = p.get_region(regionA)
            B = p.get_region(regionB)
            if not A or not B:
                logger.warning("ERROR Could not find both regions! Not dividing!")
                continue
            r = Region(f"Divide_{regionA}_over_{regionB}", "dummy", "dummy", "eff")
            for ah in A.histograms:
                # h = Histogram(ah.observable, ah.bins, ah.xmin, ah.xmax, ah.xtitle, ah.ytitle, ah.color)
                h = ah.copy()
                bh = B.get_histogram(ah.observable)
                h.replace(
                    np.divide(
                        ah.histogram,
                        bh.histogram,
                        out=np.zeros_like(ah.histogram),
                        where=bh.histogram != 0,
                    )
                )
                r.add_histogram(h, "ratio")
            p.add_region(r)

    # ==========================================================================

    # ==========================================================================
    def _subtract_process(
        self,
        configMgr,
        process,
        backgrounds,
        bg_fluctuation=None,
        process_fluctuation=None,
    ):
        """
        Internal method for subtracting processes.

        Args:
            configMgr (obj:ConfigMgr) : path to the configuration manager object.

            process (str) : name of the process, usually is 'data'.

            backgrounds (list) : name of the backgrounds

            bg_fluctuation (tuple) : percentage of fluctuation, and mode. e.g (100, "H")
                                    mean flucutate high with 100%

            process_fluction (tuple) : poisson fluctuation for process. e.g. (multiplier, "poisson,H")

        Returns:
            background subtracted process.
        """
        my_process = configMgr.get_process(process).copy()
        add_tag = ""

        if not (process_fluctuation is None):
            my_process.scale(process_fluctuation[0], process_fluctuation[1])
            if "H" in process_fluctuation[1]:
                add_tag += f"_PH{process_fluctuation[0]}"
            elif "L" in process_fluctuation[1]:
                add_tag += f"_PL{process_fluctuation[0]}"
            else:
                logger.warning("you need to specific fluctuation high/low (H/L)")
        bg_name = ""

        for bk in backgrounds:
            bk_process = configMgr.get_process(bk).copy()
            if not (bg_fluctuation is None):
                bk_process.scale(bg_fluctuation[0] / 100.0, bg_fluctuation[1])
            my_process = my_process - bk_process
            bg_name = bg_name + "_" + bk + "_"

        if not (bg_fluctuation is None):
            add_tag += f"_{bg_fluctuation[1]}{bg_fluctuation[0]}"

        my_process.name = f"{process}_sub_{bg_name}{add_tag}"
        logger.usr_log(f"subtracted process: {process}_sub_{bg_name}{add_tag}")
        configMgr.append_process(my_process)

        return my_process

    def SubtractProcess(
        self, configMgr, process, backgrounds, bg_fluctuation, process_fluctuation=False
    ):
        """
        Subtracting processes.

        Args:
            configMgr (obj:ConfigMgr) : path to the configuration manager object.

            process (str) : name of the process, usually is 'data'.

            backgrounds (list) : name of the backgrounds

            bg_fluctuation (list) : flucation in backgrounds, e.g [5,100,200] in percentage

            process_fluction (bool) : turning on/off the fluctuation for process. Poisson uncertainty.

        Returns:
            no return
        """

        bg_fluc_mode = ["H", "L"]  # fluctuate high or low for bg
        sig_fluc_mode = ["H", "L"]  # fluctuate high or low for signal

        if process_fluctuation:
            if bg_fluctuation:
                for bg_fluc in bg_fluctuation:
                    for bg_mode in bg_fluc_mode:
                        for sig_mode in sig_fluc_mode:
                            self._subtract_process(
                                configMgr,
                                process,
                                backgrounds,
                                (bg_fluc, bg_mode),
                                (1, f"poisson,{sig_mode}"),
                            )

                        self._subtract_process(
                            configMgr,
                            process,
                            backgrounds,
                            (bg_fluc, bg_mode),
                            None,
                        )
            else:
                for sig_mode in sig_fluc_mode:
                    self._subtract_process(
                        configMgr,
                        process,
                        backgrounds,
                        None,
                        (1, f"poisson,{sig_mode}"),
                    )
        else:
            if bg_fluctuation:
                for bg_fluc in bg_fluctuation:
                    for bg_mode in bg_fluc_mode:
                        self._subtract_process(
                            configMgr, process, backgrounds, (bg_fluc, bg_mode), None
                        )

        self._subtract_process(configMgr, process, backgrounds, None, None)

        configMgr.output_processed_tag += self.class_name
        configMgr.output_processed_tag += self.purpose

    @staticmethod
    def Subtract_MC(
        config,
        data_name="data",
        rename="",
        skip_processes=None,
        skip_patterns=None,
        list_processes=None,
        systematic=None,
        remove_neg=True,
        copy=True,
    ):
        if not isinstance(config, ConfigMgr):
            raise TypeError(f"Invalid type {type(config)}")

        logger.info(f"doing subtraction for {data_name}")
        config_copy = config.copy() if copy else config

        if list_processes is None:
            list_processes = config_copy.list_processes()

        if data_name not in list_processes:
            logger.critical(f"Unable to find {data_name}")
            return None

        if skip_processes is None:
            skip_processes = []
        skip_processes = set(skip_processes + [rename])

        if skip_patterns is None:
            skip_patterns = []

        _syst_name = None
        _syst_full_name = None
        _syst_type = None
        if systematic:
            logger.info(f"Getting systmeatic {systematic}")

        # note data_copy here is a Process object
        data_pset = config_copy.get_process_set(data_name)
        data_copy = data_pset.get(systematic).copy()
        data_copy.dtype = f"subtracted-{data_name}"
        append_mode = None
        found_match_sys = False
        for process_name in list_processes:
            if (
                process_name == data_name
                or process_name in skip_processes
                or any(fnmatch.fnmatch(process_name, p) for p in skip_patterns)
            ):
                logger.info(f"\t--skip {process_name}")
                continue

            if systematic:
                process = config_copy.get_process_set(process_name).get(systematic)
            else:
                process = config_copy.get_process(process_name)

            show_sys = ""
            if process.systematics:
                show_sys = systematic
                if not found_match_sys:
                    found_match_sys = True
                    _syst_name = process.systematics.name
                    _syst_full_name = process.systematics.full_name
                    _syst_type = process.systematics.sys_type
                    append_mode = "merge"

            data_copy.sub(process)
            logger.info(f"\t--subtraced {process_name} {show_sys}")

        if systematic is not None:
            if not found_match_sys:
                logger.warning(
                    f"{systematic} always return Nominal for {process_name}!"
                )
            else:
                data_copy.systematics = SystematicsBase(
                    _syst_name, _syst_full_name, "dummy", _syst_type
                )

        if rename:
            data_copy.name = rename

        if remove_neg:
            for r in data_copy.regions:
                for h in r.histograms:
                    h.remove_negative_bin()

        config_copy.append_process(data_copy, mode=append_mode)

        return config_copy

    @staticmethod
    def Sum_MC(
        config,
        rename="",
        skip_processes=None,
        skip_patterns=None,
        systematic=None,
        remove_neg=True,
        copy=True,
    ):
        if not isinstance(config, ConfigMgr):
            raise TypeError(f"Invalid type {type(config)}")

        config_copy = config.copy() if copy else config

        list_processes = config_copy.list_processes()

        if skip_processes is None:
            skip_processes = []
        skip_processes = set(skip_processes + [rename])

        if skip_patterns is None:
            skip_patterns = []

        # filtering processes
        filtered_processes = []
        for process_name in list_processes:
            if process_name in skip_processes:
                logger.info(f"\t--skip {process_name} by name")
                continue
            if any(fnmatch.fnmatch(process_name, p) for p in skip_patterns):
                logger.info(f"\t--skip {process_name} by pattern")
                continue
            filtered_processes.append(process_name)

        _syst_name = None
        _syst_full_name = None
        _syst_type = None
        if systematic:
            logger.info(f"Getting systmeatic {systematic}")

        # note data_copy here is a Process object
        first_pset = config_copy.get_process_set(filtered_processes[0])
        first_copy = first_pset.get(systematic).copy()
        first_copy.dtype = f"sum-{first_copy.name}"
        append_mode = None
        found_match_sys = False
        for process_name in filtered_processes[1:]:
            if systematic:
                process = config_copy.get_process_set(process_name).get(systematic)
            else:
                process = config_copy.get_process(process_name)

            show_sys = ""
            if process.systematics:
                show_sys = systematic
                if not found_match_sys:
                    found_match_sys = True
                    _syst_name = process.systematics.name
                    _syst_full_name = process.systematics.full_name
                    _syst_type = process.systematics.sys_type
                    append_mode = "merge"

            first_copy.add(process)
            logger.info(f"\t--add {process_name} {show_sys}")

        if systematic is not None:
            if not found_match_sys:
                logger.warning(
                    f"{systematic} always return Nominal for {process_name}!"
                )
            else:
                first_copy.systematics = SystematicsBase(
                    _syst_name, _syst_full_name, "dummy", _syst_type
                )

        if rename:
            first_copy.name = rename

        if remove_neg:
            for r in first_copy.regions:
                for h in r.histograms:
                    h.remove_negative_bin()

        config_copy.append_process(first_copy, mode=append_mode)

        return config_copy

    @staticmethod
    def Divide_Process(proc_a, proc_b, name=None):
        region_a = set(proc_a.list_regions())
        region_b = set(proc_b.list_regions())

        # getting the region names that are both in proc_a and proc_b
        regions = region_a.intersection(region_b)

        c_proc = proc_a.copy()
        c_proc.clear()
        c_proc.name = name or f"{proc_a.name}/{proc_b.name}"

        for rname in regions:
            r_a = proc_a.get_region(rname)
            r_b = proc_b.get_region(rname)
            hist_set_a = set(r_a.list_observables())
            hist_set_b = set(r_b.list_observables())
            hist_set = hist_set_a.intersection(hist_set_b)

            # make a copy of region
            c_region = r_a.copy()
            c_region.clear()
            for hname in hist_set:
                ha = r_a.get_observable(hname)
                hb = r_b.get_observable(hname)
                c_region.add_histogram(ha / hb)
            c_proc.add_region(c_region)

        return c_proc


# ==============================================================================


# ==============================================================================
def run_ABCD_TF(config, process, systematic=None, copy=True, **kwargs):
    """
    computing transfer factor
    Args:
        config : configMgr.ConfigMgr
            instance of ConfigMgr.

        process : str or list(str)
            name of the process(es) for computing the transfer factor.

        systematic : tuple(str)
            systematic full name tuple,
            use for retrieving Process object through core.ProcessSet.get()

        copy : bool, default = True,
            make copy of the input ConfigMgr instance, so it won't affect the
            original instance.
    """

    if isinstance(config, ConfigMgr):
        my_configMgr = config.copy() if copy else config
    else:
        my_configMgr = ConfigMgr.open(config)

    if isinstance(process, str):
        process = [process]

    for p in process:
        tf_regions = abcd.transfer_factor(my_configMgr, p, systematic, **kwargs)
        if tf_regions:
            my_process = my_configMgr.get_process_set(p).get(systematic)
            for tf in tf_regions:
                my_process.add_region(tf)

    abcd.set_tf(my_configMgr)

    return my_configMgr


# ------------------------------------------------------------------------------


def run_ABCD_Fakes(config, const_tf, *, ext_tf=None, systematic=None, copy=True):
    if isinstance(config, ConfigMgr):
        my_configMgr = config.copy() if copy else config
    else:
        my_configMgr = ConfigMgr.open(config)

    if not abcd.check_tf(config):
        logger.critical("Cannot find abcd tf")
        return None

    fakes = abcd.fakes(my_configMgr, const_tf, ext_tf=ext_tf, systematic=systematic)
    if fakes:
        for fake_process in fakes:
            my_configMgr.append_process(fake_process, mode="merge")

    return my_configMgr


# ------------------------------------------------------------------------------


def _run_ABCD_Fakes_Estimation(config, *, process="data", const_tf=False, ext_tf=None):
    sub_config = HistManipulate.Subtract_MC(config, "data", "subtracted_data")
    tf_config = run_ABCD_TF(sub_config, "subtracted_data")
    fake_config = run_ABCD_Fakes(tf_config, const_tf, ext_tf=ext_tf)
    fake_config.remove_process_set("subtracted_data")
    return fake_config


# ------------------------------------------------------------------------------
def run_ABCD_Fakes_EventLevel(
    config,
    tf_config,
    tf_pname,
    tf_obs,
    *,
    full_run=True,
    match_tf_process=False,
    skip_process=None,
    use_mp=False,
    step_size=None,
    force_refill=False,
    systematic=None,
    copy=False,
    **kwargs,
):
    """
    Runner method for ABCD fakes estimation on event level

    Args:
        config (physana.configMgr.ConfigMgr) : configMgr object.

        tf_conifg (tuple(collinarw.configMgr.ConigMgr) : TF config.

        tf_pname (str) : TF process name.

        tf_obs( tuple(list(str), str) or dict(str:tuple(list(str),str)) ) :
            see tf_obs in abcd.event_level_fakes

    """

    m_config = ConfigMgr.open(config)

    # use open method when string is passed
    if isinstance(tf_config, str):
        tf_config = ConfigMgr.open(tf_config)

    skip_process = skip_process or []
    if isinstance(skip_process, str):
        skip_process = [skip_process]

    if isinstance(tf_pname, list):
        fakes = {}
        for m_tf_pname in tf_pname:
            fakes[m_tf_pname] = abcd.event_level_fakes(
                m_config,
                tf_config,
                m_tf_pname,
                tf_obs,
                match_tf_process=match_tf_process,
                skip_process=skip_process,
                use_mp=use_mp,
                step_size=step_size,
                systematic=systematic,
                **kwargs,
            )
    else:
        fakes = abcd.event_level_fakes(
            m_config,
            tf_config,
            tf_pname,
            tf_obs,
            match_tf_process=match_tf_process,
            skip_process=skip_process,
            use_mp=use_mp,
            step_size=step_size,
            systematic=systematic,
            **kwargs,
        )

    # this copy step is required. avoid copying process_sets
    # since we are going reset it anyway.
    skip_deep_copy = [
        "process_sets",
        "corrections",
        "meta_data",
        "systematics",
        "regions",
        "aux_regions",
        "histograms",
        "histograms2D",
        "reserved_branches",
        "default_weight",
    ]
    new_config = ConfigMgr.copy_from(m_config, skip=skip_deep_copy, shallow=True)
    new_config.process_sets = []

    if fakes:
        if isinstance(fakes, dict):
            buff = []
            for tf_p, fake_processes in fakes.items():
                _new_config = new_config.copy()
                for fake_process in fake_processes:
                    _new_config.append_process(fake_process, mode="merge")
                if match_tf_process:  # match the process name to the one in tf
                    _fake_only = _new_config
                else:
                    _fake_only = HistManipulate.Subtract_MC(
                        _new_config, "data", f"fakes-{tf_p}", skip_process
                    )
                    for _pname in _fake_only.list_processes():
                        if _pname != f"fakes-{tf_p}":
                            _fake_only.remove_process_set(_pname)
                buff.append(_fake_only)
            elevel_fake = ConfigMgr.merge(buff)
        else:
            for fake_process in fakes:
                new_config.append_process(fake_process, mode="merge")
            if match_tf_process:
                elevel_fake = new_config
            else:
                elevel_fake = HistManipulate.Subtract_MC(
                    new_config,
                    "data",
                    "fakes",
                    skip_process,
                    systematic=systematic,
                    copy=False,
                )
    del fakes

    if full_run and not match_tf_process:
        if force_refill:
            copy = False
            logger.info("refilling configs")
            refill_config = m_config.copy()
            refill_config.clear_process()
            refill_config.prepared = False
            m_config = run_HistMaker(refill_config, split_type="region", n_workers=16)
        logger.info("completing fakes output file.")
        if copy:
            shallow_list = skip_deep_copy.remove("process_sets")
            m_config = ConfigMgr.copy_from(m_config, skip=shallow_list, shallow=True)
        sub_config = HistManipulate.Subtract_MC(
            m_config,
            "data",
            "sub_data",
            skip_process,
            systematic=systematic,
            copy=False,
        )
        tf_config = run_ABCD_TF(sub_config, "sub_data", systematic, copy=False)
        fake_config = run_ABCD_Fakes(
            tf_config, False, systematic=systematic, copy=False
        )
        fake_config.remove_process_set("sub_data")
        # replace old fakes with event level fake
        for fake_name, fake_pset in elevel_fake.pset_items():
            if fnmatch.fnmatch(fake_name, "fakes*"):
                fake_config.remove_process(fake_name, systematic=systematic)
                for fake_p in fake_pset:
                    fake_p.process_type = 'fakes'
                    fake_p.dtype = 'fakes'
                    fake_p.title = 'fakes'
                    fake_config.append_process(fake_p, mode="merge")
        new_config = fake_config
    else:
        new_config = elevel_fake

    return new_config


# ------------------------------------------------------------------------------
def _run_abcd_fakes_estimation(
    config,
    tf_param,
    data="data",
    signal="wjets",
    correction=None,
    skip_process=[],
    *,
    use_mp=False,
    ext_tf_config=None,
    systematic=None,
    copy=True,
    correlation_correction=None,
    executor=None,
    as_completed=None,
    workers=None,
):
    """
    complete regular routine for ABCD fakes estimation. It basically utilize the
    methods from above, and return a config object at the end with fakes estamation.

    Args:
        config (physana.ConfigMgr) : an instance of physana.ConfigMgr

        tf_param (dict(str:(str,str))) : transfer factor parametrization, the format follows
            from the run_ABCD_Fakes_EventLevel, e.g. {region_name: (("obs","obs2"), "hist_name")}

        data (str) : name of the process for data. the default is 'data'

        signal (str) : name of the process for signal, the default is 'wjets'

        correction ((str,histogram) or (str, str)) : observable to be used for signal
            correction and the corresponding distribution. An histogram object or path
            to the pickled histogram object is fine. This might need more finer controller
            in later version.

        skip_process (list(str)) : list of process to be skipped during the fake
            estimation procedure.
    """
    t_start = time.time()

    # use the open method in case that config is a string
    config = ConfigMgr.open(config)

    if correction is None:
        # if no correction is provided, the usual fake esstimation will be run

        logger.info("Getting transfer factor.")
        # subtracting all MCs from the data for deriving
        if ext_tf_config is None:
            sub_config = HistManipulate.Subtract_MC(
                config,
                data,
                "sub_data",
                skip_processes=skip_process,
                systematic=systematic,
                copy=copy,
            )
            tf_config = run_ABCD_TF(sub_config, "sub_data", systematic, copy=False)
        else:
            tf_config = ext_tf_config

        # estimation fakes with event level and parametrized TF
        logger.info("Running fake estimation.")
        fakes = run_ABCD_Fakes_EventLevel(
            config,
            tf_config,
            "sub_data",
            tf_param,
            skip_process=skip_process + ["sub_data"],
            use_mp=use_mp,
            systematic=systematic,
            correlation_correction=correlation_correction,
            executor=executor,
            as_completed=as_completed,
            workers=workers,
        )

        # clean up extra process sets:
        if not copy:
            fakes.remove_process_set("sub_data")
            config.remove_process_set("sub_data")

        return fakes
    else:
        # TODO: check if copy is needed.
        c_config = config.copy() if copy else config

        # adding phase-space correction file
        c_config.corrections.add_correction_file(correction)

        # if it's not copying the whole ConfigMgr object,
        # make a copy of the refilling process to overwrite the correction later.
        signal_copy = None
        if not copy:
            # in the case that no such systematic for the signal process,
            # we need to make sure the nominal will not re-merge later.
            # this require to confirm that systematic of the return process.
            signal_copy = config.get_process_set(signal).get(systematic)
            if systematic is not None and signal_copy.systematic is None:
                signal_copy = None
            else:
                signal_copy = signal_copy.copy()

        logger.info(f"refill process : {signal}")
        refill_process(c_config, [signal], use_mp=use_mp, systematic=systematic)

        if ext_tf_config is None:
            sub_config = HistManipulate.Subtract_MC(
                c_config,
                data,
                "sub_data",
                skip_processes=skip_process,
                systematic=systematic,
                copy=False,
            )
            tf_config = run_ABCD_TF(sub_config, "sub_data", systematic, copy=False)
        else:
            tf_config = ext_tf_config

        fakes = run_ABCD_Fakes_EventLevel(
            c_config,
            tf_config,
            "sub_data",
            tf_param,
            skip_process=skip_process + ["sub_data"],
            use_mp=use_mp,
            systematic=systematic,
            correlation_correction=correlation_correction,
            executor=executor,
            as_completed=as_completed,
            workers=workers,
        )

        # reused the variable name, make a copy of the origin config
        c_config = config.copy() if copy else config
        c_config.append_process(fakes.get_process_set("fakes").get(systematic).copy())
        fakes = None

        # clean up extra process set
        if not copy:
            c_config.remove_process_set("sub_data")
            c_config.remove_process(signal, systematic=systematic)
            if signal_copy:
                c_config.append_process(signal_copy, mode="merge")

        # clear correction container buffers
        c_config.corrections.clear_buffer()

        logger.info(f"fake calculation cost {time.time()-t_start}s")

        return c_config


def _initializer():
    import logging

    logging.basicConfig()
    logging.getLogger().setLevel(logging.CRITICAL)


def _submit_handler1(jobid, func, *args, **kwargs):
    result_config = func(*args, **kwargs)
    return result_config.save(f"_fakes_{jobid}")


def _submit_handler2(queue, func, *args, **kwargs):
    result_config = func(*args, **kwargs)
    queue.put(result_config)
    return True


def _submit_handler3(jobid, *args, **kwargs):
    result_config = _run_abcd_fakes_estimation(*args, **kwargs)
    for pset in result_config.process_sets:
        if not fnmatch.fnmatch(pset.name, "fakes*"):
            result_config.remove_process_set(pset.name)
    result_config.corrections.clear_buffer()
    return result_config


def _submit_handler4(jobid, *args, **kwargs):
    config = _submit_handler3(jobid, *args, **kwargs)
    try:
        os.remove(f"temp_output/ID{jobid}.pkl")
    except:
        pass
    return config.save(f"temp_output/_finished_fakes_{jobid}")


def _result_handler1(result):
    return ConfigMgr.open(result)


def _result_handler2(queue):
    return queue.get()


def _result_handler3(result):
    return result


def _result_handler4(result):
    config = ConfigMgr.open(result)
    os.remove(result)
    return config


_handler = {
    "1": (_submit_handler1, _result_handler1),
    "2": (_submit_handler2, _result_handler2),
    "3": (_submit_handler3, _result_handler3),
    "4": (_submit_handler4, _result_handler4),
}


def run_abcd_fakes_estimation(
    config,
    tf_param,
    systematic=None,
    workers=None,
    mp_context="spawn",  # fork, spawn, forkserver
    handler="4",
    nonclosure=None,
    executor=None,
    as_completed=None,
    enforced_mp=False,
    skim_large_file=True,
    prune=True,
    *args,
    **kwargs,
):
    final_config = None
    p_size = sys.getsizeof(config.process_sets[0].nominal)
    logger.info(f"input config size {sys.getsizeof(config)/1e6} MB.")
    increase_size = p_size * len(systematic) if systematic else p_size
    logger.info(f"expect to increase {increase_size/1e6} MB.")
    skip_process = kwargs.get("skip_process", [])
    if not isinstance(systematic, (list, set)):
        kwargs.update(
            {"executor": executor, "as_completed": as_completed, "workers": workers}
        )
        final_config = _run_abcd_fakes_estimation(
            config, tf_param, systematic=systematic, *args, **kwargs
        )
    else:
        if not kwargs.get("use_mp", False):
            kwargs.update({"use_mp": True})
            skip_process.append("fakes")
            kwargs.update({"skip_process": skip_process})
            first_config = None
            for sys_name in systematic:
                if first_config is None:
                    first_config = _run_abcd_fakes_estimation(
                        config, tf_param, systematic=sys_name, *args, **kwargs
                    )
                else:
                    kwargs.update({"copy": False})
                    first_config = _run_abcd_fakes_estimation(
                        first_config, tf_param, systematic=sys_name, *args, **kwargs
                    )
            final_config = first_config
        else:
            kwargs.update({"copy": False})
            kwargs.update({"use_mp": enforced_mp})

            os_cpu = int(numpy.ceil(0.5 * os.cpu_count()))
            workers = workers or os_cpu
            if os_cpu < workers:
                workers = os_cpu
            logger.info(f"using {min(workers, os_cpu)} workers")

            if isinstance(handler, str):
                submit_handler, result_handler = _handler[handler]
            else:
                submit_handler, result_handler = handler

            do_closing = False
            if executor is None or as_completed is None:
                do_closing = True
                # turning off internal mp to save resources
                # when no exectutor is provided
                mp_context = mp.get_context(mp_context)
                executor = concurrent.futures.ProcessPoolExecutor(
                    workers, mp_context=mp_context, initializer=_initializer
                )
                as_completed = concurrent.futures.as_completed

            # do job submission
            tot_jobs = len(systematic)
            finished_counter = 0
            pending_jobs = []
            finished_jobs = []
            buffer_size = int(numpy.ceil(0.5 * workers))
            buffer_size = buffer_size if buffer_size > 10 else 10
            if skim_large_file:
                # note split method return a generator
                systematic = ConfigMgr.split(
                    config,
                    split_type="systematic",
                    syst_names=systematic,
                    skip_process=skip_process,
                    copy=False,  # no copy is safe since we save it anyway
                    with_name=True,  # get the name of the syst
                )
            start_t = time.time()
            for jobid, sys_name in enumerate(systematic):
                if skim_large_file:
                    # note the split above return pair (syst_name, config)
                    sys_name, submit_config = sys_name
                    submit_config = submit_config.save(f"temp_output/ID{jobid}")
                else:
                    submit_config = config
                logger.info(f"submitting {sys_name}, {jobid}/{tot_jobs}")
                submitted_job = executor.submit(
                    submit_handler,
                    jobid,
                    submit_config,
                    tf_param,
                    systematic=sys_name,
                    *args,
                    **kwargs,
                )
                pending_jobs.append(submitted_job)
            # retrieving submitted jobs
            for job in as_completed(pending_jobs):
                finished_jobs.append(result_handler(job.result()))
                logger.info(f"current fill status {len(finished_jobs)}")
                finished_counter += 1
                if len(finished_jobs) == buffer_size:
                    logger.info(f"reach job buffer size {buffer_size} ...")
                    merged_config = ConfigMgr.intersection(finished_jobs, copy=False)
                    finished_jobs = [merged_config]
                    logger.info(f"batch time cost {time.time()-start_t}s")
                    logger.info(f"total filled {finished_counter}/{tot_jobs}")
                    start_t = time.time()

            logger.info(f"remaining {len(finished_jobs)} finished jobs.")
            final_config = ConfigMgr.intersection(finished_jobs + [config], copy=False)

            # close internal executor
            if do_closing:
                executor.shutdown()

    # if non-closure file is provided, append them to the 'fakes' process.
    if nonclosure:
        m_serial = Serialization()
        nonclosure = m_serial.from_pickle(nonclosure)
        fakes_pset = final_config.get_process_set("fakes")
        for key, _data in nonclosure.items():
            lookup = ("nonclosure", "fakes", key)
            _data.systematic = SystematicsBase("nonclosure", lookup, "fakes")
            fakes_pset.computed_systematics[lookup] = _data

    if prune:
        # clean up BCD regions after fakes
        # simply lookup region with pattern *ABCD*rB* etc.
        for pset in final_config.process_sets:
            for p in pset:
                for region in ["rB", "rC", "rD"]:
                    for r in p.list_regions(f"*ABCD*{region}*"):
                        p.remove_region(r)
                # clean up transfer factor histogram
                for rname in p.list_regions("*ABCD*rA*"):
                    _region = p.get_region(rname)
                    for h in _region.histograms:
                        if h.dtype == "tf":
                            _region.remove_histogram(h)

    return final_config


# ------------------------------------------------------------------------------
# NB: Nominal (None) must be first when running with systematics:
#   - [None, syst1, syst2, ....]
def Subtract_MC(
    config, data_name="data", rename="", systematics=None, copy=True, *args, **kwargs
):
    """
    Wrapper for Histmanipulate.Subtract_MC to handle list of systematics
    """
    systematics = systematics or [None]
    if None not in systematics:
        systematics = [None] + systematics
    c_config = config.copy() if copy else config
    for sys_tuple in systematics:
        c_config = HistManipulate.Subtract_MC(
            c_config,
            data_name,
            rename,
            systematic=sys_tuple,
            copy=False,
            *args,
            **kwargs,
        )
    return c_config


# ==============================================================================
def Sum_MC(config, rename="", systematics=None, copy=True, *args, **kwargs):
    """
    Wrapper for Histmanipulate.Sum_MC to handle list of systematics
    """
    systematics = systematics or [None]
    if None not in systematics:
        systematics = [None] + systematics
    c_config = config.copy() if copy else config
    for sys_tuple in systematics:
        c_config = HistManipulate.Sum_MC(
            c_config,
            rename,
            systematic=sys_tuple,
            copy=False,
            *args,
            **kwargs,
        )
    return c_config


# ==============================================================================


def run_iterative_correction(
    config,
    signal="wjets",
    skip=["wjets_mg", "wjets_FxFx", "wjets_powheg"],
    bkgd=["zjets", "ttbar", "singletop", "diboson"],
    iteration=10,
    do_pre_iterative=True,
    prefix="",
    tf_param=None,
    output_path="./",
    corr_obs="nJet30",
    electron_type=[("electron", "electron_track_iso-pid", "electron")],
    muon_type=[("muon", "pid_official_iso_noMet", "muon")],
    enable_plot=True,
    el_cr={"ttbar": "ttbarCR_Ele", "zjets": "ZjetsCR_Ele"},
    mu_cr={"ttbar": "ttbarCR_Mu", "zjets": "ZjetsCR_Mu"},
    systematic=None,
    save=True,
    use_mp=True,
    save_iteration=False,
    correlation_correction=[],
    show_nominal_only=True,
    data_name="data",
    legend_opt="",
    ext_phsp_corr=None,
    include_signal=True,
    force_refill=False,
):
    """
    Derive phase-space correction with iterative method. Currently only ttbar,
    zjets, and provided signal process (wjets) will be considered. The signal
    CR for signal process is automatically retrieved from region-A of the ABCD
    method.

    Args:
    config : ConfigMgr/str
        config instance for deriving correction factors.

    signal : str, optional
        name of the signal process.

    skip : list(str), optional
        list of processes to skip

    bkgd : list(str), optional
        list of background processes with respect to the signal process.

    iteration : int, optional
        number of iteration to run for deriving the correction factor.

    do_pre_iterative: boolean, optional
        do pre-iterative fakes calculation. basically the usual fakes routine.

    prefix : str, optional
        prefix for output related names. e.g. prefix_signal

    tf_param : dict, optional
        tranfer factor lookup. follows the convention in the fakes calculation.

    electron_type : list(tuple(str,str,str)):
        list of correction storing keys. The format follows
        (base region, abcd tag, correction type)

    """

    config = ConfigMgr.open(config)

    prefix_signal = '_'.join([prefix, signal]).lstrip("_")
    if systematic:
        prefix_signal += f"_{'_'.join(systematic)}"
    prefix_signal = prefix_signal.replace("/", "_div_")

    # check transfer factor parametrization
    # if tf_param is not provided, default will be used.
    if tf_param is None:
        tf_param = {
            "*muon*": (("abs(lep1Eta)", "lep1Pt"), "tf_eta_vs_lepPt_mu"),
            "*electron*": (("abs(lep1Eta)", "lep1Pt"), "tf_eta_vs_lepPt_el"),
        }
        logger.warning(f"no transfer factor provided. using: {tf_param}")

    # check external phasephase correction
    if ext_phsp_corr is None:
        ext_phsp_corr = {}
    config.corrections.update(ext_phsp_corr)

    # check phasespace_corr_obs within the ConfigMgr instances
    if corr_obs not in config.phasespace_corr_obs:
        config.phasespace_corr_obs.append(corr_obs)

    # check if signal is in bkgd
    if signal in bkgd:
        include_signal = False

    # do usual fake estimation before the iterative methods
    if do_pre_iterative and include_signal:
        fake = run_abcd_fakes_estimation(
            config,
            tf_param,
            skip_process=skip,
            use_mp=use_mp,
            systematic=systematic,
            correlation_correction=correlation_correction,
            prune=False,  # do not clear BCD regions after fakes.
        )
    else:
        fake = config

    # plotting the pre-iterative fakes
    if enable_plot:
        if show_nominal_only and systematic is not None:
            pass
        else:
            run_PlotMaker.run_stack(
                fake,
                f"{output_path}/{prefix_signal}_pre_iterate",
                data=data_name,
                mcs=[signal] + bkgd + ["fakes"],
                low_yrange=(0.5, 1.7),
                logy=True,
                workers=16 if use_mp else None,
                low_ytitle="Data/Pred",
                systematic=systematic,
                legend_opt=legend_opt,
            )

    refill_bkgd = []
    for key in el_cr:
        refill_bkgd.append(key)
    # create a Correction instance for phase-space correction
    phasespace_correcion = Correction()

    # electron channel
    # get the names of the less dominated prompt backgrounds
    # e.g. singletop, diboson
    small_bkgd = set(bkgd) - el_cr.keys()
    small_bkgd = list(small_bkgd)
    for el_corr in electron_type:
        # using the abcd helper function to get the signal region name
        base_r, abcd_tag, corr_type = el_corr
        region_a = abcd.abcd_signal_region_name(fake, base_r, abcd_tag)
        _cr = []
        _cr_process = []
        for key in el_cr:
            _cr.append(el_cr[key])
            _cr_process.append(key)

        cr_regions = _cr + [region_a] if include_signal else _cr
        cr_processes = _cr_process + [signal] if include_signal else _cr_process
        phasespace_correcion.add_correction(corr_type, cr_regions, cr_processes)

        # setting backgrounds for signal region. e.g w+jets
        if include_signal:
            phasespace_correcion.set_background(region_a, small_bkgd + ["fakes"])
        # setting backgrounds for control region. e.g. ttbar, z+jets
        for _region in _cr:
            phasespace_correcion.set_background(_region, small_bkgd)

    # muon channel
    small_bkgd = set(bkgd) - mu_cr.keys()
    small_bkgd = list(small_bkgd)
    for mu_corr in muon_type:
        # using the abcd helper function to get the signal region name
        base_r, abcd_tag, corr_type = mu_corr
        region_a = abcd.abcd_signal_region_name(fake, base_r, abcd_tag)
        _cr = []
        _cr_process = []
        for key in mu_cr:
            _cr.append(mu_cr[key])
            _cr_process.append(key)

        cr_regions = _cr + [region_a] if include_signal else _cr
        cr_processes = _cr_process + [signal] if include_signal else _cr_process
        phasespace_correcion.add_correction(corr_type, cr_regions, cr_processes)

        # setting backgrounds for signal region. e.g w+jets
        if include_signal:
            phasespace_correcion.set_background(region_a, small_bkgd + ["fakes"])
        # setting backgrounds for control region. e.g. ttbar, z+jets
        for _region in _cr:
            phasespace_correcion.set_background(_region, small_bkgd)

    corr_keys = []
    for corr_key in electron_type + muon_type:
        _, _, type = corr_key
        if include_signal:
            corr_keys.append((type, signal, corr_obs, systematic))
        for _process in set(list(el_cr.keys()) + list(mu_cr.keys())):
            corr_keys.append((type, _process, corr_obs, systematic))

    # begin of the iteration correction process
    output_corr = None
    for i in range(1, iteration + 1):
        iter_fakes = fake.copy()
        corr = phasespace_correcion.derive_phasespace_correction(
            iter_fakes,
            histograms=[corr_obs],
            verbose=True,
            systematic=systematic,
            data_name=data_name,
        )
        if output_corr is None:
            output_corr = corr
        for key in corr_keys:
            if i != 1:
                if key in output_corr:
                    output_corr[key].mul(corr[key])
                else:
                    output_corr[key] = corr[key].copy()
            if key not in iter_fakes.corrections:
                iter_fakes.corrections[key] = output_corr[key].copy()
            else:
                iter_fakes.corrections[key].mul(corr[key])

        if iteration == 1 and not enable_plot and not force_refill:
            logger.info(f"early termination with {iteration=} & {enable_plot=}")
            new_fake = iter_fakes
            break

        refill_list = [signal] + refill_bkgd if include_signal else refill_bkgd
        logger.info(f"Prepare to refill {refill_list}")
        # NOTE!!!!! The refill_process() will clear internal correction buffer!
        # Need to redo the correction checking again!!
        refill_process(iter_fakes, refill_list, use_mp=use_mp, systematic=systematic)
        for key in corr_keys:
            if key not in iter_fakes.corrections:
                iter_fakes.corrections[key] = output_corr[key].copy()

        if include_signal:
            new_fake = run_abcd_fakes_estimation(
                iter_fakes,
                tf_param,
                skip_process=skip + ["fakes"],
                use_mp=use_mp,
                systematic=systematic,
                correlation_correction=correlation_correction,
                prune=False,  # do not clear BCD regions after fakes.
            )
        else:
            new_fake = iter_fakes

        # do stack plotting
        if enable_plot and save_iteration:
            if show_nominal_only and systematic is not None:
                pass
            else:
                run_PlotMaker.run_stack(
                    new_fake,
                    f"{output_path}/{prefix_signal}_iter{i}",
                    data=data_name,
                    mcs=[signal] + bkgd + ["fakes"],
                    low_yrange=(0.5, 1.7),
                    logy=True,
                    workers=16 if use_mp else None,
                    low_ytitle="Data/Pred",
                    systematic=systematic,
                )

            new_fake.save(f"{output_path}/{prefix_signal}_fake_iter{i}.pkl")

        fake = new_fake

        logger.info(f"finished iteration {i}.")

    # make stack plot for the final iteration
    if enable_plot:
        if show_nominal_only and systematic is not None:
            pass
        else:
            run_PlotMaker.run_stack(
                new_fake,
                f"{output_path}/{prefix_signal}_final",
                data=data_name,
                mcs=[signal] + bkgd + ["fakes"],
                low_yrange=(0.5, 1.7),
                logy=True,
                workers=16 if use_mp else None,
                low_ytitle="Data/Pred",
                systematic=systematic,
                legend_opt=legend_opt,
            )

    bkgd_corr = {}
    signal_corr = {}
    for key in corr_keys:
        _, process, _, _ = key
        if process in bkgd:
            bkgd_corr[key] = output_corr[key]
        elif include_signal:
            signal_corr[key] = output_corr[key]

    # saving final correction factors.
    if save:
        m_serial = Serialization()
        save_path = f"{output_path}/internal_{prefix_signal}"
        m_serial.to_shelve(output_corr, f"{save_path}_all_correction.shelf")
        m_serial.to_shelve(bkgd_corr, f"{save_path}_bkgd_correction.shelf")
        m_serial.to_shelve(signal_corr, f"{save_path}_correction.shelf")

    return signal_corr, bkgd_corr


# ==============================================================================
def compute_fakes_closure_ratio(
    fakes_pset,
    config,
    tf_params,
    closure_process="dijets",
    *,
    name="nonclosure.pkl",
    systematic=None,
    enable_plot=True,
):
    """
    Compute the ABCD closure ratio and treat the ratio as uncertainty.

    Args:
        name : str/tuple(str)
            lookup name for storing the ProcessSet.computed_systematics. Use for
            retrieving the ratio uncertainty.

        pset_a : str
            name of the process set A.

        pset_b : str
            name of the process set B.

        systematic : str, optional
            name of the systematic. use systematic full name.

    """

    config = ConfigMgr.open(config)

    # computing the closure process
    closure_tf = run_ABCD_TF(config, closure_process, systematic)
    closure = run_ABCD_Fakes_EventLevel(
        config,
        closure_tf,
        closure_process,
        tf_params,
        use_mp=True,
        full_run=False,
        match_tf_process=True,
        systematic=systematic,
    )

    process_a = closure.get_process_set(closure_process).get(systematic)
    process_b = config.get_process_set(closure_process).get(systematic)
    process_ratio = HistManipulate.Divide_Process(process_a, process_b, process_a.name)

    if enable_plot:
        closure.get_process_set(closure_process).name += "(closure)"
        closure.append_process(process_b)
        run_PlotMaker.run_stack(
            closure,
            f"{closure_process}_closure",
            data=closure_process,
            mcs=[f"{closure_process}(closure)"],
            low_yrange=(0.5, 1.7),
            logy=True,
            workers=16,
            low_ytitle="Fakes/MC",
        )

    process_fakes = fakes_pset.get(systematic)
    diff_ratio = 1 - process_ratio
    for r in diff_ratio.regions:
        for obs in r.histograms:
            obs.bin_content = np.abs(obs.bin_content)

    scale_up = process_fakes * (1 + diff_ratio)
    scale_down = process_fakes * (1 - diff_ratio)

    m_serial = Serialization()
    m_serial.to_pickle({"up": scale_up, "down": scale_down}, name)


# ==============================================================================
def sum_process_sets(process_sets, title=None):
    """
    Summing process sets. This is different from the ProcessSet.__add__, since
    ProcessSet.__add__ has a more strict requirement on matching the systematic.
    for example, two systematics differ by 'source' will not be add/merge in
    ProcessSet.__add__. In such case, nominal will be used. Also, disjoin regions
    will not be included.
    """
    syst_list = [None]
    for pset in process_sets:
        syst_list += list(pset.list_systematic_full_name())
    syst_list = list(set(syst_list))

    names = [pset.name for pset in process_sets]
    new_name = f"sum({','.join(names)})"

    pset_copy = process_sets[0].copy(shallow=True)
    pset_copy.reset()
    pset_copy.name = title if title else new_name
    for syst in syst_list:
        logger.info(f"Summing with {syst}.")
        sum_process = None
        for other_pset in process_sets:
            if sum_process is None:
                sum_process = other_pset.get(syst).copy()
            else:
                other_process = other_pset.get(syst)
                sum_process.add(other_process)
                if sum_process.systematics is None:
                    sum_process.systematics = other_process.systematics
        if sum_process:
            if syst is not None and sum_process.systematics is None:
                logger.warning(f"getting nominal for {syst}?")
            sum_process.name = title if title else new_name
            pset_copy.add(sum_process)

    return pset_copy


# ==============================================================================
def inclusive_repr(hist, inplace=False):
    """
    Change 1D histogram to inclusive representation.
    """
    c_hist = hist if inplace else hist.copy()
    old_content = c_hist.bin_content.copy()
    old_sumW2 = c_hist.sumW2.copy()
    for i in range(len(c_hist.bin_content)):
        c_hist.bin_content[i] = old_content[i:].sum()
        c_hist.sumW2[i] = old_sumW2[i:].sum()

        if c_hist.systematics_band is None:
            continue

        for band in c_hist.systematics_band.values():
            old_sum = old_content[i:].sum() or 1.0
            old_i = old_content[i:]
            components = band.components["up"].keys()
            up_compts = band.components["up"]
            for name in components:
                up_compts[name][i] = (up_compts[name][i:] * old_i).sum() / old_sum
            components = band.components["down"].keys()
            dn_compts = band.components["down"]
            for name in components:
                dn_compts[name][i] = (dn_compts[name][i:] * old_i).sum() / old_sum

    return c_hist


# ==============================================================================
def reduce_binning(hist, inplace=False):
    """
    reduce binning of the histogram
    TESTING!
    """
    c_hist = hist if inplace else hist.copy()
    reduced_bins = hist.bins[::2]
    # underflow = hist.bin_content[0]
    # overflow = hist.bin_content[-1]
    new_content = np.zeros(reduced_bins.shape)
    new_content[[0, -1]] = hist.bin_content[[0, -1]]
    # breakpoint()
    new_content[1:-1] += hist.bin_content[1:-1][0::2]
    new_content[1:-1] += hist.bin_content[1:-1][1::2]
    c_hist.bins = reduced_bins
    c_hist.bin_content = new_content

    return c_hist


# ==============================================================================
def rbin_merge(hist, inplace=False):
    """
    reduce binning of the histogram
    TESTING!
    """
    c_hist = hist if inplace else hist.copy()
    reduced_bins = hist.bins[:-1]
    old_content = hist.bin_content[:-1]
    old_sumW2 = hist.sumW2[:-1]
    old_content[-1] += hist.bin_content[-1]
    old_sumW2[-1] += hist.sumW2[-1]

    if c_hist.systematics_band:

        for band in c_hist.systematics_band.values():
            for name, val in band.components["up"].items():
                overflow = (
                    val[-1] * hist.bin_content[-1] + val[-2] * hist.bin_content[-2]
                )
                overflow /= hist.bin_content[-1] + hist.bin_content[-2]
                old_val = val[:-1]
                old_val[-1] = overflow
                band.components["up"][name] = old_val
                band.components["up"][name].shape = old_val.shape
                band.shape = old_content.shape
            for name, val in band.components["down"].items():
                overflow = (
                    val[-1] * hist.bin_content[-1] + val[-2] * hist.bin_content[-2]
                )
                overflow /= hist.bin_content[-1] + hist.bin_content[-2]
                old_val = val[:-1]
                old_val[-1] = overflow
                band.components["down"][name] = old_val
                band.components["down"][name].shape = old_val.shape
                band.shape = old_content.shape

    c_hist.bins = reduced_bins
    c_hist.bin_content = old_content
    c_hist.sumW2 = old_sumW2

    return c_hist


# ==============================================================================
def _weighted_rolling_average1d(
    content, weight=None, window=3, inplace=True, renormalize=True, update_weight=False
):
    nbin = len(content)
    if window > nbin:
        raise ValueError("window size cannot be larger than bin size")

    content_copy = content if inplace else content.copy()
    if weight is None:
        nom_weight = np.ones(nbin)
        nom_weight /= nom_weight.sum()
    else:
        nom_weight = weight / weight.sum()

    lhalf_w = int(window / 2)
    rhalf_w = window - lhalf_w
    lbound = lhalf_w
    rbound = nbin - rhalf_w

    for i in range(lbound):
        s_index = slice(None, window)
        if renormalize:
            _weight = nom_weight[s_index] / nom_weight[s_index].sum()
        else:
            _weight = nom_weight[s_index]
        top = (content_copy[s_index] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[s_index].sum()

    for i in range(lbound, rbound):
        s_index = slice(i - lhalf_w, i + rhalf_w)
        if renormalize:
            _weight = nom_weight[s_index] / nom_weight[s_index].sum()
        else:
            _weight = nom_weight[s_index]
        top = (content_copy[s_index] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[s_index].sum()

    for i in range(rbound, nbin):
        s_index = slice(nbin - rhalf_w, None)
        if renormalize:
            _weight = nom_weight[s_index] / nom_weight[s_index].sum()
        else:
            _weight = nom_weight[s_index]
        top = (content_copy[s_index] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[s_index].sum()

    return content_copy


# ==============================================================================
def _weighted_rolling_average1d_oneside(
    content, weight=None, window=3, inplace=True, renormalize=True, update_weight=False
):
    nbin = len(content)
    if window > nbin:
        raise ValueError("window size cannot be larger than bin size")

    content_copy = content if inplace else content.copy()
    if weight is None:
        nom_weight = np.ones(nbin)
        nom_weight /= nom_weight.sum()
    else:
        nom_weight = weight / weight.sum()

    rbound = nbin - window

    for i in range(rbound):
        s_index = slice(i, window)
        if renormalize:
            _weight = nom_weight[s_index] / nom_weight[s_index].sum()
        else:
            _weight = nom_weight[s_index]
        top = (content_copy[s_index] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[s_index].sum()

    for i in range(rbound, nbin):
        s_index = slice(i, nbin)
        if renormalize:
            _weight = nom_weight[s_index] / nom_weight[s_index].sum()
        else:
            _weight = nom_weight[s_index]
        top = (content_copy[s_index] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[s_index].sum()

    return content_copy


# ==============================================================================
def _weighted_rolling_average1d_varbin(
    content, windows, weight=None, inplace=True, renormalize=True, update_weight=False
):
    nbin = len(content)

    content_copy = content if inplace else content.copy()
    if weight is None:
        nom_weight = np.ones(nbin)
        nom_weight /= nom_weight.sum()
    else:
        nom_weight = weight / weight.sum()

    for i, window in enumerate(windows):
        if renormalize:
            _weight = nom_weight[window] / nom_weight[window].sum()
        else:
            _weight = nom_weight[window]
        top = (content_copy[window] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[window].sum()

    for i, window in enumerate(reversed(windows), start=-nbin + 1):
        if renormalize:
            _weight = nom_weight[window] / nom_weight[window].sum()
        else:
            _weight = nom_weight[window]
        top = (content_copy[window] * _weight).sum()
        bot = _weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)
        if update_weight:
            nom_weight[i] = _weight[window].sum()

    return content_copy


# ==============================================================================
def _weighted_rolling_average1d_varbin2(
    content, windows, weights, inplace=True, renormalize=True, update_weight=False
):
    nbin = len(content)

    content_copy = content if inplace else content.copy()
    for i, (window, weight) in enumerate(zip(windows, weights)):
        top = (content_copy[window] * weight).sum()
        bot = weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)

    for i, (window, weight) in enumerate(
        zip(reversed(windows), reversed(weights)), start=-nbin + 1
    ):
        top = (content_copy[window] * weight).sum()
        bot = weight.sum()
        content_copy[i] = np.nan_to_num(top / bot)

    return content_copy


# ==============================================================================
def systematic_smoothing(
    hist,
    smooth_type,
    smoother=None,
    inplace=False,
    smooth_syst=True,
    smooth_stats=False,
    smooth_content=False,
    overflow=True,
    do_reverse=True,
    filter_band=None,
    **kwargs,
):
    c_hist = hist if inplace else hist.copy()

    from functools import partial

    if smoother is None:
        if smooth_type == "savgol_filter":
            from scipy.signal import savgol_filter

            # yhat = savgol_filter(y, 51, 3) # window size, polynomial order
            smoother = partial(savgol_filter, **kwargs)
        elif smooth_type == "gaussian_filter":
            from scipy.ndimage import gaussian_filter1d

            smoother = partial(gaussian_filter1d, **kwargs)
        elif smooth_type == "weighted_average_1d":
            smoother = partial(_weighted_rolling_average1d, **kwargs)
        elif smooth_type == "weighted_average_1d_oneside":
            smoother = partial(_weighted_rolling_average1d_oneside, **kwargs)
        elif smooth_type == "weighted_average_1d_varbin":
            smoother = partial(_weighted_rolling_average1d_varbin, **kwargs)
        elif smooth_type == "weighted_average_1d_varbin2":
            smoother = partial(_weighted_rolling_average1d_varbin2, **kwargs)
        else:
            raise ValueError(f"Unrecognized smooth type {smooth_type}")

    start = None if overflow else 1
    end = None if overflow else -1

    # smoothing stats
    if smooth_stats:
        c_hist._sumW2[start:end] = smoother(c_hist.sumW2[start:end])
        if do_reverse:
            c_hist._sumW2[start:end] = np.flip(
                smoother(np.flip(c_hist.sumW2[start:end]))
            )

    if smooth_content:
        c_hist._bin_content[start:end] = smoother(c_hist.bin_content[start:end])
        if do_reverse:
            c_hist._bin_content[start:end] = np.flip(
                smoother(np.flip(c_hist.bin_content[start:end]))
            )

    if not smooth_syst:
        return c_hist

    if c_hist.systematics_band is None:
        raise RuntimeError("Cannot find systematic band")

    if filter_band is None:
        filter_band = set()

    for bname, band in c_hist.systematics_band.items():
        if bname in filter_band:
            continue
        components = band.components["up"].keys()
        up_compts = band.components["up"]
        dn_compts = band.components["down"]
        for name in components:
            smoothed_up = smoother(up_compts[name][start:end])
            smoothed_dn = smoother(dn_compts[name][start:end])
            if do_reverse:
                smoothed_up = np.flip(smoother(np.flip(smoothed_up)))
                smoothed_dn = np.flip(smoother(np.flip(smoothed_dn)))
            up_compts[name][start:end] = smoothed_up
            dn_compts[name][start:end] = smoothed_dn

    return c_hist
