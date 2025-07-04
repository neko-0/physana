import click
import logging
from .lazy_import import lazy_import as lazy

log = logging.getLogger(__name__)
run_HistMaker = lazy("physana.algorithm.interface")
routines_hmaker = lazy("physana.routines.run_histmaker_json")


@click.group(name='histmaker')
def cli():
    """entry point for HistMaker"""


@cli.command(name='run')
@click.option(
    "--config", type=str, help="path to the JSON configuration for HistMaker."
)
@click.option("--combine/--no-combine", default=False)
@click.option("--verbose/--no-verbose", default=False, help="setting logging level.")
@click.option("--local/--no-local", default=False)
@click.option("--nworkers", type=int, default=None, help="number of workers.")
def run_histmaker_json(config, combine, verbose, local, nworkers):
    """
    Run HistMaker from a JSON configuration file.
    """

    if combine:
        histmaker_json = routines_hmaker.combine_json_setups(config)
    else:
        config = config.split(',')
        if len(config) > 1:
            histmaker_json = routines_hmaker.combine_json_setups(config)
        else:
            histmaker_json = routines_hmaker.JSONHistSetup(config[0])

    # initalize parse all the blocks in the JSON
    histmaker_json.initialize()

    if verbose:
        routines_hmaker.set_verbose()
    if local:
        histmaker_json.others["local"] = True
    if nworkers:
        histmaker_json.others["max_num_jobs"] = nworkers

    histmaker_json.launch()


@cli.command(name='fill')
@click.option("--file", type=str, help="input config name.")
@click.option("--output", type=str, help="output config name.")
@click.option("--forcefill/--no-forcefill", default=False)
@click.option("--start", type=int, default=None, help="entry start.")
@click.option("--stop", type=int, default=None, help="entry stop.")
def histmaker_fill(file, output, forcefill, start, stop):
    """
    Command line interface to fill a ConfigMgr object.
    Currently only the default HistMaker can be used.
    """
    run_HistMaker.run_algorithm(
        file, algorithm=None, forcefill=forcefill, entry_start=start, entry_stop=stop
    ).save(output)


@cli.command(name='run-syst')
@click.option("--target", type=str, help="folder for prepared systematic run.")
@click.option("--split", type=str, default=None, help="split type.")
@click.option("--nworkers", type=int, default=10, help="number of workers.")
@click.option("--start", type=int, default=None, help="start of file index.")
@click.option("--end", type=int, default=None, help="end of file index.")
def histmaker_run_prepared_systematic(target, split, nworkers, start, end):
    """
    Command line interface to fill a ConfigMgr object.
    Currently only the default HistMaker can be used.
    """
    run_HistMaker.run_prepared_systematic(
        "splitted_tracker",
        output=target,
        nworkers=nworkers,
        split_type=split,
        mp_context="fork",
        n_workers=nworkers,
        start_findex=start,
        end_findex=end,
    )


'''
@cli.command()
@click.option("--config", type=str, help="path to the configuration file (.py file).")
@click.option("--oname", type=str, help="output file name.")
@click.option("--weight-obs", type=str)
@click.option("--weight-file", type=str)
@click.option("--scale_factor", type=float, default=1.0)
def weight_gen(config, oname, weight_obs, weight_file, scale_factor):
    """
    Given a config, reduce ntuples to histograms with addition of external weight distribution
    """
    start = time.time()
    run_HistMaker_weight_gen(config, oname, weight_obs, weight_file, scale_factor)
    log.info(f"cost {time.time() - start}")
'''
