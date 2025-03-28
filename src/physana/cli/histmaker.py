import click
import time
import logging
from .lazy_import import lazy_import as lazy

log = logging.getLogger(__name__)
run_HistMaker = lazy("physana.algorithm.interface")
configMgr = lazy("physana.configs.base")


@click.group(name='histmaker')
def cli():
    """entry point for HistMaker"""


@cli.command(name='run')
@click.option("--config", type=str, help="path to the configuration file (.py file).")
@click.option("--oname", type=str, help="output file name.")
@click.option("--batch-dir", default="./", type=str)
@click.option("--findex", multiple=True, type=str)
@click.option("--submit-batch/--no-submit-batch", default=False)
@click.option("--merge/--no-merge", default=False)
def run_histmaker(config, oname, batch_dir, findex, submit_batch, merge):
    """
    Given a config, reduce ntuples to histograms.
    """
    start = time.time()
    if merge:
        run_HistMaker.merge_output(config, oname)
    else:
        run_HistMaker.run_HistMaker(config, oname)
    log.info(f"cost {time.time() - start}")


@cli.command(name='fill')
@click.option("--input", type=str, help="input config name.")
@click.option("--output", type=str, help="output config name.")
@click.option("--forcefill/--no-forcefill", default=False)
def histmaker_fill(input, output, forcefill):
    """
    Command line interface to fill a ConfigMgr object.
    Currently only the default HistMaker can be used.
    """
    config = configMgr.ConfigMgr.open(input)
    if config.filled and not forcefill:
        log.warning(f"config {input} already filled.")
    else:
        config = run_HistMaker.run_algorithm(config, histmaker=None)
        config.save(output)


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
