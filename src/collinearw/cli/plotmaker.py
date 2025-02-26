import click
import time
import logging
from .lazy_import import lazy_import as lazy

log = logging.getLogger(__name__)
run_PlotMaker = lazy("collinearw.run_PlotMaker")


def time_it(func, args):
    start = time.time()
    func(*args)
    log.info(f"cost {time.time() - start}")


@click.group(name='plotmaker')
def cli():
    """entry point for PlotMaker"""


@cli.command(name='run')
@click.option("--config", type=str, help="path to ConfigMgr file")
def run_plotmaker(config):
    """
    Make all plots
    """
    start = time.time()
    run_PlotMaker.run_PlotMaker(config)
    log.info(f"cost {time.time() - start}")


@cli.command("compare-ab")
@click.option("-config", type=str, help="path to ConfigMgr file")
@click.option("-process", type=str, help="name of process")
@click.option("-otag", type=str, default="compare_AB", help="output tag for plots")
@click.option("-text", type=str, default="", help="mode")
@click.option("-yrange", type=(float, float), default=(-0.1, 0.5), help="y range")
def plot_ABCompare(config, process, otag, text, yrange):
    time_it(
        run_PlotMaker.run_ABCD_ABCompare,
        (config, process, text, yrange, otag),
    )
