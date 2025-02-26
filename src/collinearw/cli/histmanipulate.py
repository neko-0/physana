import click
import time
import logging
from .lazy_import import lazy_import as lazy

log = logging.getLogger(__name__)
histManipulate = lazy("collinearw.histManipulate")


def time_it(func, args):
    start = time.time()
    func(*args)
    log.info(f"cost {time.time() - start}")


@click.group(name='histmanipulate')
def cli():
    """entry point for histmanipulate"""


@cli.command()
@click.option("--config", type=str, help="path to ConfigMgr file (.pkl file).")
@click.option("--process", type=str, help="signal process")
@click.option("--oname", type=str, default="", help="output file name")
def abcd_tf(config, process, oname):
    """
    ABCD transfer factor.
    """
    time_it(histManipulate.run_ABCD_TF, args=(config, process, oname))


@cli.command()
@click.option("--config", type=str, help="path to ConfigMgr file (.pkl file).")
@click.option("--const-tf/--no-const-tf", default=False, help="using constant TF")
@click.option("--oname", type=str, default="", help="output file name")
@click.option("-v", type=str, default="critical", help="verbosity level")
def abcd_fake(config, const_tf, oname, v):
    """
    Fake background estimation using result from ABCD.
    """
    logging.getLogger("collinearw.histManipulate").setLevel(getattr(logging, v.upper()))
    logging.getLogger("collinearw.core").setLevel(getattr(logging, v.upper()))
    logging.getLogger("collinearw.configMgr").setLevel(getattr(logging, v.upper()))
    time_it(histManipulate.run_ABCD_Fakes, args=(config, const_tf, oname))
