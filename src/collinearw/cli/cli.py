import click
from ..version import __version__
from . import histmaker, histmanipulate, plotmaker, utility


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
def collinearw():
    """
    Improving the modeling of collinear W+jets.
    """
    pass


collinearw.add_command(histmaker.cli)
collinearw.add_command(histmanipulate.cli)
collinearw.add_command(plotmaker.cli)
collinearw.add_command(utility.cli)
