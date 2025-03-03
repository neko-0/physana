import click
from ..version import __version__
from . import histmaker, histmanipulate, plotmaker, utility


@click.group(context_settings=dict(help_option_names=['-h', '--help']))
@click.version_option(version=__version__)
def physana():
    """
    Improving the modeling of collinear W+jets.
    """
    pass


physana.add_command(histmaker.cli)
physana.add_command(histmanipulate.cli)
physana.add_command(plotmaker.cli)
physana.add_command(utility.cli)
