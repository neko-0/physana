import click
from pathlib import Path
from .lazy_import import lazy_import as lazy

configMgr = lazy("physana.configs.base")
serialization = lazy("physana.serialization")


@click.group(name='conversion')
def cli():
    """entry point for conversion utilities"""


@cli.command()
@click.option("--config", type=str, help="Input path to ConfigMgr file.")
@click.option("--output", type=str, default=None, help="Output path to ROOT file.")
def config_to_root(config, output):
    config_open = configMgr.ConfigMgr.open
    dump_to_root = serialization.to_root.dump_config_histograms

    config = Path(config)
    output = Path(output) if output else config.with_suffix(".root")

    config = str(config.resolve())
    output = str(output.resolve())

    dump_to_root(config_open(config), output)

    print(f"Dumped ConfigMgr to ROOT file: {output}")
