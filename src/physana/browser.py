from rich.tree import Tree
from rich import print as richprint


def print_config(config):
    config_tree = Tree(str(config.get_output_file()))
    for process in config.process_sets:
        config_tree.add(process.name)
    richprint(config_tree)
