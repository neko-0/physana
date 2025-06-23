import click
import os
import concurrent.futures
import logging
from glob import glob
from pathlib import Path
from tqdm import tqdm
from .lazy_import import lazy_import as lazy

configMgr = lazy("physana.configs.base")
merge_tools = lazy("physana.configs.merge_tools")
unfolding = lazy("physana.strategies.unfolding")
tools = lazy("physana.tools")

logger = logging.getLogger(__name__)


@click.group(name='utility')
def cli():
    """entry point for general utilities"""


@cli.command()
@click.argument("config-manager", type=str)
def browse(config_manager):
    ConfigMgr = configMgr.ConfigMgr
    mgr = ConfigMgr.open(config_manager)
    mgr.print_config()


@cli.command()
@click.argument("config-manager", type=str)
@click.option("--out_path", type=str, help="new output directory")
def set(config_manager, out_path):
    ConfigMgr = configMgr.ConfigMgr
    mgr = ConfigMgr.open(config_manager)
    if out_path:
        mgr.out_path = out_path
    mgr.save()


@cli.command()
@click.option("--file", type=str, help="path to ConfigMgr file")
@click.option("--type", type=str, help="new output directory")
def filter_region(file, type):
    ConfigMgr = configMgr.ConfigMgr
    ifile = ConfigMgr.open(file)
    ifile.filter_region(type)
    ifile.save(f"{ifile.ofilename}_filterd")


@cli.command()
@click.option("--config", type=str, help="path to ConfigMgr file")
@click.option("--process", type=str, default="", help="name of process")
@click.option("--regions", type=str, default="", help="name of regions")
@click.option("--all/--no-all", default=False, help="process all contents.")
def nevents(config, all, process, regions):
    ConfigMgr = configMgr.ConfigMgr
    ifile = ConfigMgr.open(config)
    if all:
        logger.info("processing all.")
        import json

        output = {}
        for p in ifile.processes:
            output[p.name] = {}
            output[p.name]["selection"] = p.selection
            output[p.name]["regions"] = {}
            for r in p.regions:
                output[p.name]["regions"][r.name] = {
                    "selection": r.selection,
                    "events": r.effective_event_count,
                    "total": r.total_event_count,
                }
        with open("{}/all_nevent.json".format(ifile.out_path), "w") as f:
            json.dump(output, f)
    else:
        if regions:
            with open("{}/{}_nevent.log".format(ifile.out_path, process), "w") as f:
                for r in regions.replace(" ", "").split(","):
                    my_r = ifile.get_process(process).get_region(r)
                    f.write("{} : {}\n".format(my_r.name, my_r.effective_event_count))
        else:
            import json

            output = {}
            for r in ifile.get_process(process).regions:
                output[r.name] = {
                    "selection": r.selection,
                    "events": r.effective_event_count,
                }
            with open("{}/{}_nevent.json".format(ifile.out_path, process), "w") as f:
                json.dump(output, f)


def batch_merge(configs, output_name, n=5):
    """
    Merge multiple config objects in parallel batches.
    """
    tool_merge = merge_tools.merge
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(tool_merge, configs[i * n : (i + 1) * n])
            for i in range((len(configs) + n - 1) // n)
        ]
        results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
        merged_config = tool_merge(results)

        return merged_config.save(output_name)


@cli.command()
@click.option("--input", type=str, help="Input ConfigMgr files")
@click.option("--output", type=str, help="Output name")
def merge(input, output):
    config_open = configMgr.ConfigMgr.open
    tool_merge = merge_tools.merge

    input_configs = []
    for x in input.replace(" ", "").split(","):
        input_configs += glob(x)

    print("list of files: \n" + "\n".join(input_configs))

    tool_merge((config_open(x) for x in input_configs)).save(output)


@cli.command()
@click.option("--output", type=str, help="output name")
@click.option("--input", type=str, help="input config files")
@click.option(
    "--loadall/--no-loadall", default=False, help="load all configs first before merge"
)
def intersection(output, input, loadall):
    """
    Merging ConfigMgr objects without duplication and adding of contents.
    """
    ConfigMgr = configMgr.ConfigMgr
    tool_intersection = merge_tools.intersection
    input_configs = []
    for x in input.replace(" ", "").split(","):
        _configs = glob(x)
        input_configs += _configs
        print("Adding files \n" + "\n".join(_configs))
    print(f"Merging total {len(input_configs)} config objects to {output}")
    if loadall:
        configs = (x for x in ConfigMgr.open_files(input_configs))
    else:
        config_open = ConfigMgr.open
        configs = (config_open(f) for f in input_configs)
    tool_intersection(configs, copy=False).save(output)


@cli.command()
@click.option("--output", type=str, help="output name")
@click.option("--nominal", type=str, help="nominal config")
@click.option("--migratein", type=str, help="migration IN config")
@click.option("--migrateout", type=str, default=None, help="migration OUT config")
def merge_migration(output, nominal, migratein, migrateout):
    """
    Merging ConfigMgr with phase-space migration.
    """
    ConfigMgr = configMgr.ConfigMgr
    nom = ConfigMgr.open(nominal)
    migration_in = ConfigMgr.open(migratein)
    migrate_out = ConfigMgr.open(migrateout) if migrateout else None
    merged = unfolding.utils.merge_migration(nom, migration_in, migrate_out, True)
    merged.set_output_location(".")
    merged.save(output)


@cli.command()
@click.option("--input", type=str, help="input config name")
@click.option("--output", type=str, help="output config name.")
@click.option(
    "--include", type=str, default=None, help="process removal list (comma separated)"
)
@click.option(
    "--exclude", type=str, default=None, help="process inclusion list (comma separated)"
)
def filter_process(input, output, include, exclude):
    """
    Removing processes from ConfigMgr object
    """
    ConfigMgr = configMgr.ConfigMgr
    include = [] if include is None else include.split(",")
    exclude = [] if exclude is None else exclude.split(",")
    i_config = ConfigMgr.open(input)
    for pset in i_config.list_processes():
        if exclude and pset in exclude:
            i_config.remove_process_set(pset)
        elif include and pset not in include:
            i_config.remove_process_set(pset)
    i_config.set_output_location(".")
    i_config.save(output)


@cli.command()
@click.option("--input", type=str, help="input config name")
@click.option("--output", type=str, help="output config name")
def extract_nominal(input, output):
    """
    Extract nominal processes from the given ConfigMgr object
    """
    ConfigMgr = configMgr.ConfigMgr
    config = ConfigMgr.open(input)
    splitted = config.self_split("systematic")
    nominal = next(splitted)
    nominal.set_output_location(".")
    nominal.save(output)


@cli.command()
@click.option("--output", type=str, help="output config name")
@click.option("--input", type=str, help="input config name")
@click.option("--nfiles", type=int, default=0, help="number of input config used")
def merge_replica(output, input, nfiles):
    """
    Mergeing ConfigMgr outputs from the Bootstrap.
    """
    ConfigMgr = configMgr.ConfigMgr
    input_configs = []
    for x in input.split(","):
        _configs = glob(x)
        if nfiles:
            _configs = _configs[:nfiles]
        input_configs += _configs
        for y in _configs:
            print(f"adding {y}")

    first_config = ConfigMgr.open(input_configs.pop(0))
    hist_list = []
    for process in first_config.processes:
        for region in process.regions:
            for hist in region.histograms:
                name = f"{process.name}//nominal//{region.name}//{hist.name}"
                hist_list.append((hist, name))

    for rhs_config in map(ConfigMgr.open, input_configs):
        for hist, name in hist_list:
            hist.merge_replica(rhs_config.get(name))

    first_config.save(output)


# =============================================================================
@cli.command()
@click.option("--input", type=str, help="input config name")
@click.option("--output", type=str, help="output config name")
@click.option("--name", type=str, help="process name")
@click.option("--filter", type=str, help="process name")
def filter_region_type(input, output, name, filter):
    """
    Use for filtering region type for a given process
    """
    ConfigMgr = configMgr.ConfigMgr
    input_c = ConfigMgr.open(input)
    process_set = input_c.get(name)
    filter_types = filter.split(",")
    for p in process_set:
        p.regions = [r for r in p.regions if r.type not in filter_types]
    input_c.save(output)


# =============================================================================
@cli.command()
@click.option("--input", type=str, help="input config name")
@click.option("--output", type=str, help="output config name")
@click.option("--hists", type=str, help="common separated histogram names")
def remove_histograms(input, output, hists):
    """
    Removing histograms from the all processes/regions by names.
    """
    input_c = configMgr.ConfigMgr.open(input)
    process_set = input_c.process_sets
    hists = hists.split(",")
    regions = (r for pset in process_set for p in pset for r in p.regions)
    for r in regions:
        exist_hists = r.list_histograms()
        for hist in hists:
            if hist not in exist_hists:
                continue
            r.remove_histogram(hist)
    input_c.histograms = [h for h in input_c.histograms if h.name not in hists]
    input_c.histograms2D = [h for h in input_c.histograms2D if h.name not in hists]
    meta_obs = input_c.meta_data['unfold']['observables']
    for h in hists:
        meta_obs.pop(h, None)
    input_c.save(output)


# =============================================================================
@cli.command()
@click.option("--nominal", type=str, help="nominal input config name")
@click.option("--output", type=str, help="output config name")
@click.option("--swap", type=str, help="config for swaping")
def swap(nominal, swap, output):
    """
    replacing process from another ConfigMgr.
    Only process with same name and systematics will be swapped.
    """
    nominal = configMgr.ConfigMgr.open(nominal)
    nominal.swap_processes(configMgr.ConfigMgr.open(swap))
    nominal.save(output)


# =============================================================================
@cli.command()
@click.option("--input", type=str, help="list of input config name. comma separetd.")
@click.option("--output", type=str, help="output config name")
@click.option(
    "--nominal",
    type=str,
    default=None,
    help="first one of the input list will be used if nominal is not specified.",
)
def merge_band(input, output, nominal=None):
    """
    Merging systematic bands from other ConfigMgr objects.
    """
    config_list = []
    for x in input.split(","):
        _configs = glob(x)
        config_list += _configs
    if nominal:
        config_list = [nominal] + config_list
    config = configMgr.ConfigMgr.open(config_list[0])
    for x in config_list[1:]:
        config = configMgr.intersect_histogram_systematic_band(config, x)
    if config_list:  # check if empty input
        config.save(output)


# =============================================================================
# for SLAC batch run
# =============================================================================
@cli.command("lsf")
@click.option("--file", type=str, help="py file to be used on lsf batch.")
@click.option("--runtime", type=str, default="00:15", help="estimate run time")
@click.option("--ncore", type=int, default=5, help="number of requesting core")
@click.option("--dir", type=str, default=None, help="output directory for logs")
def slac_lsf(file, runtime, ncore, dir):
    file = Path(file)
    if not file.name:
        raise ValueError(f"Invlid file {file.resolve()}")
    file.parent.mkdir(parents=True, exist_ok=True)

    if dir:
        filename = Path(f"{dir}/{file.name}.sh")
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename = filename.resolve()
    else:
        filename = Path(f"{file}.sh").resolve()
    with open(filename, "w") as runscript:
        runscript.write("#!/bin/bash \n")
        runscript.write(f"#BSUB -W{runtime} \n")
        runscript.write(f"#BSUB -n {ncore} \n")

        runscript.write("cd ~\n")
        runscript.write("source ~/physana/envsetup.sh \n")
        runscript.write(f"python {file.resolve()}")

    cmd = "bsub "
    cmd += "-R \"centos7\" "
    cmd += f"-o {filename}.log "
    cmd += f"-e {filename}_error.log "
    cmd += f"< {filename}"
    os.system(cmd)


# =============================================================================
# Check data completeness
# =============================================================================
@cli.command()
@click.option("--files", type=str, help="list of files to be checked")
@click.option("--match-info", type=str, default=None, help="matching configurate.")
def check_data(files, match_info):
    list_of_files = glob(files)
    logger.info("Checking data completeness for: \n" + "\n".join(list_of_files))
    if match_info:
        metadata_type, lookup, nfiles, nevents = match_info.split(",")
        nfiles = int(nfiles)
        nevents = int(nevents)
        data_year = {lookup: (nfiles, nevents)}
    else:
        metadata_type = "campaign"
        data_year = None
    tools.check_data_completeness(
        tqdm(list_of_files, leave=False), metadata_type, data_year
    )
