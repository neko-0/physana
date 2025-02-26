import pickle
import klepto
import json
import shelve
import concurrent.futures
import multiprocessing
import asyncio
import logging
from pathlib import Path

from .base import SerializationBase
from .mixin import SerialProcessSet
from .base import _pickle_save
from .base import async_from_pickles

sem = asyncio.Semaphore(10)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class SerialConfig(SerializationBase):
    def to_klepto(self, config, name, archive_type="dir_archive", *args, **kwargs):
        kwargs.setdefault("protocol", pickle.HIGHEST_PROTOCOL)
        config_generator = config.self_split("process", copy=False)
        archive_method = getattr(klepto.archives, archive_type)
        data = {}
        for i, config in enumerate(config_generator):
            data[f"id{i}"] = config
        cache = archive_method(name, data, *args, **kwargs)
        cache.dump()

    def to_shelve(self, config, name, *args, **kwargs):
        name = Path(name)
        name.parent.mkdir(parents=True, exist_ok=True)
        name = str(name.resolve())
        with shelve.open(name, flag="n", *args, **kwargs) as m_shelf:
            for id, process in enumerate(config.self_split("process", copy=False)):
                m_shelf[f"id{id}"] = process

    def from_shelve(self, name, *args, **kwargs):
        output = []
        with shelve.open(name, *args, **kwargs) as m_shelf:
            for id in m_shelf.keys():
                output.append(m_shelf[id])
        return output

    def to_dir(self, config, name, nworkers=8):
        split_type = "process"
        name = Path(name).resolve()
        config_generator = config.self_split(split_type, copy=False)
        file_map = {}
        nfiles = 0

        meta_data_path = Path(f"{name}/metadata").resolve()
        meta_data = {}
        meta_data["header"] = {
            "source": f"{config.out_path}/{config.ofilename}",
            "n_process_sets": len(config.process_sets),
            "process_name": config.list_processes(),
            "split_type": split_type,
            "nfiles": nfiles,
            "backend": "pickle",
            "metadata_path": str(meta_data_path),
        }
        meta_data["content"] = file_map
        with open(f"{meta_data_path}.json", "w") as fp:
            json.dump(meta_data, fp)

        futures = []
        with concurrent.futures.ProcessPoolExecutor(
            nworkers, mp_context=multiprocessing.get_context('forkserver')
        ) as exe:
            for i, m_config in enumerate(config_generator):
                config_name = Path(f"{name}/id{i}.pkl")
                future = exe.submit(_pickle_save, m_config, config_name)
                futures.append(future)
                nfiles += 1
                meta_data["header"]["nfiles"] = nfiles
                file_map[f"{config_name.resolve()}"] = m_config.name
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                config_name = future.result()
                ofilename = f"{config_name.resolve()}"
                name = file_map.pop(ofilename)
                file_map[f"{name}"] = ofilename
                if i % 100 == 0:
                    print(f"saved {i}/{nfiles}")
                    with open(f"{meta_data_path}.json", "w") as fp:
                        json.dump(meta_data, fp)

        with open(f"{meta_data_path}.json", "w") as fp:
            json.dump(meta_data, fp)

    def from_pickles(self, files, *args, **kwargs):
        config = None
        configs = async_from_pickles(files, *args, **kwargs)
        for _config in configs:
            if config is None:
                config = _config
            else:
                config.add(_config)
        return config

    def to_dict(self, config):
        pset_s = SerialProcessSet().to_dict
        config_dict = {}
        config_dict["metadata"] = {
            "ofilename": config.ofilename,
            "src_path": config.src_path,
            "out_path": config.out_path,
        }
        config_dict["process_sets"] = {x.name: pset_s(x) for x in config.process_sets}

        return config_dict
