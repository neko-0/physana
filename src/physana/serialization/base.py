import pickle
import shelve
import json
from pathlib import Path


import lz4.frame
import yaml
import asyncio
import aiofiles

sem = asyncio.Semaphore(10)


def _pickle_save(data, name, *args, **kwargs):
    name = Path(name)
    name.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("protocol", pickle.HIGHEST_PROTOCOL)
    with lz4.frame.open(name, "wb") as f:
        pickle_stream = pickle.dumps(data, *args, **kwargs)
        f.write(pickle_stream)
    return name


def async_from_pickles(files, *args, **kwargs):
    pending = (async_read_file(file) for file in files)
    loop = asyncio.get_event_loop()
    for fdata in loop.run_until_complete(asyncio.gather(*pending)):
        yield pickle.loads(fdata, *args, **kwargs)


def from_pickles(files, *args, **kwargs):
    for file in files:
        with open(file, "rb") as f:
            data = f.read()
        yield pickle.loads(data, *args, **kwargs)


async def async_read_file(file):
    async with sem:
        async with aiofiles.open(file, "rb") as f:
            return await f.read()


def async_read_files(files):
    pending = [async_read_file(file) for file in files]
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(asyncio.gather(*pending))


def from_json(file, *args, **kwargs):
    with open(file, "r") as f:
        return json.load(f, *args, **kwargs)


def to_json(data, file, *args, **kwargs):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("indent", 4)
    kwargs.setdefault("separators", (",", ": "))
    with open(file, "w") as f:
        json.dump(data, f, *args, **kwargs)

    return file.resolve()


class SerializationBase:
    def to_pickle(self, data, name, *args, **kwargs):
        name = Path(name)
        name.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("protocol", pickle.HIGHEST_PROTOCOL)
        with lz4.frame.open(name, "wb") as f:
            f.write(pickle.dumps(data, *args, **kwargs))

    def from_pickle(self, name, *args, **kwargs):
        with lz4.frame.open(name) as f:
            return pickle.load(f, *args, **kwargs)

    def to_pickles(self, data, name, *args, **kwargs):
        """
        use for dumping more than one objects.
        """
        name = Path(name)
        name.parent.mkdir(parents=True, exist_ok=True)
        kwargs.setdefault("protocol", pickle.HIGHEST_PROTOCOL)
        with lz4.frame.open(name, "wb") as f:
            for datum in data:
                f.write(pickle.dumps(datum, *args, **kwargs))

    def from_pickles(self, name, *args, **kwargs):
        """
        loading more than one objects, return a generator.
        """
        kwargs.setdefault("protocol", pickle.HIGHEST_PROTOCOL)
        with lz4.frame.open(name) as f:
            while True:
                try:
                    yield pickle.load(f, *args, **kwargs)
                except EOFError:
                    break

    def load_nth_pickle(self, name, n, *args, **kwargs):
        with open(name, "rb") as f:
            if n <= 1:
                return pickle.load(f, *args, **kwargs)
            current_obj = 1
            bstream = f.read()
            loc = 0
            while bstream:
                nloc = bstream.find(b'.\x80\x05\x95')
                if nloc == -1:
                    break
                current_obj += 1
                loc += nloc + 1
                f.seek(loc)
                if current_obj == n:
                    break
                bstream = f.read()

            return pickle.load(f, *args, **kwargs)

    def to_shelve(self, data: dict, name, *args, **kwargs):
        name = Path(name)
        name.parent.mkdir(parents=True, exist_ok=True)
        name = str(name.resolve())
        with shelve.open(name, *args, **kwargs) as m_shelf:
            for datum in data:
                m_shelf[repr(datum)] = data[datum]

    def from_shelve(self, name, *args, **kwargs):
        try:
            return shelve.open(name, *args, **kwargs)
        except Exception as _err:
            raise IOError(f"unable to read {name}") from _err

    def to_yaml(self, data, name):
        with open(name, "w+") as ofile:
            ofile.write(yaml.dump(data, default_flow_style=None))
            ofile.write("\n")

    def to_json(self, data, name):
        raise NotImplementedError("to json format is not implemented")

    def from_json(name):
        raise NotImplementedError("from json format is not implemented")

    def to_txt(data, name):
        raise NotImplementedError("to txt format is not implemented")

    def from_txt(name):
        raise NotImplementedError("from txt format is not impletmentd")
