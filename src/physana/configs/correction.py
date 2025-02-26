import ast
from pathlib import Path

from ..serialization import Serialization


class CorrectionContainer:
    """
    Container for holding systematics histogram for weight lookup.
    This container will only try to lookup requested histogram from the database.
    It's better to use shelve for object persistance to avoid large dictionary
    deserialization. The database by default will be clear after each lookup, and
    this is handled by the db_persistance boolean switch.

    TODO:
        1) need to method to load everything into cache
        2) tuple key is parsed into string when listing, maybe use ast.literal_eval?
    """

    def __init__(self):
        self.files = {}
        self.enable_interanl_cache = True
        # if db_persistance is False, cleanup self._database after each __getitem__
        self.db_persistance = False
        self._correction = {}  # for caching
        self._file_backend = {}
        self._serial = Serialization()
        self._database = []
        self._loaded = False
        self._non_exist_keys = set()  # cache for non_existing keys

    def __getitem__(self, key):
        """
        Lazy get item method.
        """
        if key in self._correction:
            return self._correction[key]
        elif key in self._non_exist_keys:
            return None

        self.load_correction()
        repr_key = repr(key)
        for db in self._database:
            m_key = key if isinstance(db, dict) else repr_key
            try:
                found_corr = db[m_key]
                break
            except KeyError:
                continue
        else:
            found_corr = None
        if not self.db_persistance:
            self.clear_database()
        if self.enable_interanl_cache:
            self._correction[key] = found_corr
        if found_corr is None:
            self._non_exist_keys.add(key)

        return found_corr

    def __setitem__(self, key, value):
        self._correction[key] = value

    def __contains__(self, key):
        return key in self._correction

    def items(self):
        return self._correction.items()

    def keys(self):
        return self._correction.keys()

    def update(self, input: dict):
        for key in input:
            self.__setitem__(key, input[key])

    def list_correction(self):
        output = set()
        for db in self._database:
            output |= set(db.keys())
        return output

    def add_correction_file(self, filename, backend="shelve"):
        filename = Path(filename).resolve()
        if filename not in self.files:
            self.files[filename] = False
            self._file_backend[filename] = backend
        self._loaded = False

    def list_correction_file(self):
        return self.files.keys()

    def remove_correction_file(self, name):
        del self.files[name]

    def load_correction(self):
        if self._loaded:
            return
        for f in self.files:
            if self.files[f] != False:
                continue
            m_backend = self._file_backend[f]
            if m_backend == "shelve":
                db = self._serial.from_shelve(str(f), flag="r", writeback=False)
            elif m_backend == "pickle":
                db = self._serial.from_pickle(str(f))
            else:
                raise TypeError(f"Correction does not support {m_backend}")
            self._database.append(db)
            self.files[f] = True
        self._loaded = True

    def load_in_memory(self):
        self.load_correction()
        for name in self.list_correction():
            if isinstance(name, str):
                # usually handle tuple instead of str for lookup
                name = ast.literal_eval(name)
            self.__getitem__(name)

    def clear_buffer(self):
        self._correction = {}
        self.reset_files()
        self.clear_database()

    def clear_database(self):
        if not self._database:
            return
        for db in self._database:
            if not isinstance(db, dict):
                db.close()
        self._database = []
        self._loaded = False
        self.reset_files()

    def clear_files(self):
        self.files = {}
        self._file_backend = {}

    def reset_files(self):
        for f in self.files:
            self.files[f] = False

    def remove(self, name):
        del self._correction[name]

    def save(self, name):
        """
        save filename into it's dict
        """
        name = Path(name).resolve()
        self.files[name] = True

    def save_current_correction(self, output):
        self._serial.to_shelve(self._correction, output, flag="n")
