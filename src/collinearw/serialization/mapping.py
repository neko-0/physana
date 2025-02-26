"""
class for handling serialization of class object into various format
"""

from .base import SerializationBase
from .histo import SerialHistogram
from .mixin import SerialProcessSet
from .mixin import SerialProcess
from .mixin import SerialRegion
from .config import SerialConfig
from .xsec import SerialXSecFile


class Serialization(SerializationBase):
    _structure = {}
    _structure["base"] = SerializationBase()
    _structure["config"] = SerialConfig()
    _structure["process_set"] = SerialProcessSet()
    _structure["process"] = SerialProcess()
    _structure["region"] = SerialRegion()
    _structure["histogram"] = SerialHistogram()
    _structure["xsec"] = SerialXSecFile()

    def __new__(cls, key="base"):
        return Serialization._structure[key]
