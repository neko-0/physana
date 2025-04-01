"""
class for handling serialization of class object into various format
"""

from typing import Dict

from .base import SerializationBase
from .histo import SerialHistogram
from .mixin import SerialProcessSet, SerialProcess, SerialRegion
from .config import SerialConfig
from .xsec import SerialXSecFile


class Serialization(SerializationBase):
    """
    Class for handling serialization of class object into various format

    Parameters
    ----------
    key : str, optional
        The key to specify the type of serialization. Currently, the following
        keys are supported:

        - "base" : the base class for serialization
        - "config" : the class for serializing ConfigMgr
        - "process_set" : the class for serializing ProcessSet
        - "process" : the class for serializing Process
        - "region" : the class for serializing Region
        - "histogram" : the class for serializing Histogram
        - "xsec" : the class for serializing XSecFile

    Returns
    -------
    SerializationBase
        The instance of the specified serialization class
    """

    _structure: Dict[str, SerializationBase] = {}
    _structure["base"] = SerializationBase()
    _structure["config"] = SerialConfig()
    _structure["process_set"] = SerialProcessSet()
    _structure["process"] = SerialProcess()
    _structure["region"] = SerialRegion()
    _structure["histogram"] = SerialHistogram()
    _structure["xsec"] = SerialXSecFile()

    def __new__(cls, key: str = "base") -> SerializationBase:
        return Serialization._structure[key]
