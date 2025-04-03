from abc import ABC, abstractmethod
from copy import deepcopy

from ..configs import ConfigMgr


class BaseAlgorithm(ABC):

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def finalise(self) -> None:
        pass

    @abstractmethod
    def prepare(self, config: ConfigMgr) -> None:
        pass

    @abstractmethod
    def meta_data_from_config(self, config: ConfigMgr) -> None:
        pass

    @abstractmethod
    def process(self, config: ConfigMgr) -> None:
        pass

    def copy(self) -> 'BaseAlgorithm':
        return deepcopy(self)
