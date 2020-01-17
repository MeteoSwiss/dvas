
from abc import ABC, abstractmethod
from pathlib import Path

from ..config.config import RawData


class DataLinker(ABC):

    @abstractmethod
    def load(self):
        """Data loading method"""
        pass

    @abstractmethod
    def save(self):
        """Data saving method"""
        pass


class LocalDBLinker(DataLinker):

    def load(self, name):
        print(f'Execute {self.__class__.__name__}.load()')

    def save(self, name, data):
        print(f'Execute {self.__class__.__name__}.save()')


class CSVLinker(DataLinker):

    def __init__(self, repo: Path, config_mngr: RawData):
        self._repo = repo
        self._config_mngr = config_mngr

    def load(self, name):
        pass

    def save(self):
        pass
