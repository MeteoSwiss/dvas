"""Module containing class and function for data management

"""

from abc import ABC, abstractmethod
import pandas as pd

from dataclasses import dataclass, field
from typing import List

from ..config.config import IdentifierManager
from ..config.config import RawData
from .linker import DataLinker


class ProfileData(IdentifierManager):

    __NODE_ORDER = ['ms', 'param', 'ref_dt']

    def __init__(self, name: str, data: pd.Series, index_unit: str):
        self.name = name
        self.data = data
        self.index_unit = index_unit

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @property
    def data(self) -> pd.Series:
        return self._data

    @data.setter
    def data(self, val):
        self._data = val

    @property
    def index_unit(self) -> str:
        return self._index_unit

    @index_unit.setter
    def index_unit(self, val):
        self._index_unit = val

    def load(self, data_linker: DataLinker):
        data = data_linker.load(self.name)
        if data:
            self.data = data['data']
            self.name = data['name']
        return data

    def save(self, data_linker: DataLinker):
        data_to_save = {arg: self.__getattribute__(arg) for arg in self.__dict__.keys()}
        data_linker.save(self.name, data_to_save)

    def rescale(self, fct):
        pass

    def layering(self):
        pass

    def change_index_unit(self, new_unit: str):
        pass


@dataclass
class TimeProfileData(ProfileData):
    ref_time: pd.Timestamp
    offset: int = 0
    index_unit: str = 'ms'

    def resample(self):
        pass


class UAIITimeProfileData(TimeProfileData):

    __NODE_ORDER = ['flight', 'batch', 'instr', 'instr_type', 'meteo_param']


@dataclass
class GroupTimeProfileData:
    data: List[TimeProfileData]

    def synchronise(self):
        pass



class CSVReader:

    def __init__(self):


class TextExtractor(ABC):

    @abstractmethod
    def extract(self):
        """Text extractor"""
        pass

    @staticmethod
    def get_instance(name):
        pass

class TextFileExtractor(TextExtractor):
    pass