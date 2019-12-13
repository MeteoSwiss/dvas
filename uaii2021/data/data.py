"""Module containing class and function for data management

"""

from abc import ABC, abstractmethod
import pandas as pd

from dataclasses import dataclass
from typing import List

from ..config.config import RawData
from .linker import DataLinker

@dataclass
class ProfileData:
    name: str
    data: pd.Series
    index_unit: str
    linker: DataLinker

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self):
        pass

    def rescale(self):
        pass

    def layering(self):
        pass


@dataclass
class TimeProfileData(ProfileData):
    ref_time: pd.Timestamp
    offset: int = 0
    index_unit: str = 'ms'

    def resample(self):
        pass

    def load(self):
        pass

    def save(self):
        pass

    def read_csv(self, *args, **kwargs):
        """

        Parameters
        ----------
        args: Pandas read_csv positional arguments
        kwargs: Pandas read_csv keyword arguments

        Returns
        -------

        """
        pass

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