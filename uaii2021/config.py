"""Module containing class and function for config managment

"""

# Import python packages and modules
import re
from jsonschema import validate, exceptions
import yaml
from abc import ABC, abstractmethod
import numpy as np
from mdtpyhelper.check import check_arg, check_type, check_path, check_notempty
from mdtpyhelper.misc import camel_to_snake
from mdtpyhelper.misc import get_by_path

from . import RAW_DATA_CONFIG_PARAM

# Define
RAW_DATA_CONFIG_PARAM_NO_X = [arg for arg in RAW_DATA_CONFIG_PARAM if arg is not 'x']


class ConfigManager(ABC):
    """Abstract class for managing YAML config"""

    __ITEM_SEPARATOR = '.'

    @check_arg(1, check_path)
    def __init__(self, config_dir_path):
        """
        Parameters
        ----------
        config_dir_path: pathlib.Path
            Config files directory path. Directory must exist.
        """
        self.data = {}
        self.config_dir_path = config_dir_path

    @check_arg(1, check_notempty)
    @check_arg(1, check_type, (str, list))
    def __getitem__(self, item):
        """Return config item value
        """

        # Split str by item separator
        if type(item) is str:
            item = item.split(self.__ITEM_SEPARATOR)

        # Define nested function
        def find_val(nested_item):

            try:
                if len(nested_item) >= 2:
                    return get_by_path(self.data, nested_item)
                else:
                    return get_by_path(self.data, ['master'] + nested_item)

            except (KeyError, TypeError, IndexError):
                if len(nested_item) == 1:
                    raise KeyError
                find_val(nested_item[:-2] + nested_item[-1:])

        # Replace by 'x' last item arg
        item_x = item.copy()
        item_x[-1] = re.sub(r'^([\w_])+(_[\w_]*$)', r'x\2', item_x[-1])

        try:
            # Get for unmodified item
            out = find_val(item)
        except KeyError:
            try:
                # Got for modified with 'x' item
                out = find_val(item_x)
            except KeyError:
                raise ConfigItemError

        return out

    @property
    def data(self):
        """Data"""
        return self._data

    @data.setter
    @check_arg(1, check_type, dict)
    def data(self, val):
        self._data = val

    @property
    def snake_name(self):
        """Class name in snake case"""
        return camel_to_snake(self.__class__.__name__)

    @property
    def config_file_name(self):
        """Config file name"""
        return self.snake_name + '_config.yml'

    @property
    def config_dir_path(self):
        """Config files directory path"""
        return self._config_dir_path

    @property
    def config_file_path(self):
        return self.config_dir_path / self.config_file_name

    @config_dir_path.setter
    @check_arg(1, check_path)
    def config_dir_path(self, value):
        self._config_dir_path = value

    @property
    @abstractmethod
    def JSONSCHEMA(self):
        """JSON schema. Constant value"""
        pass

    @property
    @abstractmethod
    def MASTER(self):
        """Master field. Constant value"""
        pass

    def update(self):
        self.read()

    def read(self):
        self.data = self._get_document()

    def _get_document(self):
        """Get YAML document as python dict

        Returns
        -------
        dict:
            Config from YAML file
        """

        # Check file existence
        assert self.config_file_path.exists(), "Missing file {}".format(self.config_file_path)

        # Open file
        try:
            with self.config_file_path.open() as fid:
                document = yaml.load(fid, Loader=yaml.SafeLoader)
        except IOError as e:
            raise ConfigReadError(e)
        except yaml.scanner.ScannerError as e:
            raise ConfigReadError(e)

        # Append MASTER
        document.update(self.MASTER)

        # Check schema validity
        try:
            validate(instance=document, schema=self.JSONSCHEMA)

        except exceptions.ValidationError as e:
            raise ConfigReadError("Error in '{}'\n{}".format(self.config_file_path, e))

        except exceptions.SchemaError as e:
            raise ConfigReadError("Error in {}.JSONSCHEMA\n{}".format(self.__class__.__name__, e))

        return document

    @staticmethod
    def instantiate_all_childs(config_dir_path):
        """Generate a dictionary with instances of all ConfigManager childs

        Parameters
        ----------
        config_dir_path: pathlib.Path
            Config files directory path. Directory must exist.

        Returns
        -------
        dict
            key, snake name of ConfigManager child instance
            value: ConfigManager child instance
        """
        inst = set()
        for subclass in ConfigManager.__subclasses__():
            inst.add(subclass(config_dir_path))

        return {arg.snake_name: arg for arg in inst}


class RawData(ConfigManager):

    @property
    def MASTER(self):
        return {
            'master': {
                'idx_unit': 'ms',
                'dt_format': None,
                'delimiter': ';',
                'usecols': list(range(len(RAW_DATA_CONFIG_PARAM_NO_X))),
                'namecols': RAW_DATA_CONFIG_PARAM_NO_X,
                'x_dec': -3,
                'x_a': 1.0,
                'x_b': 0.0,
                'type_name': None,
                'skiprows': 0
            }
        }

    @property
    def JSONSCHEMA(self):
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "definitions": {
                "address": {
                    "type": "object",
                    "patternProperties": {
                        r"^idx_unit$": {
                            "type": "string",
                            "enum": ['dt', 'ms', 'meters']
                        },
                        r"^dt_format$": {
                            'anyOff': [
                                {"type": "null"},
                                {"type": 'string'}
                            ]
                        },
                        r"^delimiter$": {"type": 'string'},
                        r"^usecols$": {
                            "type": 'array',
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "minItems": 1,
                            "uniqueItems": True
                        },
                        r"^namecols$": {
                            "type": 'array',
                            "items": {
                                "type": "string",
                                'enum': RAW_DATA_CONFIG_PARAM_NO_X
                            },
                            "minItems": 1,
                            "uniqueItems": True
                        },
                        r"^({})_dec$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'integer', "maximum": 3, "minimum": -4},
                        r"^({})_a$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'number'},
                        r"^({})_b$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'number'},
                        r"^type_name$": {
                            'anyOff': [
                                {"type": "null"},
                                {"type": 'string'}
                            ]
                        },
                        r"^skiprows$": {
                            'anyOff': [
                                {"type": "int", 'minimum': 0},
                                {"type": 'string'}
                            ]
                        }
                    },
                    "additionalProperties": False
                }
            },

            "type": "object",
            "patternProperties": {
                r"^((master)|(\d{2}))$": {
                    "allOf": [
                        {"$ref": "#/definitions/address"}
                    ]
                }
            },
            "additionalProperties": False
        }


class QualityCheck(ConfigManager):

    @property
    def MASTER(self):
        return {
            'master': {
                'idx': [],
                'replace_param': 'x',
                'replace_val': np.nan
            }
        }

    @property
    def JSONSCHEMA(self):
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "definitions": {
                "address": {
                    "type": "object",
                    "patternProperties": {
                        r"^idx$": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "minItems": 0,

                            "uniqueItems": True
                        },
                        r"^dt_format$": {
                            'anyOff': [
                                {"type": "null"},
                                {"type": 'string'}
                            ]
                        },
                        r"^delimiter$": {"type": 'string'},
                        r"^usecols$": {
                            "type": 'array',
                            "items": {
                                "type": "integer",
                                "minimum": 0,
                            },
                            "minItems": 1,
                            "uniqueItems": True
                        },
                        r"^namecols$": {
                            "type": 'array',
                            "items": {
                                "type": "string",
                                'enum': RAW_DATA_CONFIG_PARAM_NO_X
                            },
                            "minItems": 1,
                            "uniqueItems": True
                        },
                        r"^({})_dec$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'integer', "maximum": 3, "minimum": -4},
                        r"^({})_a$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'number'},
                        r"^({})_b$".format('|'.join(RAW_DATA_CONFIG_PARAM)): {"type": 'number'},
                        r"^type_name$": {
                            'anyOff': [
                                {"type": "null"},
                                {"type": 'string'}
                            ]
                        },
                        r"^skiprows$": {
                            'anyOff': [
                                {"type": "int", 'minimum': 0},
                                {"type": 'string'}
                            ]
                        }
                    },
                    "additionalProperties": False
                }
            },

            "type": "object",
            "patternProperties": {
                r"^((master)|(\d{2}))$": {
                    "allOf": [
                        {"$ref": "#/definitions/address"}
                    ]
                }
            },
            "additionalProperties": False
        }


class ConfigReadError(Exception):
    pass

class ConfigItemError(AttributeError):
    pass