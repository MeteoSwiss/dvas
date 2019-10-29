"""Module containing class and function for config managment

"""

# Import python packages and modules
from pathlib import Path
from jsonschema import validate, exceptions
import yaml
from abc import ABC, abstractmethod
from mdtpyhelper.check import check_arg, check_type, check_path
from mdtpyhelper.misc import camel_to_snake

from . import RAW_DATA_CONFIG_PARAM


class ConfigManager(ABC):
    """Abstract class for managing YAML config"""

    def __init__(self, config_dir_path):
        """
        Parameters
        ----------
        config_dir_path: str | pathlib.Path
            Config files directory path. Directory must exist.
        """
        config_dir_path = Path(config_dir_path) if (type(config_dir_path) is str) else config_dir_path
        self.config_dir_path = config_dir_path

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

    @config_dir_path.setter
    @check_arg(1, check_path)
    def config_dir_path(self, value):
        self._config_dir_path = value

    @property
    @abstractmethod
    def JSONSCHEMA(self):
        """JSON schema. Constant value"""
        pass

    def get_document(self):
        """Get YAML document as python dict

        Returns
        -------
        dict:
            Config from YAML file
        """

        # Check file existence
        file_path = self.config_dir_path / self.config_file_name
        assert file_path.exists(), "Missing file {}".format(file_path)

        # Open file
        with file_path.open() as fid:
            document = yaml.load(fid, Loader=yaml.SafeLoader)

        # Check schema validity
        try:
            self.validate_document(document, self.JSONSCHEMA)

        except exceptions.ValidationError as e:
            raise exceptions.ValidationError("Error in '{}'\n{}".format(file_path, e))

        except exceptions.SchemaError as e:
            raise exceptions.SchemaError("Error in {}.jsonschema\n{}".format(self.__class__.__name__, e))

        return document

    @staticmethod
    @check_arg(1, check_type, dict)
    def validate_document(document, schema):
        """Validate JSON document with given schema

        Parameters
        ----------
        document: object
            JSON document to be validated
        schema: dict
            Valide JSON schema
        """
        validate(instance=document, schema=schema)

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
                                'enum': [arg for arg in RAW_DATA_CONFIG_PARAM if arg is not 'x']
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
