"""Module containing class and function for config management

"""

# Import python packages and modules
import re
from abc import ABC, abstractmethod

from pampy import match
from pampy import ANY

import json
from jsonschema import validate, exceptions

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

import numpy as np
from mdtpyhelper.check import check_arg, check_type, check_path, check_notempty
from mdtpyhelper.check import CheckfuncAttributeError
from mdtpyhelper.misc import camel_to_snake
from mdtpyhelper.misc import get_by_path

from . import CONFIG_NAN_EQ, X_CHAR
from . import PARAM_KEY_NM, NODE_PAT_DICT
from . import RAW_DATA_CONFIG_PARAM_NO_X
from . import rawdata
from . import qualitycheck


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

        self._item_order_pat = [
            r'{}'.format(NODE_PAT_DICT[key]) for key in self.NODE_ORDER
        ]

    def _split_sep(self, item):
        """Split str by item separator"""
        return item.split(self.__ITEM_SEPARATOR) if type(item) is str else item

    @check_arg(1, check_notempty)
    @check_arg(1, check_type, (str, list))
    def __getitem__(self, item):
        """Return config item value
        """

        # Define
        errmsg = "Bad item '{}' in __getitem__({})".format(item, self.data)

        # Split str by item separator
        item = self._split_sep(item)

        # Define nested function
        def find_val(nested_item):
            try:
                return get_by_path(self.data, nested_item)

            except (KeyError, TypeError, IndexError):
                if len(nested_item) <= 2:
                    raise KeyError
                return find_val(nested_item[:-3] + nested_item[-2:])

        # Get for item
        try:
            out = find_val(item)
        except KeyError:
            # Get for modified with x char item
            try:
                # Replace by x char last item arg
                item_x = item.copy()
                item_x[-1] = re.sub(
                    r'^([\w_])+(_[\w_]*$)', r'{}\2'.format(X_CHAR), item_x[-1])

                out = find_val(item_x)
            except KeyError:
                raise ConfigItemKeyError(errmsg)

        # Replace nan equivalent value
        out = np.nan if out == CONFIG_NAN_EQ else out

        return out

    @check_arg(1, check_type, (str, list))
    def get_all(self, nested_key):

        # Check list item
        item_order_pat_comp = [
            re.compile('^({})$'.format(arg)) for arg in self.item_order_pat
        ]
        self._check_list_item(nested_key, item_order_pat_comp)

        # Get all parameters
        param_keys = self._get_parameter_keys()

        # Split str by item separator
        nested_key = self._split_sep(nested_key)

        out = {key: self[nested_key + [PARAM_KEY_NM] + [key]] for key in param_keys}

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
    def PARAMETER_SCHEMA(self):
        """JSON parameter schema. Constant value.
        Must be a dict like {"type": "object", "patternProperties": ANY, "additionalProperties": False}
        """
        pass

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "parametersItem": self.PARAMETER_SCHEMA,

            "nodeItem": {
                "type": 'object',
                "patternProperties": {
                    r'^({})$'.format(NODE_PAT_DICT['param']): {"$ref": "#/parametersItem"},
                    r'^({})$'.format('(' + ')|('.join(self.item_order_pat) + ')'): {"$ref": "#/nodeItem"}
                },
                "additionalProperties": False
            },

            "type": 'object',
            "oneOf": [
                {"$ref": "#/nodeItem"}
            ]
        }

    @property
    @abstractmethod
    def ROOT_PARAMS_DEF(self):
        """Default root parameters field. Constant value.
        Must be a dict like {PARAM_KEY_NM: ANY}
        """
        pass

    @property
    @abstractmethod
    def NODE_ORDER(self):
        """Document node order. Must be key of NODE_PAT_DICT except PARAM_KEY_NM"""
        pass

    @property
    def item_order_pat(self):
        """Construct item order pattern"""
        return self._item_order_pat

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
                # Load yaml
                document = YAML().load(fid)
                document = document if document else {}

                # Use json to convert ordereddict to dict
                document = json.loads(json.dumps(document))

        except IOError as e:
            raise ConfigReadError(e)
        except YAMLError as e:
            raise ConfigReadError(e)

        # Append ROOT_PARAMS
        document.update(self.ROOT_PARAMS_DEF)

        # Check json schema validity
        try:
            validate(instance=document, schema=self.json_schema)

        except exceptions.ValidationError as e:
            raise ConfigReadError("Error in '{}'\n{}".format(self.config_file_path, e))

        except exceptions.SchemaError as e:
            raise ConfigReadError("Error in {}.json_schema\n{}".format(self.__class__.__name__, e))

        # Check node order
        try:
            # Compile pattern
            node_order_pat_comp = [
                re.compile('^({})$'.format(arg)) for arg in self.item_order_pat
            ]
            # Check nodes
            self._check_dict_nodes(
                document,
                node_order_pat_comp,
                NODE_PAT_DICT[PARAM_KEY_NM])
        except ConfigNodeError as e:
            raise ConfigReadError("Error in '{}'\n{}".format(self.config_file_path, e))

        return document

    def _get_parameter_keys(self):
        """"""

        # Get all defined parameters keys
        keys = list(self.ROOT_PARAMS_DEF[PARAM_KEY_NM].keys())

        # Replace and update parameter beginning with x char
        keys_new = []
        idx = []
        for i, key in enumerate(keys):

            pat = r'^{}(_[\w_]*$)'.format(X_CHAR)
            if re.match(pat, key):
                keys_new += [re.sub(pat, r'{}\1'.format(arg), key) for arg in RAW_DATA_CONFIG_PARAM_NO_X]
                idx.append(i)

        keys = [key for i, key in enumerate(keys) if i not in idx] + keys_new

        return keys

    @staticmethod
    def _check_list_item(document, item_pat):
        """

        Parameters
        ----------
        document
        item_pat

        Returns
        -------

        """

        def check_single_item(doc, pat):
            if type(doc) is list:
                if not doc:
                    return

                if not pat:
                    err_msg = "Bad item. No key to match in {}".format(doc)
                    raise ConfigItemKeyError(err_msg)

                if re.match(pat[0], doc[0]) is None:
                    err_msg = "Bad node. '{}' didn't match key in {}".format(pat[0], doc)
                    raise ConfigItemKeyError(err_msg)
                else:
                    check_single_item(doc[1:], pat[1:])
            else:
                err_msg = "Bad item. '{}' isn't of type list".format(doc)
                raise ConfigItemKeyError(err_msg)

        check_single_item(document, item_pat)

    @staticmethod
    def _check_dict_nodes(document, node_pat, end_node_pat):
        """Check document key node order

        Parameters
        ----------
        document: dict
            Document to be check
        node_pat: list of str | list of re.Pattern
            Ordered node pattern as regular expression
        end_node_pat: str | re.Pattern
        """

        def check_single_node(doc, pat):
            if type(doc) is dict:

                for key, sub_doc in doc.items():
                    if re.match(end_node_pat, key):
                        continue

                    if not pat:
                        err_msg = "Bad node. No key to match in {}".format(doc)
                        raise ConfigNodeError(err_msg)

                    if re.match(pat[0], key) is None:
                        err_msg = "Bad node. '{}' didn't match key in {}".format(pat[0], doc)
                        raise ConfigNodeError(err_msg)
                    else:
                        check_single_node(sub_doc, pat[1:])

            else:
                err_msg = "Bad node. '{}' isn't of type dict".format(doc)
                raise ConfigNodeError(err_msg)

        check_single_node(document, node_pat)

    @staticmethod
    @check_arg(0, check_path)
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
    def NODE_ORDER(self):
        return rawdata.NODE_ORDER

    @property
    def ROOT_PARAMS_DEF(self):
        return rawdata.ROOT_PARAMS_DEF

    @property
    def PARAMETER_SCHEMA(self):
        return rawdata.PARAMETER_SCHEMA


class QualityCheck(ConfigManager):

    @property
    def NODE_ORDER(self):
        return qualitycheck.NODE_ORDER

    @property
    def ROOT_PARAMS_DEF(self):
        return qualitycheck.ROOT_PARAMS_DEF

    @property
    def PARAMETER_SCHEMA(self):
        return qualitycheck.PARAMETER_SCHEMA


class ConfigReadError(Exception):
    pass


class ConfigNodeError(ConfigReadError):
    pass


class ConfigItemKeyError(KeyError):
    pass
