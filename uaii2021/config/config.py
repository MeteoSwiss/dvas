"""Module containing class and function for config management

"""

# Import python packages and modules
import re
from abc import ABC, abstractmethod, ABCMeta

from pathlib import Path

from itertools import takewhile

import json
from jsonschema import validate, exceptions

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

import numpy as np
from mdtpyhelper.check import check_arg, check_type, check_list_item
from mdtpyhelper.check import check_path, check_notempty
from mdtpyhelper.check import CheckfuncAttributeError
from mdtpyhelper.misc import camel_to_snake
from mdtpyhelper.misc import get_by_path
from mdtpyhelper.misc import timer

from . import ITEM_SEPARATOR
from . import CONFIG_NAN_EQ, X_CHAR
from . import CONST_KEY_NM, CONST_KEY_PATTERN, CONFIG_ITEM_PATTERN
from . import RAW_DATA_CONFIG_PARAM_NO_X
from . import rawdata
from . import qualitycheck


class IdentifierManager:
    """Abstract class for managing unique identifier"""

    __ITEM_SEPARATOR = ITEM_SEPARATOR

    def __init__(self, node_order):
        """

        Parameters
        ----------
        node_order: list
            Must be item of CONFIG_ITEM_PATTERN.keys()
        """
        # Check node_order
        assert all([arg in CONFIG_ITEM_PATTERN.keys() for arg in node_order]) is True, (
            f'{node_order} not all in {CONFIG_ITEM_PATTERN.keys()}')

        # Set protected attributes
        self._node_order = node_order
        self._node_pat = [
            CONFIG_ITEM_PATTERN[key] for key in self.node_order
        ]
        self._node_str_pat = [
            key.pattern for key in self.node_pat
        ]

    @property
    def node_order(self):
        """Identifier node order. Must be key of CONFIG_ITEM_PATTERN"""
        return self._node_order

    @property
    def node_pat(self):
        """Construct node order compiled pattern"""
        return self._node_pat

    @property
    def node_str_pat(self):
        """Construct node order string pattern"""
        return self._node_str_pat

    @classmethod
    def split(cls, item):
        """Split str by item separator into list

        Parameters
        ----------
        item: str | list

        Returns
        -------
        list of str

        Examples
        --------
        >>>IdentifierManager.split('test.foo')
        ['test','foo']
        >>>IdentifierManager.split(['test', 'foo'])
        ['test','foo']
        """
        return item.split(cls.__ITEM_SEPARATOR) if isinstance(item, str) else item

    @classmethod
    def join(cls, item):
        """Split str by item separator into list

        Parameters
        ----------
        item: str | list of str

        Returns
        -------
        str

        Examples
        --------
        >>>IdentifierManager.split(['test', 'foo'])
        'test.foo'
        >>>IdentifierManager.split('test.foo')
        'test.foo'
        """
        return cls.__ITEM_SEPARATOR.join(item) if isinstance(item, list) else item

    def get_item_id(self, id_source, item_key):
        """Return item key corresponding id

        Parameters
        ----------
        id_source: str | list of str
        item_key: str
            Item key. Must be element of node_order

        Returns
        -------
        str
        """

        # Init
        id_source = self.split(id_source)

        item_pat = self.node_pat[self.node_order.index(item_key)]

        a = (
            (item_key,) +
            self.count_match([re.fullmatch(item_pat, id_src_arg) for id_src_arg in id_source])
        )

        if a[1] >= 2:
            raise IDNodeError(f"Many identical pattern id in {id_source}")

        out = list(takewhile(lambda arg: arg[1] == 1, [a]))

        if not out:
            raise IDNodeError(f'No match in {id_source}')

        return out[0][2][0]

    def create_id(self, id_source, strict=True):
        """Create default id

        Parameters
        ----------
        id_source: list of str
        strict: bool (default: True)

        Returns
        -------
        list of str
        """

        # Init
        id_source = self.split(id_source)

        a = [
            (node_nm,) + self.count_match([re.fullmatch(node_pat_arg, id_src_arg) for id_src_arg in id_source])
            for node_nm, node_pat_arg in zip(self.node_order, self.node_pat)
        ]

        # Check for id with multiple node name
        check = [arg[0] for arg in a if arg[1] >= 2]
        if check:
            raise IDNodeError(f"Many identical pattern id in {id_source}")

        out = list(takewhile(lambda arg: arg[1] == 1, a))

        if not out:
            raise IDNodeError('No match')

        if strict:
            if len(out) != len(self.node_order):
                found_keys = list(zip(*out))
                out_not_fullfilled = [
                    arg for arg in self.node_order if arg not in found_keys[0]
                ]
                errmsg = f'Strict condition not fulfilled for {out_not_fullfilled}'
                raise IDNodeError(errmsg)

        out = [arg[2][0] for arg in out]

        return out

    def check_dict_node(self, document, end_node_pat):
        """Check dictionary key node order. End node must match end_node_pat.

        Parameters
        ----------
        document: dict
            Document to be check
        end_node_pat: str | re.Pattern
            End node pattern
        """

        def scan_nodes(doc, id_source):
            if isinstance(doc, dict) is False:
                raise IDNodeError(f"Missing end node patter {end_node_pat}")
            for key, sub_doc in doc.items():
                if re.fullmatch(end_node_pat, key):
                    if not id_source:
                        continue
                    id_out = self.create_id(id_source, strict=False)
                    if len(id_source) != len(id_out):
                        raise IDNodeError(f"Error in key {key}")
                else:
                    id_source_sub = id_source.copy()
                    id_source_sub.append(key)
                    scan_nodes(sub_doc, id_source_sub.copy())

        scan_nodes(document, [])

    @staticmethod
    def count_match(matches):
        """Return (count of match, list of match group(0)) in matches

        Parameters
        ----------
        matches: list
            Result of re.match or re.fullmatch for a list of str

        Returns
        -------
        tuple
        """

        val_flag = [arg is not None for arg in matches]
        val_grp = [arg.group(0) for arg in matches if arg is not None]
        return sum(val_flag), val_grp


class ConfigManagerMeta(ABCMeta):
    REQUIRED_ATTRIBUTES = []
    instances = []
    def __call__(cls, *args, **kwargs):
        obj = super(ConfigManagerMeta, cls).__call__(*args, **kwargs)
        for attr_name in obj.REQUIRED_ATTRIBUTES:
            if not getattr(obj, attr_name):
                raise ValueError('required attribute (%s) not set' % attr_name)
        return obj


class ConfigManager(object, metaclass=ConfigManagerMeta):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = ['_parameter_schema', '_root_params_def']

    @abstractmethod
    def __init__(self, config_dir_path, id_node_order):
        """
        Parameters
        ----------
        config_dir_path: pathlib.Path | str
            Config files directory path. Directory must exist.
        id_node_order: list of str
            IdentifierManager node_order argument
        """

        # Convert to Path
        config_dir_path = Path(config_dir_path)

        # Set protected attributes
        self._id_mngr = IdentifierManager(id_node_order)
        self._config_dir_path = config_dir_path

        # Set public attributes
        self.data = {}

    def __getitem__(self, item):
        """Return config item value

        Parameters
        ----------
        item: str | list of str

        Returns
        -------
        object
        """

        # Define
        errmsg = "Bad item '{}' in __getitem__({})".format(item, self.data)

        # Split str by item separator
        item = self._id_mngr.split(item)

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

        # Get all parameters
        param_keys = self._get_parameter_keys()

        nested_key = self._id_mngr.create_id(nested_key, strict=False)

        out = {key: self[nested_key + [CONST_KEY_NM] + [key]] for key in param_keys}

        return out

    @property
    def data(self):
        """Data"""
        return self._data

    @data.setter
    def data(self, val):
        assert isinstance(val, dict)
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

    @property
    def parameter_schema(self):
        """JSON parameter schema. Constant value.
        Must be a dict like {"type": "object", "patternProperties": ANY, "additionalProperties": False}
        """
        return self._parameter_schema

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "parametersItem": self.parameter_schema,

            "nodeItem": {
                "type": 'object',
                "patternProperties": {
                    rf"^({CONST_KEY_PATTERN.pattern})$": {"$ref": "#/parametersItem"},
                    rf"({')|('.join(self._id_mngr.node_str_pat)})": {"$ref": "#/nodeItem"}
                },
                "additionalProperties": False
            },

            "type": 'object',
            "oneOf": [
                {"$ref": "#/nodeItem"}
            ]
        }

    @property
    def root_params_def(self):
        """Default root parameters field. Constant value.
        Must be a dict like {PARAM_KEY_NM: ANY}
        """
        return self._root_params_def

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
        assert self.config_file_path.exists(), f"Missing file {self.config_file_path}"

        # Open file
        try:
            with self.config_file_path.open() as fid:
                # Load yaml
                document = YAML().load(fid)
                document = document if document else {}

                # Use json to convert ordered dict to dict
                document = json.loads(json.dumps(document))

        except IOError as e:
            raise ConfigReadError(e)
        except YAMLError as e:
            raise ConfigReadError(e)

        # Append ROOT_PARAMS
        document.update({CONST_KEY_NM: self.root_params_def})

        self.doc_tmp = document

        # Check json schema validity
        try:
            validate(instance=document, schema=self.json_schema)

        except exceptions.ValidationError as e:
            raise ConfigReadError(f"Error in '{self.config_file_path}'\n{e}")

        except exceptions.SchemaError as e:
            raise ConfigReadError(f"Error in {self.__class__.__name__}.json_schema\n{e}")

        # Check node order
        try:
            self._id_mngr.check_dict_node(document, CONST_KEY_PATTERN)
        except Exception as e:
            raise ConfigReadError(f"Error in '{self.config_file_path}'\n{e}")

        return document

    def _get_parameter_keys(self):
        """"""

        # Get all defined parameters keys
        keys = list(self.root_params_def.keys())

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
    def instantiate_all_childs(config_dir_path):
        """Generate a dictionary with instances of all ConfigManager childs

        Parameters
        ----------
        config_dir_path: str | pathlib.Path
            Config files directory path. Directory must exist.

        Returns
        -------
        dict
            key, snake name of ConfigManager child instance
            value: ConfigManager child instance
        """

        # Test
        assert isinstance(config_dir_path, (str, Path)) is True
        assert Path(config_dir_path).exists() is True

        # Create instances
        inst = set()
        for subclass in ConfigManager.__subclasses__():
            inst.add(subclass(config_dir_path))

        return {arg.snake_name: arg for arg in inst}


class RawData(ConfigManager):

    def __init__(self, config_dir_path):
        # Init super class
        super().__init__(config_dir_path, rawdata.NODE_ORDER)

        # Set required attributes
        self._parameter_schema = rawdata.PARAMETER_SCHEMA
        self._root_params_def = rawdata.ROOT_PARAMS_DEF


class QualityCheck(ConfigManager):

    def __init__(self, config_dir_path):
        # Init super class
        super().__init__(config_dir_path, qualitycheck.NODE_ORDER)

        # Set required attributes
        self._parameter_schema = qualitycheck.PARAMETER_SCHEMA
        self._root_params_def = qualitycheck.ROOT_PARAMS_DEF


class ConfigReadError(Exception):
    pass


class IDNodeError(Exception):
    pass


class ConfigItemKeyError(KeyError):
    pass


class MaxConfigInstance(Exception):
    pass
