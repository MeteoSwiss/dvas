"""Module containing class and function for config management

"""

# Import python packages and modules
import os
import re
import pprint
from abc import abstractmethod, ABC

from pathlib import Path

import json
from jsonschema import validate, exceptions

from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError

from ..dvas_helper import get_by_path

# Import current package modules
from .pattern import INSTR_TYPE_KEY, INSTR_KEY
from .pattern import EVENT_KEY, PARAM_KEY
from .pattern import FLAG_KEY
from .pattern import ORIGDATA_KEY, ORIGMETA_KEY
from .definitions import origdata, origmeta
from .definitions import instrtype
from .definitions import instrument
from .definitions import event
from .definitions import parameter
from .definitions import flag
from ..dvas_environ import path_var as env_path_var

from ..dvas_helper import RequiredAttrMetaClass


# Define
NODE_ESCAPE_CHAR = '_'


def instantiate_config_managers(config_managers, config_dir_path=None, read=True):
    """Generate a dictionary with instances of all ConfigManagers

    Args:
        config_managers (list of ConfigManager):
        config_dir_path (str | pathlib.Path, optional): Config files directory path. Directory must exist.
        read (bool, optional): (Default value = True)

    Returns:

    """

    # Create instances
    instances = set()
    for config_manager in config_managers:
        instances.add(config_manager(config_dir_path))

    # Read
    if read:
        for inst in instances:
            inst.read()

    return {arg.class_key: arg for arg in instances}


class OneLayerConfigManager(ABC, metaclass=RequiredAttrMetaClass):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = {
        'document': dict,
        'parameter_pattern_prop': dict,
        'node_params_def': dict,
        'class_key': str
    }

    @abstractmethod
    def __init__(self, config_dir_path=None):
        """
        Args:
            config_dir_path (:obj:`pathlib.Path` | :obj:`str`, optional): Default to None
                Config files directory path. Directory must exist.

        """

        # Set attributes
        self.config_dir_path = config_dir_path

        # Set required attributes
        self._document = {}
        self._parameter_pattern_prop = {}
        self._node_params_def = {}
        self._class_key = ''

    def __getitem__(self, item):
        return self._document[item]

    def __repr__(self):
        pp = pprint.PrettyPrinter()
        return pp.pformat(self.document)

    @property
    def config_dir_path(self):
        """Config files directory path"""
        return self._config_dir_path

    @config_dir_path.setter
    def config_dir_path(self, value):

        # Set environement settings if None
        if value is None:
            value = env_path_var.config_dir_path

        # Convert to Path
        value = Path(value)

        # Convert to absolute path if relative
        if not value.is_absolute():
            value = Path(os.getcwd()) / value

        # Test existence
        assert value.exists()

        self._config_dir_path = value

    @property
    def document(self):
        """Config document"""
        return self._document

    @property
    def parameter_pattern_prop(self):
        """JSON parameter schema. Constant value.
        Must be a dict like {"type": "object", "patternProperties": ANY, "additionalProperties": False}
        """
        return self._parameter_pattern_prop

    @property
    def node_params_def(self):
        """ """
        return self._node_params_def

    @property
    def class_key(self):
        """ """
        return self._class_key

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "type": 'object',
            "patternProperties": self.parameter_pattern_prop,
            "additionalProperties": False
        }

    def init_document(self):
        if self.REQUIRED_ATTRIBUTES['document'] is dict:
            self._document = {}
        elif self.REQUIRED_ATTRIBUTES['document'] is list:
            self._document = []
        else:
            self._document = None

    def append(self, value):
        if self.REQUIRED_ATTRIBUTES['document'] is dict:
            self._document.update(value)
        else:
            self._document += value

    def read(self, doc_in=None):
        """Read config

        Args:
            doc_in (:obj:`str`, optional): Default None -> read from directiory
                Else read from doc.

        """

        self.init_document()
        self._get_document(doc_in)

        # Add missing default node params
        for key, val in self.node_params_def.items():
            if key not in self.document.keys():
                self._document[key] = val

    def _get_document(self, doc_in=None):
        """Get YAML document as python dict

        Args:
            doc_in (str | pathlib.Path):

        """

        if not doc_in:
            doc_in = self.config_dir_path.rglob(
                "*" + self.class_key + "*")

        # Convert YAML string as JSON dict
        if isinstance(doc_in, str):
            self.append(
                self.read_yaml(doc_in)
            )

        # Convert YAML file as JSON dict
        else:
            for filepath in doc_in:
                self.append(
                    self.read_yaml(Path(filepath))
                )

        # Check json schema validity
        try:
            validate(instance=self.document, schema=self.json_schema)

        except exceptions.ValidationError as e:
            raise ConfigReadError(f"Error in '{filepath}'\n{e}")

        except exceptions.SchemaError as e:
            raise ConfigReadError(f"Error in {self.__class__.__name__}.json_schema\n{e}")

    @staticmethod
    def read_yaml(file):
        """

        Args:
            file (str | pathlib.Path):

        Returns:

        """

        try:

            # Load from file
            if isinstance(file, Path):

                # Check file existence
                assert file.exists(), f"Missing file {file}"

                with file.open() as fid:
                    # Load yaml
                    document = YAML().load(fid)

            # Load as string
            else:
                document = YAML().load(file)

            # Use json to convert ordered dict to dict
            document = document if document else {}
            document = json.loads(json.dumps(document))

        except IOError as e:
            raise ConfigReadError(e)
        except YAMLError as e:
            raise ConfigReadError(e)

        return document


class OrigMeta(OneLayerConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = origmeta.PARAMETER_PATTERN_PROP
        self._class_key = ORIGMETA_KEY


class OneDimArrayConfigManager(OneLayerConfigManager):

    REQUIRED_ATTRIBUTES = {
        'document': list,
        'parameter_pattern_prop': dict,
        'node_params_def': dict,
        'class_key': str,
        'const_nodes': list
    }

    @abstractmethod
    def __init__(self, *args):
        super().__init__(*args)

        # Set attributes
        self._document = []
        self._const_nodes = []

    @property
    def const_nodes(self):
        """ """
        return self._const_nodes

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "type": 'array',
            "items": {
                "type": 'object',
                "patternProperties": self.parameter_pattern_prop,
                "additionalProperties": False
            },
        }

    def read(self, doc=None):
        """Read config

        Args:
            doc (:obj:`str`, optional): Default None -> read from directiory
                Else read from doc.

        """

        self.init_document()
        self._get_document(doc)

        # Append constant node
        self.append(self.const_nodes)

        # Add missing default node params
        for key, val in self.node_params_def.items():
            for i, _ in enumerate(self.document):
                if key not in self.document[i].keys():
                    self._document[i][key] = val


class InstrType(OneDimArrayConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = instrtype.PARAMETER_PATTERN_PROP
        self._const_nodes = []
        self._node_params_def = {}
        self._class_key = INSTR_TYPE_KEY


class Instrument(OneDimArrayConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = instrument.PARAMETER_PATTERN_PROP
        self._const_nodes = []
        self._node_params_def = instrument.NODE_PARAMS_DEF
        self._class_key = INSTR_KEY


class Parameter(OneDimArrayConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = parameter.PARAMETER_PATTERN_PROP
        self._const_nodes = []
        self._node_params_def = {}
        self._class_key = PARAM_KEY


class Flag(OneDimArrayConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = flag.PARAMETER_PATTERN_PROP
        self._const_nodes = flag.CONST_NODES
        self._node_params_def = {}
        self._class_key = FLAG_KEY


class MultiLayerConfigManager(OneLayerConfigManager):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = dict(
        {
            'node_pattern': list
        },
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
    )

    @abstractmethod
    def __init__(self, *args):
        super().__init__(*args)

        # Set attributes
        self._node_pattern = []

    def __getitem__(self, item):
        """Return config item value

        Args:
            item (list of str):

        Returns:

        """

        # Define nested function
        def find_val(nested_item):
            """

            Args:
                nested_item (list of str):

            Returns:

            """
            try:
                return get_by_path(self.document, nested_item)

            except (KeyError, TypeError, IndexError):
                if len(nested_item) <= 1:
                    raise KeyError
                return find_val(nested_item[:-2] + nested_item[-1:])

        # Get for item
        try:
            out = find_val(item)
        except KeyError:
            errmsg = (
                "Bad item {} in __getitem__({})".format(item, self.document)
            )
            raise ConfigItemKeyError(errmsg)

        return out

    def get_all(self, node_keys):
        """

        Args:
            node_keys (list of str):

        Returns:

        """

        # Check node_keys
        assert list(range(len(node_keys))) == [
            next(iter(
                i for i, pattern in enumerate(self.node_pattern)
                if re.fullmatch(pattern, arg)
            ))
            for arg in node_keys
        ], "Bad node_keys pattern or sequence"

        out = {
            key: self[
                [NODE_ESCAPE_CHAR + arg for arg in node_keys] + [key]
            ]
            for key in self.node_params_def.keys()
        }

        return out

    @property
    def node_pattern(self):
        """ """
        return self._node_pattern

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        node_pattern_prop_key = rf"^{NODE_ESCAPE_CHAR}(({')|('.join(self.node_pattern)}))$"
        node_pattern_prop = {
            node_pattern_prop_key: {"$ref": "#/nodeItem"}
        }

        pattern_prop = {
            **node_pattern_prop,
            **self.parameter_pattern_prop
        }

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "nodeItem": {
                "type": 'object',
                "patternProperties": pattern_prop,
                "additionalProperties": False
            },

            "type": 'object',
            "oneOf": [
                {"$ref": "#/nodeItem"}
            ]
        }

    def _get_document(self, doc_in=None):
        """Get YAML document as python dict

        Args:
            doc_in (str | pathlib.Path):

        """

        # Call parent method
        super()._get_document(doc_in)

        # Check node order
        self._check_dict_nodes(self.document, self.node_pattern)

    @staticmethod
    def _check_dict_nodes(document, node_pat):
        """Check document key node order

        Args:
            document:
            node_pat:

        Returns:

        """

        def check_single_node(doc, pat):
            """

            Args:
                doc:
                pat:

            Returns:

            """
            if type(doc) is dict:

                for key, sub_doc in doc.items():

                    # Skip if key doesn't begin by '_'
                    if re.match(rf'^[^{NODE_ESCAPE_CHAR}]', key):
                        continue

                    if not pat:
                        err_msg = "Bad node. No key to match in {}".format(doc)
                        raise ConfigNodeError(err_msg)

                    if re.fullmatch(rf"_{pat[0]}", key) is None:
                        err_msg = "Bad node. '{}' didn't match key in {}".format(pat[0], doc)
                        raise ConfigNodeError(err_msg)
                    else:
                        check_single_node(sub_doc, pat[1:])

            else:
                err_msg = "Bad node. '{}' isn't of type dict".format(doc)
                raise ConfigNodeError(err_msg)

        check_single_node(document, node_pat)


class OrigData(MultiLayerConfigManager):
    """ """

    def __init__(self, *args):
        super().__init__(*args)

        # Set required attributes
        self._parameter_pattern_prop = origdata.PARAMETER_PATTERN_PROP
        self._node_params_def = origdata.NODE_PARAMS_DEF
        self._node_pattern = origdata.NODE_PATTERN
        self._class_key = ORIGDATA_KEY


class ConfigInitError(Exception):
    """ """
    pass


class ConfigReadError(Exception):
    """ """
    pass


class ConfigNodeError(Exception):
    """ """
    pass


class IDNodeError(Exception):
    """ """
    pass


class ConfigItemKeyError(KeyError):
    """ """
    pass


class MaxConfigInstance(Exception):
    """ """
    pass
