"""
Module containing class and function for config management.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import python packages and modules
import re
import pprint
from abc import ABC
from pathlib import Path
import json
from jsonschema import validate, exceptions
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError


# Import current package modules
from .definitions import origdata, origmeta
from .definitions import instrtype, instrument
from .definitions import parameter, flag
from .definitions import tag
from ..dvas_environ import path_var as env_path_var
from ..dvas_helper import get_by_path
from ..dvas_helper import RequiredAttrMetaClass
from ..dvas_helper import TypedProperty
from ..dvas_helper import camel_to_snake


# Define
NODE_ESCAPE_CHAR = '_'


def instantiate_config_managers(*args, read=True):
    """Generate a dictionary with instances of all ConfigManagers

    Args:
        args (ConfigManager):
            ConfigManager to instantiate
        read (bool, optional): Read config during class instantiation.
            Default to True.

    Returns:
        dict: key are ConfigManager name, value are ConfigManager instances

    """

    # Create instances
    instances = []

    for config_manager in args:
        instances.append(config_manager())

    # Read
    if read:
        for inst in instances:
            inst.read()

    return {arg.CLASS_KEY: arg for arg in instances}


class ConfigManager(ABC, metaclass=RequiredAttrMetaClass):
    """Abstract clas for managing YAML config"""

    REQUIRED_ATTRIBUTES = {
        'DOC_TYPE': type,
    }

    #: type: Type of document. Only dict or list types.
    DOC_TYPE = None

    document = TypedProperty((dict, list))
    """dict: Config document. Must be redefined as well to avoid 
    list/dict reference overlap     
    """

    def __init__(self):
        """Constructor"""
        self.init_document()

    def __getitem__(self, item):
        """Overwrite __getitem__ method"""
        return self.document[item]

    def __repr__(self):
        """Overwrite __repr__ method"""
        p_printer = pprint.PrettyPrinter()
        return p_printer.pformat(self.document)

    def init_document(self):
        """Initialise document attribute"""
        if self.DOC_TYPE is dict:
            self.document = {}
        else:
            self.document = []


class OneLayerConfigManager(ConfigManager):
    """Abstract class for managing 'one-layer' YAML config

    'one-layer' means YAML file of such type:
        tag1: value_tag1
        tag2: value_tag2
        ...
        tagN: value_tagN

    """

    REQUIRED_ATTRIBUTES = dict(
        **ConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'PARAMETER_PATTERN_PROP': dict,
            'NODE_PARAMS_DEF': dict,
            'CLASS_KEY': str
        }
    )

    # Define required attributes
    DOC_TYPE = dict

    PARAMETER_PATTERN_PROP = None
    """dict: JSON parameter schema. Constant value.
            Must be a dict like {
                "type": "object",
                "patternProperties": ANY,
                "additionalProperties": False
            }
    """
    #: dict: Default root node parameters
    NODE_PARAMS_DEF = None
    #: str: Class key denomination
    CLASS_KEY = None

    @property
    def json_schema(self):
        """dict: JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "type": 'object',
            "patternProperties": self.PARAMETER_PATTERN_PROP,
            "additionalProperties": False
        }

    def append(self, value):
        """Append value to document attribute

        Args:
            value (dict or list): Value to append

        """
        if self.DOC_TYPE is dict:
            self.document.update(value)
        else:
            self.document += value

    def read(self, doc_in=None):
        """Read config

        Args:
            doc_in (:obj:`str`, optional): Default None -> read from directory
                Else read from doc.

        """

        self.init_document()
        self._get_document(doc_in)

        # Add missing default node params
        for key, val in self.NODE_PARAMS_DEF.items():
            if key not in self.document.keys():
                self.document[key] = val

    def _get_document(self, doc_in=None):
        """Get YAML document as python dict

        Args:
            doc_in (str | pathlib.Path):

        """

        # Search yml config files
        if doc_in is None:

            # Init
            pat = re.compile(
                f'({self.CLASS_KEY})|({camel_to_snake(self.CLASS_KEY)})',
                flags=re.IGNORECASE
            )

            # Filter (case insensitive)

            #TODO
            # Add allowed config files extensions in globals
            doc_in = [
                arg for arg in env_path_var.config_dir_path.rglob("*.*")
                if (
                    (pat.search(arg.stem) is not None) and
                    (arg.suffix in ['.yml', '.yaml'])
                )
            ]

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

        except exceptions.ValidationError as exc:
            raise ConfigReadError(
                f"Error in '{list(doc_in)}'\n{exc}"
            )

        except exceptions.SchemaError as exc:
            raise ConfigReadError(
                f"Error in {self.__class__.__name__}.json_schema\n{exc}"
            )

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

        except IOError as exc:
            raise ConfigReadError(exc)
        except YAMLError as exc:
            raise ConfigReadError(exc)

        return document


class OrigMeta(OneLayerConfigManager):
    """Original metadata config manager"""

    PARAMETER_PATTERN_PROP = origmeta.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = origmeta.KEY

    #: dict: Config document
    document = TypedProperty((dict, list))


class OneDimArrayConfigManager(OneLayerConfigManager):
    """Abstract class for managing 'one-dim-array' YAML config

    'one-dim-array' means YAML file of such type:
        - tag11: value_tag11
          tag12: value_tag12
          ...
          tag1N: value_tag1N
        - tag21: value_tag21
          tag22: value_tag22
          ...
          tag2N: value_tag2N
        ...
        - tagM1: value_tagM1
          tagM2: value_tagM2
          ...
          tagMN: value_tagMN

    """
    REQUIRED_ATTRIBUTES = dict(
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'CONST_NODES': list,
            'NODE_GEN': str,
        }
    )

    # Define required attributes
    DOC_TYPE = list

    #: list: Constant node value
    CONST_NODES = None

    NODE_GEN = None
    """str: Node name able to be generated by regexp. Use empty str to assign
    none node.
    """

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        return {
            "$schema": "http://json-schema.org/draft-07/schema#",

            "type": 'array',
            "items": {
                "type": 'object',
                "patternProperties": self.PARAMETER_PATTERN_PROP,
                "additionalProperties": False
            },
        }

    def read(self, doc_in=None):
        """Read config

        Args:
            doc_in (:obj:`str`, optional): Default None -> read from directiory
                Else read from doc_in.

        """

        self.init_document()
        self._get_document(doc_in)

        # Append constant node
        self.append(self.CONST_NODES)

        # Add missing default node params
        for key, val in self.NODE_PARAMS_DEF.items():
            for i, _ in enumerate(self.document):
                if key not in self.document[i].keys():
                    self.document[i][key] = val


class InstrType(OneDimArrayConfigManager):
    """Instrument type config manager"""

    PARAMETER_PATTERN_PROP = instrtype.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = instrtype.KEY
    CONST_NODES = instrtype.CONST_NODES
    NODE_GEN = instrtype.NODE_GEN

    #: dict: Config document
    document = TypedProperty((dict, list))


class Instrument(OneDimArrayConfigManager):
    """Instrument config manager"""

    PARAMETER_PATTERN_PROP = instrument.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = instrument.NODE_PARAMS_DEF
    CLASS_KEY = instrument.KEY
    CONST_NODES = instrument.CONST_NODES
    NODE_GEN = instrument.NODE_GEN

    #: dict: Config document
    document = TypedProperty((dict, list))


class Parameter(OneDimArrayConfigManager):
    """Parameter config manager """

    PARAMETER_PATTERN_PROP = parameter.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = parameter.KEY
    CONST_NODES = []
    NODE_GEN = parameter.NODE_GEN

    #: dict: Config document
    document = TypedProperty((dict, list))


class Flag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = flag.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = flag.KEY
    CONST_NODES = flag.CONST_NODES
    NODE_GEN = flag.NODE_GEN

    #: dict: Config document
    document = TypedProperty((dict, list))


class Tag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = tag.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = tag.KEY
    CONST_NODES = []
    NODE_GEN = tag.NODE_GEN

    #: dict: Config document
    document = TypedProperty((dict, list))


class MultiLayerConfigManager(OneLayerConfigManager):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = dict(
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
        **{'NODE_PATTERN': list},
    )

    # Define required attributes
    DOC_TYPE = dict

    #: list: Node pattern (order matter)
    NODE_PATTERN = None

    def __getitem__(self, item):
        """Overwrite __getitem method.

        Args:
            item (list of str): Document item as list

        Returns:
            object

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
                i for i, pattern in enumerate(self.NODE_PATTERN)
                if re.fullmatch(pattern, arg)
            ))
            for arg in node_keys
        ], "Bad node_keys pattern or sequence"

        out = {
            key: self[
                [NODE_ESCAPE_CHAR + arg for arg in node_keys] + [key]
            ]
            for key in self.NODE_PARAMS_DEF.keys()
        }

        return out

    @property
    def json_schema(self):
        """JSON schema. Constant value"""

        node_pattern_prop_key = (
            rf"^{NODE_ESCAPE_CHAR}(({')|('.join(self.NODE_PATTERN)}))$"
        )
        node_pattern_prop = {
            node_pattern_prop_key: {"$ref": "#/nodeItem"}
        }

        pattern_prop = {
            **node_pattern_prop,
            **self.PARAMETER_PATTERN_PROP
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
        self._check_dict_nodes(self.document, self.NODE_PATTERN)

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
            if isinstance(doc, dict):

                for key, sub_doc in doc.items():

                    # Skip if key doesn't begin by '_'
                    if re.match(rf'^[^{NODE_ESCAPE_CHAR}]', key):
                        continue

                    if not pat:
                        err_msg = f"Bad node. No key to match in {doc}"
                        raise ConfigNodeError(err_msg)

                    if re.fullmatch(rf"_{pat[0]}", key) is None:
                        err_msg = (
                            f"Bad node. '{pat[0]}' didn't match key in {doc}"
                        )
                        raise ConfigNodeError(err_msg)

                    check_single_node(sub_doc, pat[1:])

            else:
                err_msg = "Bad node. '{}' isn't of type dict".format(doc)
                raise ConfigNodeError(err_msg)

        check_single_node(document, node_pat)


class OrigData(MultiLayerConfigManager):
    """Original data config manager"""

    PARAMETER_PATTERN_PROP = origdata.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = origdata.NODE_PARAMS_DEF
    CLASS_KEY = origdata.KEY
    NODE_PATTERN = origdata.NODE_PATTERN

    #: dict: Config document
    document = TypedProperty((dict, list))


class ConfigReadError(Exception):
    """Error while reading config"""


class ConfigNodeError(Exception):
    """Error in config node"""


class ConfigItemKeyError(KeyError):
    """Error in config key item"""
