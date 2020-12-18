"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: User configuration management.

"""

# Import python packages and modules
from abc import ABCMeta, abstractmethod
import re
import pprint
from functools import reduce
import operator
from pathlib import Path
import json
from jsonschema import validate, exceptions
from ruamel.yaml import YAML
from ruamel.yaml.error import YAMLError
from pampy.helpers import Union
import sre_yield

# Import current package modules
from .definitions import origdata, csvorigmeta
from .definitions import instrtype, instrument
from .definitions import parameter, flag
from .definitions import tag
from ..environ import path_var as env_path_var
from ..environ import glob_var as env_glob_var
from ..helper import get_by_path
from ..helper import RequiredAttrMetaClass
from ..helper import TypedProperty
from ..helper import camel_to_snake

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


class ConfigManager(metaclass=RequiredAttrMetaClass):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = {
        'DOC_TYPE': type,
    }

    #: type: Type of document. Choices: [dict, list].
    DOC_TYPE = None

    document = TypedProperty(Union[dict, list])
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

    def append(self, value):
        """Append value to document attribute

        Args:
            value (dict or list): Value to append

        """
        if self.DOC_TYPE is dict:
            self.document.update(value)
        else:
            self.document += value


class OneLayerConfigManager(ConfigManager):
    """Abstract class for managing 'one-layer' YAML config.

    'one-layer' means YAML file of such type::

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

       Must be a dict like::

         {"type": "object",
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

    def read(self, doc_in=None):
        """Read config from config files

        Args:
            doc_in (str, optional): Read input str instead of file. Defaults to None.

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

        cfg_file_suffix = ['.' + arg for arg in env_glob_var.config_file_ext]

        # Search yml config files
        if doc_in is None:

            # Init
            pat = re.compile(
                f'({self.CLASS_KEY})|({camel_to_snake(self.CLASS_KEY)})',
                flags=re.IGNORECASE
            )

            # Filter (case insensitive)
            doc_in = [
                arg for arg in env_path_var.config_dir_path.rglob("*.*")
                if (
                    (pat.search(arg.stem) is not None) and
                    (arg.suffix in cfg_file_suffix)
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
        """Read YAML document.

        Args:
            file (str | pathlib.Path): YAML document

        Returns:
            dict

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


class CSVOrigMeta(OneLayerConfigManager):
    """CSV original metadata config manager"""

    PARAMETER_PATTERN_PROP = csvorigmeta.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = csvorigmeta.KEY

    #: dict: Config document
    document = TypedProperty(OneLayerConfigManager.DOC_TYPE)


class OneDimArrayConfigManager(OneLayerConfigManager):
    """Abstract class for managing 'one-dim-array' YAML config.

    'one-dim-array' means YAML file of such type::

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
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Instrument(OneDimArrayConfigManager):
    """Instrument config manager"""

    PARAMETER_PATTERN_PROP = instrument.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = instrument.NODE_PARAMS_DEF
    CLASS_KEY = instrument.KEY
    CONST_NODES = instrument.CONST_NODES
    NODE_GEN = instrument.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Parameter(OneDimArrayConfigManager):
    """Parameter config manager """

    PARAMETER_PATTERN_PROP = parameter.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = parameter.KEY
    CONST_NODES = []
    NODE_GEN = parameter.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Flag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = flag.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = flag.KEY
    CONST_NODES = flag.CONST_NODES
    NODE_GEN = flag.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Tag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = tag.PARAMETER_PATTERN_PROP
    NODE_PARAMS_DEF = {}
    CLASS_KEY = tag.KEY
    CONST_NODES = tag.CONST_NODES
    NODE_GEN = tag.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class MultiLayerConfigManager(OneLayerConfigManager):
    """Abstract class for managing YAML config"""

    REQUIRED_ATTRIBUTES = dict(
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'NODE_PATTERN': list,
            'CONST_NODES': dict,
        },
    )

    # Define required attributes
    DOC_TYPE = dict

    #: list: Node pattern (order matter)
    NODE_PATTERN = None

    #: dict: Constant node value
    CONST_NODES = None

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

    def get_val(self, node_keys, key_param):
        """Return single node_keys value

        Args:
            node_keys (list of str): Node keys
            key_param (str): Key parameter

        Returns
            `object`

        Raises:


        """

        try:
            out = self[
                [NODE_ESCAPE_CHAR + arg for arg in node_keys] + [key_param]
            ]
        except ConfigItemKeyError as _:
            raise KeyError(
                f"Bad key '{node_keys + [key_param]}' " +
                f"for {self.__class__.__name__}"
            )

        return out

    def get_all_default(self, node_keys):
        """Return all default values

        Args:
            node_keys (list of str): Node keys

        Returns:
            dict

        """

        out = {
            key: self.get_val(node_keys, key)
            for key in [*self.NODE_PARAMS_DEF.keys(), *self.CONST_NODES.keys()]
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

    def read(self, doc_in=None):
        """Overwrite read method"""

        # Call super method
        super().read(doc_in=doc_in)

        # Append constant node
        self.document.update(self.CONST_NODES)

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
    CONST_NODES = origdata.CONST_NODES

    #: dict: Config document
    document = TypedProperty(MultiLayerConfigManager.DOC_TYPE)


class OneDimArrayConfigLinker:
    """Link to OneDimArrayConfigManager
    config managers."""

    #: list: Default instantiated config managers
    CFG_MNGRS_DFLT = [Parameter, InstrType, Instrument, Flag, Tag]

    def __init__(self, cfg_mngrs=None):
        """
        Args:
            cfg_mngrs (list of OneDimArrayConfigManager): Config managers
        """

        if cfg_mngrs is None:
            cfg_mngrs = self.CFG_MNGRS_DFLT

        # Set attributes
        self._cfg_mngr = instantiate_config_managers(*cfg_mngrs)

    def get_document(self, key):
        """Interpret the generator syntax if necessary and
        return the config document.

        The generator syntax is apply only in OneDimArrayConfig with
        not empty NODE_GEN. Generator syntax is based on regexpr. Fields which
        are not generator can contain expression to be evaluated. The syntax
        used is the one described in ConfigExprInterpreter.eval

        Args:
            key (str): Config manager key

        Returns:
            dict

        Raises:
            - ConfigGenMaxLenError: Error for to much generated items.

        """

        # Define
        array_old = self._cfg_mngr[key].document
        node_gen = self._cfg_mngr[key].NODE_GEN
        array_new = []

        # Loop over te config array items
        for doc in array_old:

            # Test if node generator allowed
            if node_gen:

                # Init new sub dict
                sub_dict_new = {}

                # Generate from regexp generator
                node_gen_val = sre_yield.AllMatches(doc[node_gen])

                # Check length
                if (n_val := len(node_gen_val)) > env_glob_var.config_gen_max:
                    raise ConfigGenMaxLenError(
                        f"{n_val} generated config field. " +
                        f"Max allowed {env_glob_var.config_gen_max}"
                    )

                # Update sub dict
                sub_dict_new.update({node_gen: list(node_gen_val)})

                # Loop over other config item key
                for key in filter(lambda x: x != node_gen, doc.keys()):

                    # Update new sub dict for current key
                    sub_dict_new.update(
                        {
                            key: [
                                ConfigExprInterpreter.eval(
                                    doc[key], node_gen_val[i].group
                                )
                                for i in range(len(node_gen_val))
                            ]
                        }
                    )

                # Rearange dict of list in list of dict
                res = [
                    dict(zip(sub_dict_new, arg))
                    for arg in zip(*sub_dict_new.values())
                ]

            # Case without generator
            else:
                res = [doc]

            # Append to new array
            array_new += res

        return array_new


class ConfigExprInterpreter(metaclass=ABCMeta):
    """Abstract config expression interpreter class

        Notes:
            This class and subclasses construction are based on the interpreter
            design pattern.

        """

    _FCT = str

    @classmethod
    def set_callable(cls, fct):
        """Set strategy
        Args:
            fct (callable): Function/Methode called by 'get' expression
        """

        # Test
        assert callable(fct), "'fct' must be a callable"

        cls._FCT = fct

    @abstractmethod
    def interpret(self):
        """Interpreter method"""

    @staticmethod
    def eval(expr, get_fct):
        """Evaluate str expression

        Args:
            expr (str): Expression to evaluate
            get_fct (callable): Function use by 'get'

        Examples:
            >>> import re
            >>> mymatch = re.match('^a(\d)', 'a1b')
            >>> print(ConfigExprInterpreter.eval("cat('My test', ' ', get(1))", mymatch.group))
            My test 1
        """

        # Define
        str_expr_dict = {
            'cat': CatExpr,
            'rpl': ReplExpr, 'repl': ReplExpr,
            'rpls': ReplStrictExpr, 'repl_strict': ReplStrictExpr,
            'get': GetExpr,
            'upper': UpperExpr, 'lower': LowerExpr,
            'supper': SmallUpperExpr, 'small_upper': SmallUpperExpr,
        }

        # Set get_value
        ConfigExprInterpreter.set_callable(get_fct)

        # Treat expression
        try:
            # Eval
            expr_out = eval(expr, str_expr_dict)

            # Interpret
            expr_out = expr_out.interpret()

        # TODO
        #  Detail exception
        except Exception:
            expr_out = expr

        return expr_out


class NonTerminalConfigExprInterpreter(ConfigExprInterpreter):
    """Implement an interpreter operation for non terminal symbols in the
    grammar.
    """

    def __init__(self, *args):
        self._expression = args

    def interpret(self):
        """Non terminal interpreter method"""

        # Apply interpreter
        res_interp = [
            (arg if isinstance(arg, ConfigExprInterpreter)
             else NoneExpr(arg)).interpret()
            for arg in self._expression
        ]

        if len(self._expression) > 1:
            return reduce(self.fct, res_interp)
        else:
            return self.fct(res_interp[0])

    @abstractmethod
    def fct(self, *args):
        """Function between expression args"""


class CatExpr(NonTerminalConfigExprInterpreter):
    """String concatenation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.add(a, b)


class ReplExpr(NonTerminalConfigExprInterpreter):
    """Replace dict key by its value. If key is missing, return key"""

    def fct(self, a, b):
        """Implement fct method"""
        try:
            out = operator.getitem(a, b)
        except KeyError:
            out = b
        return out


class ReplStrictExpr(NonTerminalConfigExprInterpreter):
    """Replace dict key by its value. If key is missing, return ''"""

    def fct(self, a, b):
        """Implement fct method"""
        try:
            out = operator.getitem(a, b)
        except KeyError:
            out = ''
        return out


class LowerExpr(NonTerminalConfigExprInterpreter):
    """Lower case"""

    def fct(self, a):
        """Implement fct method"""
        return a.lower()


class UpperExpr(NonTerminalConfigExprInterpreter):
    """Upper case"""

    def fct(self, a):
        """Implement fct method"""
        return a.upper()


class SmallUpperExpr(NonTerminalConfigExprInterpreter):
    """Upper case 1st character"""

    def fct(self, a):
        """Implement fct method"""
        return a[0].upper() + a[1:].lower()


class TerminalConfigExprInterpreter(ConfigExprInterpreter):
    """Implement an interpreter operation for terminal symbols in the
    grammar.
    """

    def __init__(self, arg):
        self._expression = arg


class GetExpr(TerminalConfigExprInterpreter):
    """Get catch value"""

    def interpret(self):
        """Implement fct method"""
        return self._FCT(self._expression)


class NoneExpr(TerminalConfigExprInterpreter):
    """Apply none interpreter"""

    def interpret(self):
        """Implement fct method"""
        return self._expression


class ConfigReadError(Exception):
    """Error while reading config"""


class ConfigNodeError(Exception):
    """Error in config node"""


class ConfigItemKeyError(KeyError):
    """Error in config key item"""


class ConfigGenMaxLenError(Exception):
    """Exception class for max length config generator error"""
