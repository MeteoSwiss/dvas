"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

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
from .definitions import model
from .definitions import parameter, flag
from .definitions import tag
from ..environ import path_var
from ..environ import glob_var as env_glob_var
from ..helper import get_by_path
from ..helper import RequiredAttrMetaClass
from ..helper import TypedProperty
from ..helper import camel_to_snake
from ..database.model import Parameter as TableParameter
from ..hardcoded import FLAG_PRM_NAME_SUFFIX, FLAG_PRM_DESC_PREFIX
from ..errors import ConfigError
from ..errors import ConfigPathError, ConfigReadYAMLError, ConfigCheckJSONError
from ..errors import ConfigReadError, ConfigNodeError
from ..errors import ConfigGetError, ConfigLabelNameError
from ..errors import ConfigGenMaxLenError
from ..errors import ExprInterpreterError, NonTerminalExprInterpreterError, TerminalExprInterpreterError


# Define
NODE_ESCAPE_CHAR = '_'


def instantiate_config_managers(*args, read=True):
    """Generate a dictionary with instances of all specified ConfigManagers

    Args:
        args (ConfigManager):
            ConfigManager to instantiate
        read (bool, optional): Read config automatically after instantiation.
            Default to True.

    Returns:
        dict: instances of ConfigManager

    """

    # Create instances
    instances = []
    for config_manager in args:
        instances.append(config_manager())

    # Read
    if read:
        for inst in instances:
            try:
                inst.read()
            except ConfigError as exc:
                raise ConfigError(f"Error in reading instance of '{inst.CLASS_KEY}'") from exc

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
        self.init_document()

    def __getitem__(self, item):
        try:
            out = self.document[item]

        except (KeyError, TypeError) as exc:
            raise ConfigGetError from exc

        return out

    def __repr__(self):
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

    @abstractmethod
    def read(self, doc_in):
        """Abstract read method"""


class OneLayerConfigManager(ConfigManager):
    """Abstract class for managing 'one-layer' YAML config.

    'one-layer' means YAML file of such type::

        label1: value_label1
        label2: value_label2
        ...
        labelN: value_labelN

    """

    REQUIRED_ATTRIBUTES = dict(
        **ConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'PARAMETER_PATTERN_PROP': dict,
            'LABEL_VAL_DEF': dict,
            'CLASS_KEY': str
        }
    )

    # Define required attributes
    DOC_TYPE = dict

    PARAMETER_PATTERN_PROP = None
    """dict: Parameter JSON 7 schema. Constant value.

       Must be a dict like::

         {"type": "object",
          "patternProperties": ANY,
          "additionalProperties": False
         }

    """
    #: dict: Default values of labels
    LABEL_VAL_DEF = None
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
            doc_in (:obj:`str`, optional): Default None -> read from directory
                Else read from doc_in.

        Raises:
            ConfigReadError: Error in reading YAML config data.

        """

        # Init
        self.init_document()

        try:
            # Get
            self._get_document(doc_in)

        except ConfigReadError as exc:
            raise ConfigReadError('Error in user config') from exc

        # Add default values of missing labels
        for key, val in self.LABEL_VAL_DEF.items():
            if key not in self.document.keys():
                self.document[key] = val

        # Validate hard coded config
        try:
            self._validate_json(self.document)

        except ConfigReadError as exc:
            raise ConfigReadError('Error in hard coded config') from exc


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

            # Test
            if (
                (path_var.config_dir_path is None) or
                (path_var.config_dir_path.exists() is False)
            ):
                raise ConfigPathError(f'Missing config dir path::{path_var.config_dir_path}')

            # Filter (case insensitive)
            doc_in = [
                arg for arg in path_var.config_dir_path.rglob("*.*")
                if (
                    (pat.search(arg.stem) is not None) and
                    (arg.suffix in cfg_file_suffix)
                )
            ]

        # Convert YAML string as JSON dict
        if isinstance(doc_in, str):
            try:
                self.append(
                    self.read_yaml(doc_in)
                )
            except ConfigReadError as exc:
                raise ConfigReadError("Error in reading config 'str'") from exc

        # Convert YAML file as JSON dict
        else:
            for filepath in doc_in:
                try:
                    self.append(
                        self.read_yaml(Path(filepath))
                    )
                except ConfigReadError as exc:
                    raise ConfigReadError(f"Error in reading config file '{filepath}'") from exc

    def read_yaml(self, yaml_doc):
        """Read YAML document.

        Args:
            yaml_doc (str | pathlib.Path): YAML document

        Returns:
            dict

        """

        try:

            # Load from file
            if isinstance(yaml_doc, Path):

                # Check file existence
                assert yaml_doc.exists(), f"Missing file {yaml_doc}"

                # Load yaml
                with yaml_doc.open() as fid:
                    document = YAML().load(fid)

            # Load as string
            else:
                document = YAML().load(yaml_doc)

            # Use json to convert ordered dict to dict
            document = document if document else {}
            document = json.loads(json.dumps(document))

        except (AssertionError, IOError, YAMLError) as exc:
            raise ConfigReadYAMLError("Error in reading YAML document") from exc

        # Validate JSON
        self._validate_json(document)

        return document

    def _validate_json(self, document):
        """Validate JSON document

        Args:
            document (dict): JSON document

        Raises:
            ConfigCheckJSONError: Error in validation or JSON schema.

        """

        # Check json schema validity
        try:
            validate(instance=document, schema=self.json_schema)

        except exceptions.ValidationError as exc:
            p_printer = pprint.PrettyPrinter()
            raise ConfigCheckJSONError(f"JSON validation error::\n{p_printer.pformat(document)}") from exc

        except exceptions.SchemaError as exc:
            p_printer = pprint.PrettyPrinter()
            raise ConfigCheckJSONError(f"JSON schema error::\n{p_printer.pformat(self.json_schema)}") from exc


class CSVOrigMeta(OneLayerConfigManager):
    """CSV original metadata config manager"""

    PARAMETER_PATTERN_PROP = csvorigmeta.PARAMETER_PATTERN_PROP
    LABEL_VAL_DEF = {}
    CLASS_KEY = csvorigmeta.KEY

    #: dict: Config document
    document = TypedProperty(OneLayerConfigManager.DOC_TYPE)


class OneDimArrayConfigManager(OneLayerConfigManager):
    """Abstract class for managing 'one-dim-array' YAML config.

    'one-dim-array' means YAML file of such type::

        - label11: value_label11
          label12: value_label12
          ...
          label1N: value_label1N

        - label21: value_label21
          label22: value_label22
          ...
          label2N: value_label2N
        ...
        - labelM1: value_labelM1
          labelM2: value_labelM2
          ...
          labelMN: value_labelMN

    """
    REQUIRED_ATTRIBUTES = dict(
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'CONST_LABELS': list,
            'NODE_GEN': str,
        }
    )

    # Define required attributes
    DOC_TYPE = list

    #: list: Constant labels
    CONST_LABELS = None

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
            doc_in (:obj:`str`, optional): Default None -> read from directory
                Else read from doc_in.
        Raises:
            ConfigReadError: Error in reading YAML config data.

        """

        # Init
        self.init_document()

        try:
            # Get
            self._get_document(doc_in)

        except ConfigReadError as exc:
            raise ConfigReadError('Error in user config') from exc

        # Append constant labels
        self.append(self.CONST_LABELS)

        # Add default values of missing labels
        for key, val in self.LABEL_VAL_DEF.items():
            for i, _ in enumerate(self.document):
                if key not in self.document[i].keys():
                    self.document[i][key] = val

        # Validate hard coded config
        try:
            self._validate_json(self.document)

        except ConfigReadError as exc:
            raise ConfigReadError('Error in hard coded config') from exc

    def _get_document(self, doc_in=None):
        """Override method"""

        # Call super method
        super()._get_document(doc_in=doc_in)

        # Generate automatic labels
        if self.NODE_GEN:

            document_new = []
            for doc in self.document:

                # Init new sub dict
                sub_dict_new = {}

                # Generate from regexp generator
                node_gen_val = sre_yield.AllMatches(doc[self.NODE_GEN])

                # Check length
                if (n_val := len(node_gen_val)) > env_glob_var.config_gen_max:
                    raise ConfigGenMaxLenError(
                        f"Generator {doc[self.NODE_GEN]} will generate {n_val} config field. " +
                        f"Max allowed {env_glob_var.config_gen_max}"
                    )

                # Update sub dict
                sub_dict_new.update({self.NODE_GEN: list(node_gen_val)})

                # Loop over other config item key
                for doc_key in filter(lambda x: x != self.NODE_GEN, doc.keys()):
                    # Update new sub dict for current key
                    sub_dict_new.update(
                        {
                            doc_key: [
                                ConfigExprInterpreter.eval(
                                    doc[doc_key], node_gen_val[i].group
                                )
                                for i in range(len(node_gen_val))
                            ]
                        }
                    )

                # Rearange dict of list in list of dict
                document_new += [
                    dict(zip(sub_dict_new, arg))
                    for arg in zip(*sub_dict_new.values())
                ]

            # Copy now doc
            self.document = document_new.copy()



class Model(OneDimArrayConfigManager):
    """Instrument type config manager"""

    PARAMETER_PATTERN_PROP = model.PARAMETER_PATTERN_PROP
    LABEL_VAL_DEF = model.LABEL_VAL_DEF
    CLASS_KEY = model.KEY
    CONST_LABELS = model.CONST_LABELS
    NODE_GEN = model.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Parameter(OneDimArrayConfigManager):
    """Parameter config manager """

    PARAMETER_PATTERN_PROP = parameter.PARAMETER_PATTERN_PROP
    LABEL_VAL_DEF = parameter.LABEL_VAL_DEF
    CLASS_KEY = parameter.KEY
    CONST_LABELS = []
    NODE_GEN = parameter.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)

    def _get_document(self, doc_in=None):
        """Override method"""

        # Call super method
        super()._get_document(doc_in=doc_in)

        # Duplicate parameters into there flag item
        # Remark: It's not necessarily the most elegant way to duplicate parameters to get the flag side...

        # Define mapping
        arg_key_to_dict = {
            TableParameter.prm_name.name: lambda x: f"{x}{FLAG_PRM_NAME_SUFFIX}",
            TableParameter.prm_desc.name: lambda x: f"{FLAG_PRM_DESC_PREFIX}{x[0].lower()}{x[1:]}",
            TableParameter.prm_unit.name: lambda _: '',
        }

        # Create duplicate array of flags
        array_prm_flg = [
            {
                arg_key: arg_key_to_dict[arg_key](arg_val) for arg_key, arg_val in arg.items()
            }
            for arg in self.document
        ]

        # Append
        self.document += array_prm_flg


class Flag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = flag.PARAMETER_PATTERN_PROP
    LABEL_VAL_DEF = {}
    CLASS_KEY = flag.KEY
    CONST_LABELS = flag.CONST_LABELS
    NODE_GEN = flag.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class Tag(OneDimArrayConfigManager):
    """Flag config manager """

    PARAMETER_PATTERN_PROP = tag.PARAMETER_PATTERN_PROP
    LABEL_VAL_DEF = {}
    CLASS_KEY = tag.KEY
    CONST_LABELS = tag.CONST_LABELS
    NODE_GEN = tag.NODE_GEN

    #: dict: Config document
    document = TypedProperty(OneDimArrayConfigManager.DOC_TYPE)


class MultiLayerConfigManager(OneLayerConfigManager):
    """Abstract class for managing 'multi-layer' YAML config."""

    REQUIRED_ATTRIBUTES = dict(
        **OneLayerConfigManager.REQUIRED_ATTRIBUTES,
        **{
            'NODE_PATTERN': list,
        },
    )

    # Define required attributes
    DOC_TYPE = dict

    #: list: Node pattern (order matter)
    NODE_PATTERN = None

    def __getitem__(self, item):
        """Override __getitem__ method.

        Args:
            item (list of str): Document item as list

        Returns:
            object

        Raises:
            ConfigGetError: Error in getting config label value

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

            except (KeyError, TypeError, IndexError) as exc:
                if len(nested_item) <= 1:
                    raise ConfigLabelNameError from exc
                return find_val(nested_item[:-2] + nested_item[-1:])

        # Get for item
        try:
            out = find_val(item)
        except ConfigLabelNameError as exc:
            errmsg = (
                f"Can't find item {item} in\n{self}"
            )
            raise ConfigGetError(errmsg) from exc

        return out

    def get_val(self, node_labels, final_label):
        """Return single node_labels value

        Args:
            node_labels (list of str): Node keys. If the escape character is missing in the prefix of the node,
                it is added automatically.
            final_label (str): Key parameter

        Returns
            `object`

        Raises:
            ConfigGetError: Error in getting config label value

        """

        # Convert node to list
        if isinstance(node_labels, str):
            node_labels = [node_labels]

        # Add node escape char if necessary
        node_labels_mod = [
            (arg if arg.startswith(NODE_ESCAPE_CHAR) else NODE_ESCAPE_CHAR + arg)
            for arg in node_labels
        ]

        try:
            out = self[node_labels_mod + [final_label]]

        except ConfigGetError as exc:
            raise ConfigGetError(
                f"Can't find '{node_labels + [final_label]}' in config"
            ) from exc

        return out

    def get_all(self, node_labels):
        """Return all values for a given node labels. Only values specified in defaults labels will be returned.

        Args:
            node_labels (list of str): Node keys

        Returns:
            dict

        Raises:
            ConfigGetError: Error in getting config label value

        """

        out = {
            key: self.get_val(node_labels, key)
            for key in [*self.LABEL_VAL_DEF.keys()]
        }

        return out

    def get_default(self):
        """Return all default values"""
        return self.get_all([])

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
        """Check document labels - nodes - order.

        Args:
            document (dict): JSON document
            node_pat (list): Node order pattern

        Raises:
            ConfigNodeError: Error in node order

        """

        def check_single_node(doc, pat):
            """Check single document labels order

            Args:
                doc (dict): JSON docuument
                pat (list): Node order pattern

            Raises:
                ConfigNodeError: Error in node order

            """
            if isinstance(doc, dict):

                for key, sub_doc in doc.items():

                    # Skip if key doesn't begin by '_'
                    if re.match(rf'^[^{NODE_ESCAPE_CHAR}]', key):
                        continue

                    if not pat:
                        pprinter = pprint.PrettyPrinter()
                        err_msg = f"Bad node label.\nNo matching keys in\n{pprinter.pformat(doc)}"
                        raise ConfigNodeError(err_msg)

                    if re.fullmatch(rf"{NODE_ESCAPE_CHAR}{pat[0]}", key) is None:
                        pprinter = pprint.PrettyPrinter()
                        err_msg = (
                            f"Bad node label.\n'{pprinter.pformat(pat[0])}' didn't match any keys in\n{pprinter.pformat(doc)}"
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
    LABEL_VAL_DEF = origdata.LABEL_VAL_DEF
    CLASS_KEY = origdata.KEY
    NODE_PATTERN = origdata.NODE_PATTERN

    #: dict: Config document
    document = TypedProperty(MultiLayerConfigManager.DOC_TYPE)


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
        """Interprete expression.

        Args:
            expr (str|ConfigExprInterpreter): Expression to evaluate.
            get_fct (callable): Function use by 'get'

        Syntax:
            cat(<str_1>, ..., <str_n>): Concatenate str_1 to str_n


        Raises:
            ExprInterpreterError: Error while interpreting expression

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
            if isinstance(expr, str):
                expr_out = eval(expr, str_expr_dict)
            else:
                expr_out = expr

            # Interpret
            expr_out = expr_out.interpret()

        except (NameError, SyntaxError, AttributeError) as exc:
            expr_out = expr

        except (NonTerminalExprInterpreterError, TerminalExprInterpreterError) as exc:
            raise ExprInterpreterError(f"Error in '{expr}'") from exc

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
        try:
            out = operator.add(a, b)
        except TypeError as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out


class ReplExpr(NonTerminalConfigExprInterpreter):
    """Replace dict key or list index by its value. If key is missing, return key"""

    def fct(self, a, b):
        """Implement fct method"""
        try:
            out = operator.getitem(a, b)
        except (KeyError, IndexError):
            out = b
        except TypeError as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out


class ReplStrictExpr(NonTerminalConfigExprInterpreter):
    """Replace dict key or list index by its value. If key is missing, return ''"""

    def fct(self, a, b):
        """Implement fct method"""
        try:
            out = operator.getitem(a, b)
        except (KeyError, IndexError):
            out = ''
        except TypeError as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out


class LowerExpr(NonTerminalConfigExprInterpreter):
    """Lower case"""

    def fct(self, a):
        """Implement fct method"""
        try:
            out = a.lower()
        except AttributeError as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out


class UpperExpr(NonTerminalConfigExprInterpreter):
    """Upper case"""

    def fct(self, a):
        """Implement fct method"""
        try:
            out = a.upper()
        except AttributeError as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out


class SmallUpperExpr(NonTerminalConfigExprInterpreter):
    """Upper case 1st character"""

    def fct(self, a):
        """Implement fct method"""
        try:
            out = a[0].upper() + a[1:].lower()
        except (AttributeError, TypeError) as exc:
            raise NonTerminalExprInterpreterError() from exc

        return out

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
        try:
            out = self._FCT(self._expression)
        except (IndexError, AttributeError) as exc:
            raise TerminalExprInterpreterError() from exc

        return out


class NoneExpr(TerminalConfigExprInterpreter):
    """Apply none interpreter"""

    def interpret(self):
        """Implement fct method"""
        return self._expression
