"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database exploring tools

"""

# Import from python packages
from cmd import Cmd
from pprint import pprint
from abc import abstractmethod, ABCMeta
import operator
from functools import reduce
from datetime import datetime
from pandas import Timestamp
from pampy.helpers import Iterable, Union

# Import from current package
from .database import DatabaseManager
from .model import Parameter, Info
from .model import Object as TableObject
from .model import Info as TableInfo
from .model import Parameter as TableParameter
from .model import Tag as TableTag
from .model import InfosObjects as TableInfosObjects
from .model import InfosTags
from ..hardcoded import TAG_EMPTY_NAME
from ..hardcoded import TAG_RAW_NAME, TAG_GDP_NAME
from ..helper import TypedProperty as TProp
from ..helper import check_datetime


class DatabasePrompt(Cmd):
    prompt = 'db> '
    intro = "Welcome! Type '?' to list commands"

    _EXIT_CMD = ['x', 'q', 'exit', 'quit']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Init attributes
        self._db_mngr = DatabaseManager()
        self._db_reader = ReadDatabase()

    def default(self, inp):
        if inp in self._EXIT_CMD:
            return self.do_exit(inp)

    def do_exit(self, inp):
        pprint("Bye! Bye!")
        return True

    def do_prm(self, inp):
        pprint(self._db_mngr.get_table(Parameter))

    def help_prm(self):
        pprint("Display 'Parameter' table")

    def do_info(self, inp):
        print(type(inp))
        pprint(self._db_reader.info(inp))

    def help_info(self):
        pprint("Display 'Info' table")


class ReadDatabase:
    """Class used to display DB content"""

    def __init__(self):
        self._db_mngr = DatabaseManager()

    def info(self, expr):
        out = SearchInfoExpr.eval(expr)
        return self._db_mngr.get_table(Info, search={'where': Info.info_id.in_(out)})


class SearchInfoExpr(metaclass=ABCMeta):
    """Abstract search info expression interpreter class.

    .. uml::

        @startuml
        footer Interpreter design pattern

        class SearchInfoExpr {
            {abstract} interpret()
            {static} eval()
        }

        class LogicalSearchInfoExpr {
            _expression: List
            interpret()
            {abstract} fct(*arg)
        }

        SearchInfoExpr <|-- LogicalSearchInfoExpr : extends
        LogicalSearchInfoExpr o--> SearchInfoExpr

        class TerminalSearchInfoExpr {
            interpret()
            {abstract} get_filter()
        }

        SearchInfoExpr <|-- TerminalSearchInfoExpr : extends

        @enduml

    """

    def __init__(self, str_expr_dict_stgy):
        self._str_expr_dict_stgy = str_expr_dict_stgy

    @abstractmethod
    @property
    def str_expr_dict(self):
        return self._str_expr_dict_stgy.execute()

    @abstractmethod
    def interpret(self):
        """Interpreter method"""

    @staticmethod
    def eval(str_expr, prm_name=None, filter_empty=False):
        """Evaluate search expression

        Args:
            str_expr (str): Expression to evaluate
            prm_name (str, `optional`): Search parameter. Default to None.
            filter_empty (bool, `optional`): Filter for empty data. Default to False.

        Returns:
            List of Info.info_id

        Search expression grammar:
            - all(): Select all
            - [datetime ; dt]('<ISO datetime>', ['=='(default) ; '>=' ; '>' ; '<=' ; '<' ; '!=']): Select by datetime
            - [serialnumber ; srn]('<Serial number>'): Select by serial number
            - [product_id ; pid](<Product>): Select by product
            - tags(['<Tag>' ; ('<Tag 1>', ...,'<Tag n>')]): Select by tag
            - prm('<Parameter name>'): Select by parameter name
            - and_(<expr 1>, ..., <expr n>): Intersection
            - or_(<expr 1>, ..., <expr n>): Union
            - not_(<expr>): Negation, correspond to all() without <expr>

        Shortcut expressions:
            - raw(): Same as tags('raw')
            - gdp(): Same as tags('gdp')

        """

        # Define
        str_expr_dict = {
            'all': AllExpr,
            'datetime': DatetimeExpr, 'dt': DatetimeExpr,
            'serialnumber': SerialNumberExpr, 'srn': SerialNumberExpr,
            'product_id': ProductExpr, 'pid': ProductExpr,
            'tags': TagExpr,
            'prm': ParameterExpr,
            'and_': AndExpr,
            'or_': OrExpr,
            'not_': NotExpr,
            'raw': RawExpr,
            'gdp': GDPExpr,
        }

        # Eval expression
        expr = eval(str_expr, str_expr_dict)

        # Test if double str (prevent exception for use in shell module)
        if isinstance(expr, str):
            expr = eval(expr, str_expr_dict)

        # Add empty tag if False
        if filter_empty is True:
            expr = AndExpr(NotExpr(TagExpr(TAG_EMPTY_NAME)), expr)

        # Filter parameter
        if prm_name:
            expr = AndExpr(ParameterExpr(prm_name), expr)

        # Interpret expression
        expr_res = expr.interpret()

        # Convert id as table element
        qry = TableInfo.select().where(TableInfo.info_id.in_(expr_res))
        out = [arg for arg in qry.iterator()]

        # TODO
        #  Raise exception

        return out


class LogicalSearchInfoExpr(SearchInfoExpr):
    """
    Implement an interpret operation for nonterminal symbols in the grammar.
    """

    def __init__(self, *args):
        self._expression = args

    def interpret(self):
        """Non terminal interpreter method"""
        return reduce(
            self.fct,
            [arg.interpret() for arg in self._expression]
        )

    @abstractmethod
    def fct(self, *args):
        """Logical function between expression args"""


class AndExpr(LogicalSearchInfoExpr):
    """And operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.and_(a, b)


class OrExpr(LogicalSearchInfoExpr):
    """Or operation"""

    def fct(self, a, b):
        """Implement fct method"""
        return operator.or_(a, b)


class NotExpr(LogicalSearchInfoExpr):
    """Not operation"""

    def __init__(self, arg):
        self._expression = [AllExpr(), arg]

    def fct(self, a, b):
        """Implement fct method"""
        return operator.sub(a, b)


class TerminalSearchInfoExpr(SearchInfoExpr):
    """
    Implement an interpret operation associated with terminal symbols in
    the grammar.
    """

    QRY_BASE = (
        TableInfo
        .select().distinct()
        .join(TableInfosObjects).join(TableObject).switch(TableInfo)
        .join(TableParameter).switch(TableInfo)
        .join(InfosTags).join(TableTag).switch(TableInfo)
    )

    def __init__(self, arg):
        self.expression = arg

    def interpret(self):
        """Terminal expression interpreter"""
        return set(
            arg.info_id for arg in
            self.QRY_BASE.where(self.get_filter()).iterator()
        )

    @abstractmethod
    def get_filter(self):
        """Return query where method filter"""


class AllExpr(TerminalSearchInfoExpr):
    """All filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return


class DatetimeExpr(TerminalSearchInfoExpr):
    """Datetime filter"""

    _OPER_DICT = {
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '<': operator.lt,
        '>=': operator.ge,
        '>=': operator.le,
    }
    expression = TProp(Union[str, Timestamp, datetime], check_datetime)

    def __init__(self, arg, op='=='):
        self.expression = arg
        self._op = self._OPER_DICT[op]

    def get_filter(self):
        """Implement get_filter method"""
        return self._op(TableInfo.edt, self.expression)


class SerialNumberExpr(TerminalSearchInfoExpr):
    """Serial number filter"""

    expression = TProp(
        Union[str, Iterable[str]],
        setter_fct=lambda x: [x] if isinstance(x, str) else list(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.srn.in_(self.expression)


class ProductExpr(TerminalSearchInfoExpr):
    """Product filter"""

    expression = TProp(
        Union[int, Iterable[int]],
        setter_fct=lambda x: [x] if isinstance(x, int) else list(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.pid.in_(self.expression)


class TagExpr(TerminalSearchInfoExpr):
    """Tag filter"""

    expression = TProp(
        Union[str, Iterable[str]], lambda x: set((x,)) if isinstance(x, str) else set(x)
    )

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_(self.expression)


class ParameterExpr(TerminalSearchInfoExpr):
    """Parameter filter"""

    expression = TProp(str, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableParameter.prm_name == self.expression


class RawExpr(TerminalSearchInfoExpr):
    """Raw filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_([TAG_RAW_NAME])


class GDPExpr(TerminalSearchInfoExpr):
    """GDP filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_([TAG_GDP_NAME])




class Visitor(metaclass=ABCMeta):
    """"""

    @abstractmethod
    def visit_info(self):
        """"""


class
