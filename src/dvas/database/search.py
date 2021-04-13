"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database exploring tools

"""

# Import from python packages
from abc import abstractmethod, ABCMeta
import operator
from functools import reduce
from datetime import datetime
from pandas import Timestamp
from pampy.helpers import Iterable, Union
from playhouse.shortcuts import model_to_dict

# Import from current package
from .model import Object as TableObject
from .model import Info as TableInfo
from .model import Parameter as TableParameter
from .model import Tag as TableTag
from .model import InfosObjects as TableInfosObjects
from .model import InfosTags
from .model import Model as TableModel
from ..hardcoded import TAG_EMPTY_NAME
from ..hardcoded import TAG_RAW_NAME, TAG_GDP_NAME
from ..helper import TypedProperty as TProp
from ..helper import check_datetime
from ..errors import SearchError


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

    _str_expr_dict = None
    _qry = None
    _id = None

    @classmethod
    def set_stgy(cls, method):
        """Set class attribute in function of the strategy"""

        if method == 'info':
            stgy = InfoStrategy()

        elif method == 'prm':
            stgy = PrmStrategy()

        elif method == 'obj':
            stgy = ObjectStrategy()

        cls._str_expr_dict = stgy.str_expr_dict
        cls._qry = stgy.qry
        cls._id = stgy.id

    @abstractmethod
    def interpret(self):
        """Interpreter method"""

    @staticmethod
    def eval(expr, prm_name=None, filter_empty=False, out='id', recurse=False):
        """Evaluate search expression

        Args:
            expr (SearchInfoExpr, str): Expression to evaluate
            prm_name (str, `optional`): Search parameter. Default to None.
            filter_empty (bool, `optional`): Filter for empty data. Default to False.
            out (str, `optional`): 'id' return table element.
                'dict' return table as dict. Default to 'table'.
            recurse (bool, `optional`): Search recursively DB content. Default to False.

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

        Raises:
            - SearchError: Error in search expression

        """

        # Test
        assert out in ['id', 'dict'], "Bad value for 'out'"

        try:

            # Skip if SearchInfoExpr
            if isinstance(expr, SearchInfoExpr):
                pass

            # Eval expression if str
            if isinstance(expr, str):
                expr = eval(expr, SearchInfoExpr._str_expr_dict)

                # Eval twice for nested str
                if isinstance(expr, str):
                    expr = eval(expr, SearchInfoExpr._str_expr_dict)

                # Test
                assert isinstance(expr, SearchInfoExpr), "eval(expr) must return a 'SearchInfoExpr' type"

            # Add empty tag if False
            if filter_empty is True:
                expr = AndExpr(NotExpr(TagExpr(TAG_EMPTY_NAME)), expr)

            # Filter parameter
            if prm_name:
                expr = AndExpr(ParameterExpr(prm_name), expr)

            # Interpret expression
            expr_res = expr.interpret()

            # Convert results
            qry = SearchInfoExpr._qry.where(
                getattr(SearchInfoExpr._qry.model, SearchInfoExpr._id).in_(expr_res)
            )
            if out == 'id':
                out = [arg for arg in qry.iterator()]

            else:
                out = [model_to_dict(arg, recurse=recurse) for arg in qry]

        except (Exception, AssertionError) as exc:
            # TODO
            #  Detail exception
            raise SearchError(exc)

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

    def __init__(self, arg):
        self.expression = arg

    def interpret(self):
        """Terminal expression interpreter"""
        return set(
            getattr(arg, self._id) for arg in
            self._qry.where(self.get_filter()).iterator()
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


class OIDExpr(TerminalSearchInfoExpr):
    """OID filter"""

    expression = TProp(int, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.oid == self.expression


class SearchStrategyAC(metaclass=ABCMeta):
    """Abstract class (AC) for a search strategy"""

    @property
    @abstractmethod
    def str_expr_dict(self):
        """dict: Str equivalent expression"""
        return {
            'all': AllExpr,
            'and_': AndExpr,
            'or_': OrExpr,
            'not_': NotExpr,
        }

    @property
    @abstractmethod
    def qry(self):
        """peewee.ModelSelect: Query"""

    @property
    @abstractmethod
    def id(self):
        """str: Query main table id name"""


class InfoStrategy(SearchStrategyAC):
    """Search Info strategy"""

    @property
    def str_expr_dict(self):
        """dict: Str equivalent expression"""
        return dict(
            **super().str_expr_dict,
            **{
                'datetime': DatetimeExpr, 'dt': DatetimeExpr,
                'serialnumber': SerialNumberExpr, 'srn': SerialNumberExpr,
                'object_id': OIDExpr, 'oid': OIDExpr,
                'product_id': ProductExpr, 'pid': ProductExpr,
                'tags': TagExpr,
                'prm': ParameterExpr,
                'raw': RawExpr,
                'gdp': GDPExpr,
            }
        )

    @property
    def qry(self):
        """peewee.ModelSelect: Query"""
        return (
            TableInfo
            .select().distinct()
            .join(TableInfosObjects).join(TableObject).switch(TableInfo)
            .join(TableParameter).switch(TableInfo)
            .join(InfosTags).join(TableTag).switch(TableInfo)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'info_id'


class PrmStrategy(SearchStrategyAC):
    """Search Parameter strategy"""

    @property
    def str_expr_dict(self):
        """dict: Str equivalent expression"""
        return dict(
            **super().str_expr_dict,
            **{
                'datetime': DatetimeExpr, 'dt': DatetimeExpr,
                'prm': ParameterExpr,
            }
        )

    @property
    def qry(self):
        """peewee.ModelSelect: Query"""
        return (
            TableParameter
            .select().distinct()
            .join(TableInfo).switch(TableParameter)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'prm_id'


class ObjectStrategy(SearchStrategyAC):
    """Search Object strategy"""

    @property
    def str_expr_dict(self):
        """dict: Str equivalent expression"""
        return dict(
            **super().str_expr_dict,
            **{
                'datetime': DatetimeExpr, 'dt': DatetimeExpr,
                'serialnumber': SerialNumberExpr, 'srn': SerialNumberExpr,
                'object_id': OIDExpr, 'oid': OIDExpr,
                'product_id': ProductExpr, 'pid': ProductExpr,
                'tags': TagExpr,
                'raw': RawExpr,
                'gdp': GDPExpr,
            }
        )

    @property
    def qry(self):
        """peewee.ModelSelect: Query"""
        return (
            TableObject
            .select().distinct()
            .join(TableModel).switch(TableObject)
            .join(TableInfosObjects).join(TableInfo)
            .join(InfosTags).join(TableTag).switch(TableInfo)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'oid'
