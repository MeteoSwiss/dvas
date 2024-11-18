"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Local database exploring tools

"""

# Import from python packages
import logging
from abc import abstractmethod, ABCMeta
from collections.abc import Iterable
import re
import operator
from functools import reduce
from datetime import datetime
from pandas import DataFrame
from pandas import Timestamp
from peewee import JOIN
from playhouse.shortcuts import model_to_dict

# Import from current package
from .model import Object as TableObject
from .model import Info as TableInfo
from .model import Prm as TableParameter
from .model import Tag as TableTag
from .model import InfosObjects as TableInfosObjects
from .model import InfosTags as TableInfosTags
from .model import DataSource as TablDataSource
from .model import Model as TableModel
from ..hardcoded import TAG_EMPTY, TAG_ORIGINAL, TAG_GDP
from ..hardcoded import EID_PAT, RID_PAT, TOD_PAT
from ..helper import TypedProperty as TProp
from ..helper import check_datetime
from ..errors import SearchError, DvasError

# Setup the local logger
logger = logging.getLogger(__name__)

# If some DB debug is required, uncomment the following lines to get all the peewee logs
# logger2 = logging.getLogger('peewee')
# logger2.addHandler(logging.StreamHandler())
# logger2.setLevel(logging.DEBUG)

# Global define
EID_PAT_COMPILED = re.compile(EID_PAT)
RID_PAT_COMPILED = re.compile(RID_PAT)
TOD_PAT_COMPILED = re.compile(TOD_PAT)


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
    _exclude = None

    @classmethod
    def set_stgy(cls, method):
        """Set class attribute in function of the strategy"""

        if method == 'info':
            stgy = InfoStrategy()

        elif method == 'prm':
            stgy = PrmStrategy()

        elif method == 'obj':
            stgy = ObjectStrategy()

        else:
            raise DvasError(f"Unknown stgy: {stgy}")

        cls._str_expr_dict = stgy.str_expr_dict
        cls._qry = stgy.qry
        cls._id = stgy.id
        cls._exclude = stgy.exclude

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
                'dict' return table as dict. Default to 'id'.
            recurse (bool, `optional`): Search recursively DB content. Default to False.

        Returns:
            List of Info.info_id

        Search expression grammar for 'info' method:
            - all(): Select all
            - [datetime ; dt]('<ISO datetime>', ['=='(default) ; '>=' ; '>' ; '<=' ; '<' ; '!=']):
                Select by datetime
            - [serialnumber ; srn]('<Serial number>'): Select by serial number
            - [product_id ; pid](<Product>): Select by product
            - [object_id, oid](<Object id>): Select by object id
            - [model_id, mid](<Model id>): Select by model id
            - tags(['<Tag>' ; ('<Tag 1>', ...,'<Tag n>')]): Select by tag
            - prm('<Parameter name>'): Select by parameter name
            - and_(<expr 1>, ..., <expr n>): Intersection
            - or_(<expr 1>, ..., <expr n>): Union
            - not_(<expr>): Negation, correspond to all() without <expr>

        Shortcut expressions:
            - original(): Same as tags('original')
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
                assert isinstance(expr, SearchInfoExpr), \
                    "eval(expr) must return a 'SearchInfoExpr' type"

            # Add empty tag if False
            if filter_empty is True:
                expr = AndExpr(NotExpr(TagExpr(TAG_EMPTY)), expr)

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
                out = [model_to_dict(arg, recurse=recurse, backrefs=True,
                                     exclude=SearchInfoExpr._exclude) for arg in qry]

        except (Exception, AssertionError) as exc:
            # TODO
            #  Detail exception
            raise SearchError(exc)

        return out

    @staticmethod
    def extract_global_view():
        """ Extract global view from DB

        Return:
            pd.DataFrame

        """

        # Set strategy
        SearchInfoExpr.set_stgy('obj')

        # Get DB explore result
        res = SearchInfoExpr.eval('all()', recurse=True, out='dict')

        # Convert to DataFrame
        res = DataFrame.from_dict(res)

        # Go through 'model' column
        res[TableModel.mid.name] = \
            res[TableObject.model.name].apply(lambda x: x[TableModel.mid.name])
        res[TableModel.mdl_name.name] = \
            res[TableObject.model.name].apply(lambda x: x[TableModel.mdl_name.name])
        res[TableModel.mdl_desc.name] = \
            res[TableObject.model.name].apply(lambda x: x[TableModel.mdl_desc.name])
        res.drop(columns=[TableObject.model.name], inplace=True)

        # Go through 'infos_objects'
        # TODO: Fix this ugly hardcoded ref to 'infos_objects' and 'infos_tags'
        res[TablDataSource.src.name] = res['infos_objects'].apply(
            lambda x:
            x[0][TableInfosObjects.info.name][TableInfo.data_src.name][TablDataSource.src.name]
        )
        res[TableInfo.edt.name] = res['infos_objects'].apply(
            lambda x: x[0][TableInfosObjects.info.name][TableInfo.edt.name]
        )
        res['eid'] = res['infos_objects'].apply(
            lambda x: SearchInfoExpr.get_eid(x[0][TableInfosObjects.info.name]['infos_tags']))
        res['rid'] = res['infos_objects'].apply(
            lambda x: SearchInfoExpr.get_rid(x[0][TableInfosObjects.info.name]['infos_tags']))
        res['tod'] = res['infos_objects'].apply(
            lambda x: SearchInfoExpr.get_tod(x[0][TableInfosObjects.info.name]['infos_tags']))
        res['is_gdp'] = res['infos_objects'].apply(
            lambda x: SearchInfoExpr.get_isgdp(x[0][TableInfosObjects.info.name]['infos_tags']))
        res.drop(columns=['infos_objects'], inplace=True)

        # TODO: @fpvogt I leave it to you to put the columns in the order you prefer ;-)

        return res

    @staticmethod
    def get_eid(infos_tags):
        """Return eid"""

        try:
            out = next(
                arg[TableInfosTags.tag.name][TableTag.tag_name.name]
                for arg in infos_tags
                if EID_PAT_COMPILED.match(arg[TableInfosTags.tag.name][TableTag.tag_name.name])
                is not None
            )

        except StopIteration:
            out = None

        return out

    @staticmethod
    def get_rid(infos_tags):
        """Return eid"""

        try:
            out = next(
                arg[TableInfosTags.tag.name][TableTag.tag_name.name]
                for arg in infos_tags
                if RID_PAT_COMPILED.match(arg[TableInfosTags.tag.name][TableTag.tag_name.name])
                is not None
            )

        except StopIteration:
            out = None

        return out

    @staticmethod
    def get_tod(infos_tags):
        """ Return the TimeOfDay """

        try:
            out = next(
                arg[TableInfosTags.tag.name][TableTag.tag_name.name]
                for arg in infos_tags
                if TOD_PAT_COMPILED.match(arg[TableInfosTags.tag.name][TableTag.tag_name.name])
                is not None
            )

        except StopIteration:
            out = None

        return out

    @staticmethod
    def get_isgdp(infos_tags):
        """Return eid"""

        try:
            next(
                arg[TableInfosTags.tag.name][TableTag.tag_name.name]
                for arg in infos_tags
                if arg[TableInfosTags.tag.name][TableTag.tag_name.name] == TAG_GDP
            )
            out = True

        except StopIteration:
            out = False

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
        '<=': operator.le,
    }
    expression = TProp(str | Timestamp | datetime, check_datetime)

    def __init__(self, arg, op='=='):
        self.expression = arg
        self._op = self._OPER_DICT[op]

    def get_filter(self):
        """Implement get_filter method"""
        return self._op(TableInfo.edt, self.expression)


class SerialNumberExpr(TerminalSearchInfoExpr):
    """Serial number filter"""

    expression = TProp(str | Iterable,
                       setter_fct=lambda x: [x] if isinstance(x, str) else list(x))

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.srn.in_(self.expression)


class ProductExpr(TerminalSearchInfoExpr):
    """Product filter"""

    expression = TProp(int | Iterable,
                       setter_fct=lambda x: [x] if isinstance(x, int) else list(x))

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.pid.in_(self.expression)


class TagExpr(TerminalSearchInfoExpr):
    """Tag filter"""

    expression = TProp(str | Iterable, lambda x: set((x,)) if isinstance(x, str) else set(x))

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_(self.expression)


class ParameterExpr(TerminalSearchInfoExpr):
    """Parameter filter"""

    expression = TProp(str, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableParameter.prm_name == self.expression


class OriginalExpr(TerminalSearchInfoExpr):
    """Original filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_([TAG_ORIGINAL])


class GDPExpr(TerminalSearchInfoExpr):
    """GDP filter"""

    def __init__(self):
        pass

    def get_filter(self):
        """Implement get_filter method"""
        return TableTag.tag_name.in_([TAG_GDP])


class OIDExpr(TerminalSearchInfoExpr):
    """OID filter"""

    expression = TProp(int, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableObject.oid == self.expression


class MIDExpr(TerminalSearchInfoExpr):
    """MID filter"""

    expression = TProp(str, lambda x: x)

    def get_filter(self):
        """Implement get_filter method"""
        return TableModel.mid == self.expression


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

    @property
    @abstractmethod
    def exclude(self):
        """list: Field instances which should be excluded from the result dictionary."""


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
                'model_id': MIDExpr, 'mid': MIDExpr,
                'tags': TagExpr,
                'prm': ParameterExpr,
                'original': OriginalExpr,
                'gdp': GDPExpr,
            }
        )

    @property
    def qry(self):
        """peewee.ModelSelect: Query"""
        return (
            TableInfo
            .select().distinct()
            .join(TableInfosObjects).join(TableObject).join(TableModel).switch(TableInfo)
            .join(TableParameter).switch(TableInfo)
            .join(TableInfosTags).join(TableTag).switch(TableInfo)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'info_id'

    @property
    def exclude(self):
        """list: Field instances which should be excluded from the result dictionary."""
        return [TableInfo.datas]  # noqa pylint: disable=E1101


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
            .join(TableInfo, JOIN.LEFT_OUTER).switch(TableParameter)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'prm_id'

    @property
    def exclude(self):
        """list: Field instances which should be excluded from the result dictionary."""
        return [TableInfo.datas]  # noqa pylint: disable=E1101


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
                'original': OriginalExpr,
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
            .join(TableInfosObjects, JOIN.LEFT_OUTER).join(TableInfo)
            .join(TableInfosTags).join(TableTag).switch(TableInfo)
        )

    @property
    def id(self):
        """str: Query main table id name"""
        return 'oid'

    @property
    def exclude(self):
        """list: Field instances which should be excluded from the result dictionary."""
        return [TableInfo.datas]  # noqa pylint: disable=E1101
