"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Package helper classes and functions.

"""

# Import external packages and modules
from pathlib import Path
from re import compile, IGNORECASE
from datetime import datetime
from copy import deepcopy as dc
from functools import wraps, reduce
from abc import ABC, ABCMeta, abstractmethod
from inspect import getmodule
from operator import getitem
import pytz
from pampy import match as pmatch
from pampy import MatchError
from pandas import to_datetime


def camel_to_snake(name):
    """Convert camel case to snake case

    Args:
        name (str): Camel case string

    Returns:
        str
    """

    # Define module global
    first_cap_re = compile('(.)([A-Z][a-z]+)')
    all_cap_re = compile('([a-z0-9])([A-Z])')

    # Convert
    return all_cap_re.sub(
        r'\1_\2',
        first_cap_re.sub(r'\1_\2', name)
    ).lower()


class SingleInstanceMetaClass(type):
    """Metaclass to create single instance of class

    Note:
        `Source code
        <https://www.pythonprogramming.in/singleton-class-using-metaclass-in-python.html>`__

    """

    def __init__(cls, name, bases, dic):
        """Constructor"""
        cls.__single_instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        """Class __call__ method"""

        # Will return 1st created instance if empty and if class name is not
        # in 'DVAS_SKIP_SINGLETON' environment variable
        if cls.__single_instance is not None:
            return cls.__single_instance
        single_obj = cls.__new__(cls)
        single_obj.__init__(*args, **kwargs)
        cls.__single_instance = single_obj
        return single_obj


class RequiredAttrMetaClass(ABCMeta):
    """Meta class for requiring specific attribute to abstract parent class

    Use this meta abstract class to construct a parent abstract class to
    require special attributes into children.

    """

    REQUIRED_ATTRIBUTES = {}
    """dict: Required class attributes.
    key: attribute name, value: required attribute type
    """

    def __call__(cls, *args, **kwargs):
        """Class __call__ method"""
        obj = super(RequiredAttrMetaClass, cls).__call__(*args, **kwargs)

        for attr_name, dtype in obj.REQUIRED_ATTRIBUTES.items():

            if not hasattr(obj, attr_name):
                errmsg = (
                    f"required attribute {attr_name} not defined in class {obj}"
                )
                raise ValueError(errmsg)

            #TODO
            # Use pampy to check pattern
            obj_attr = getattr(obj, attr_name)
            if not isinstance(obj_attr, dtype):
                errmsg = (
                    f"Attribute '{attr_name}' badly set in class {obj}. " +
                    f"Must be a {dtype} instead a {type(obj_attr)}"
                )
                raise ValueError(errmsg)

        return obj


class ContextDecorator(ABC):
    """Use this class as superclass of a context manager to convert it into a decorator.

    Note:
        `Source code <http://sametmax.com/les-context-managers-et-le-mot-cle-with-en-python/>`__

    """

    def __init__(self):
        """Abstract constructor"""
        self._func = None

    def __call__(self, func):
        """Class __call__ method.

        Args:
            func (callable): Decorated function

        """

        @wraps(func)
        def decorated(*args, **kwargs):
            self._func = func
            with self:
                return func(*args, **kwargs)
        return decorated

    @property
    def func(self):
        """callable: Contain the decorated function. Set automatically only
        when class is used as decorated. Default to None.
        """
        return self._func

    @abstractmethod
    def __enter__(self):
        """Abstract __enter__ method"""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Abstract __exit__ method"""


class TimeIt(ContextDecorator):
    """Code elapsed time calculator context manager/decorator.

    """

    def __init__(self, header_msg=''):
        """Constructor.

        Args:
            header_msg (str): User defined elapsed time header. Default to ''.

        """
        super().__init__()
        self._start = None
        self._head_msg = header_msg

    def __enter__(self):
        """Class __enter__ method"""

        # Start time
        self._start = datetime.now()

        # Set msg header
        if (self.func is None) and (self._head_msg == ''):
            self._head_msg = 'Execution time'
        elif self.func is not None:
            self._head_msg = (
                '{}.{} execution time'
            ).format(
                getmodule(self.func).__name__,
                self.func.__qualname__
            )
        else:
            self._head_msg += 'execution time'

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""

        # Calculate execution time
        delta = datetime.now() - self._start

        # Print
        print(f'{self._head_msg}: {delta}', flush=True)


def deepcopy(func):
    """Use a deepcopy of the class when calling the method.
    The method keywords must contain the 'inplace' argument.
    """

    @wraps(func)
    def decorated(*args, inplace=True, **kwargs):

        if inplace:
            func(*args, **kwargs)
            res = None
        else:
            res = dc(args[0])
            func(res, *args[1:], **kwargs)
        return res

    return decorated


class TypedProperty:
    """Typed property class

    Note:
        Adapted from `Stackoverflow.
        <https://stackoverflow.com/questions/34884947/
        understanding-a-python-descriptors-example-typedproperty>`__

    """
    def __init__(self, pampy_match, setter_fct=None, args=None, kwargs=None, getter_fct=None):
        """Constructor

        Args:
            pampy_match (type or tuple of type): Data type
            setter_fct: Function applied before assign value in setter method.
                The function can include special check and raises -
                use TypeError to raise appropriate exception.
            args (tuple): setter function args
            kwargs (dict): setter function kwargs
        """
        # Set attributes
        self._pampy_match = pampy_match
        self._getter_fct = (lambda x: x) if getter_fct is None else getter_fct
        self._setter_fct = (lambda x: x) if setter_fct is None else setter_fct
        self._setter_fct_args = tuple() if args is None else args
        self._setter_fct_kwargs = dict() if kwargs is None else kwargs

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._getter_fct(instance.__dict__[self._name])

    def __set__(self, instance, val):
        # Test type
        try:
            instance.__dict__[self._name] = pmatch(
                val, self._pampy_match, self._setter_fct(
                    val, *self._setter_fct_args, **self._setter_fct_kwargs
                )
            )
        except (MatchError, TypeError) as first_error:
            raise TypeError(f'Bad type while assignment of {self._name} <- {val}') from first_error

    def __set_name__(self, instance, name):
        """Attribute name setter"""
        self._name = name

    @staticmethod
    def re_str_choice(choices, ignore_case=False):
        """Method to create re.compile for a list of str

        Args:
            choices (list of str): Choice of strings
            ignore_case (bool): Ignore case if True. Default to False.

        Return:
            re.compile

        """

        # Create pattern
        pattern = '^(' + ')|('.join(choices) + ')$'

        # Create re.compile
        if ignore_case:
            out = compile(pattern, IGNORECASE)
        else:
            out = compile(pattern)

        return out


def get_by_path(root, items, sep='.'):
    """Access a nested object in root by item sequence.

    Args:
        root (dict|list|class): Object to access
        items (list|str): Item sequence. String sequence must be separated
            by sep value.
        sep (str, optional): Separator.

    Returns:
        object: Nested value

    Raises:
        Exception associated to lists and dicts.

    Examples:
    >>>get_by_path([1,[2,3]], [1, 0])
    2

    >>>get_by_path({'a': [0, 1]}, ['a', 0])
    0

    >>>get_by_path({'a': {'b': 1, 'c':0}}, ['a', 'c'])
    0

    >>>get_by_path({'a': {'b': 1, 'c':0}}, 'a.c')
    0

    """

    # Split '.'
    if isinstance(items, str):
        items = items.split(sep)

    # Get item/attr
    if isinstance(root, (dict, list)):
        out = reduce(getitem, items, root)
    else:
        out = reduce(getattr, items, root)

    return out


def check_path(value, exist_ok=False):
    """Test and set input argument into pathlib.Path object.

    Args:
        value (pathlib.Path, str): Argument to be tested
        exist_ok (bool, optional): If True check existence. Default to False.

    Returns:
        pathlib.Path

    Raises:
        TypeError: In case if path does not exist falls exist_ok is True

    """

    # Test existence
    if exist_ok is True:
        try:
            out = Path(value).resolve(strict=True)
        except FileNotFoundError as first_error:
            raise TypeError(f"Path '{value}' does not exist") from first_error
    else:
        try:
            out = Path(value).resolve(strict=False)
        except (TypeError, OSError) as first_error:
            raise TypeError(f"Bad path name for '{value}'") from first_error

    return out


def check_datetime(val, utc=True):
    """Test and set input argument into datetime.datetime.

    Args:
        val (str | datetime | pd.Timestamp): Datetime
        utc (bool): Check UTC. Default to True.

    Returns:
        datetime.datetime

    """
    if utc:
        try:
            assert (out := to_datetime(val).to_pydatetime()).tzinfo == pytz.UTC
        except (ValueError, AssertionError) as first_error:
            raise TypeError(f"Not UTC or bad datetime format for '{val}'") from first_error

    else:
        try:
            out = to_datetime(val).to_pydatetime()
        except ValueError as first_error:
            raise TypeError(f"Bad datetime format for '{val}'") from first_error

    return out


def unzip(val):
    """Unzip list of tuple

    Args:
        val (list of tuples): Zipped list

    Returns:
        list

    """
    return zip(*val)


def get_dict_len(val):
    """Return the length of a dict (recursively)

    Args:
        val (dict): Dict to get length

    Returns:
        int

    """
    out = 0
    for arg in val.values():
        if isinstance(arg, dict):
            out += get_dict_len(arg)
        else:
            out += 1
    return out
