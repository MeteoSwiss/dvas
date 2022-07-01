"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Package helper classes and functions.

"""

# Import external packages and modules
from pathlib import Path
import inspect
from inspect import getmembers, isroutine
import re
from datetime import datetime
from copy import deepcopy as dc
from functools import wraps, reduce
from abc import ABC, ABCMeta, abstractmethod
from contextlib import AbstractContextManager
from weakref import WeakValueDictionary
from operator import getitem
from pandas import to_datetime

# Import from this module
from .errors import DvasError


def camel_to_snake(name):
    """Convert camel case to snake case

    Args:
        name (str): Camel case string

    Returns:
        str
    """

    # Define module global
    first_cap_re = re.compile('(.)([A-Z][a-z]+)')
    all_cap_re = re.compile('([a-z0-9])([A-Z])')

    # Convert
    return all_cap_re.sub(
        r'\1_\2',
        first_cap_re.sub(r'\1_\2', name)
    ).lower()


class SingleInstanceMetaClass(type):
    """Metaclass to create single instance of class

    Note:
        `Source code
        <https://stackoverflow.com/questions/43619748/destroying-a-singleton-object-in-python>`__

    """
    _instances = WeakValueDictionary()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # This variable declaration is required to force a
            # strong reference on the instance.
            instance = super(SingleInstanceMetaClass, cls).__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

    @classmethod
    def has_instance(mcs, inst):
        """Check if instance

        Args:
            inst (type): Instance type to check

        """
        return inst in mcs._instances.keys()


class RequiredAttrMetaClass(ABCMeta):
    """Meta class for requiring specific attribute to abstract parent class

    Use this meta abstract class to construct a parent abstract class to
    require special attributes into children.

    """

    REQUIRED_ATTRIBUTES = {}
    """dict: Required class attributes.
    key: attribute name, value: required attribute type
    """

    def __init__(cls, *args, **kwargs):
        """ Without this __init__ function, the Child classes all show the init signature of the
        __call__ method below, when accessing their help.

        This corrects this unwanted behavior (see #84 for details).

        Adapted from the reply of johnbaltis on
        `SO <https://stackoverflow.com/questions/49740290>`__ .
        """

        # Restore the class init signature
        sig = inspect.signature(cls.__init__)
        parameters = tuple(sig.parameters.values())
        cls.__signature__ = sig.replace(parameters=parameters[1:])

        # Finally, call the real __init__
        super().__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        """Class __call__ method"""
        obj = super(RequiredAttrMetaClass, cls).__call__(*args, **kwargs)

        for attr_name, dtype in obj.REQUIRED_ATTRIBUTES.items():

            if not hasattr(obj, attr_name):
                errmsg = (
                    f"required attribute {attr_name} not defined in class {obj}"
                )
                raise ValueError(errmsg)

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


class TimeIt(AbstractContextManager):
    """Code elapsed time calculator context manager."""

    def __init__(self, header_msg='', logger=None):
        """Constructor.

        Args:
            header_msg (str): User defined elapsed time header. Default to ''.
            logger (logging.Logger, `optional`): Print output to log (debug level only).
                Defaults to None.

        """
        super().__init__()
        self._start = None
        self._head_msg = header_msg
        self._logger = logger

    def __enter__(self):
        """Class __enter__ method"""

        # Start time
        self._start = datetime.now()

        # Set msg header
        if self._head_msg == '':
            self._head_msg = 'Execution time'
        else:
            self._head_msg += ' execution time'

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""

        # Calculate execution time
        delta = datetime.now() - self._start

        # Print
        msg = f'{self._head_msg}: {delta}'
        if self._logger is None:
            print(msg, flush=True)
        else:
            self._logger.debug(msg)


def deepcopy(func):
    """ Use a deepcopy of the class when calling a given "func" function.

    Intended to be used as a decorator, that will "correctly" handle the decorated function
    signature AND its docstring.

    Note:
      This implementation was inspired by the following sources:

        - The reply from `metaperture` to `this SO post
          <https://stackoverflow.com/questions/1409295/set-function-signature-in-python>`__
        - `This excellent article
          <https://utilipy.readthedocs.io/en/latest/examples/making-decorators.html>`__ by
          N. Starkman.
        - The `wrapt docs
          <https://wrapt.readthedocs.io/en/latest/decorators.html#signature-changing-decorators>`__
    """

    @wraps(func)
    def decorated(*args, inplace=True, **kwargs):
        """ Decorating function

        Args:
            inplace (bool, optional): if False, will return a deepcopy. Defaults to True.

        """

        if inplace:
            func(*args, **kwargs)
            res = None
        else:
            res = dc(args[0])
            func(res, *args[1:], **kwargs)
        return res

    # I now shall deal with the decorated function signature.
    # I need to add the 'inplace' Parameter to it.
    new_param = inspect.Parameter('inplace', inspect.Parameter.KEYWORD_ONLY, default=True)
    sig = inspect.signature(decorated)
    func_params = tuple(sig.parameters.values())
    # Here, I cannot just add a new Parameter blindly. I have to do keep it in the proper order.
    if func_params[-1].name == 'kwargs':
        func_params = func_params[:-1] + (new_param, func_params[-1],)
    else:
        func_params = func_params + (new_param,)
    # Set the nnew parameters in the signature
    sig = sig.replace(parameters=func_params)
    # I also need to adjust the docstring to document this inplace parameter.
    # I'll append some clear message to the existing docstring.
    decorated.__signature__ = sig
    decorated.__doc__ += """--- Decorating function infos ---

        Args:
            inplace (bool, optional): If False, will return a deepcopy. Defaults to True.

        ---   ---   ---   ---   ---   ---
    """

    return decorated


class TypedProperty:
    """Typed property class

    Note:
        Adapted from `Stackoverflow.
        <https://stackoverflow.com/questions/34884947/
        understanding-a-python-descriptors-example-typedproperty>`__

    """
    def __init__(self, match, setter_fct=None, args=None, kwargs=None, getter_fct=None,
                 allow_none=False):
        """Constructor

        Args:
            match (type or types.UnionType): Data type(s), used in a isinstance check
            setter_fct (callable, `optional`): Function applied before assign value in setter
                method. The function can include special check and raises -
                use TypeError to raise appropriate exception. Default to lambda x: x
            args (tuple, `optional`): setter function args. Default to None.
            kwargs (dict, `optional`): setter function kwargs. Default to None.
            getter_fct (callable, `optional`): Function applied before returning attributes in
                getter method. Default to lambda x: x
            allow_none (bool, `optional`): Allow none value (bypass pampy match and setter fct).
                Defaults to False.

        Note:
            pampy no longer used from v0.6 onwards ...

        """
        # Set attributes
        self._match = match
        self._getter_fct = (lambda x: x) if getter_fct is None else getter_fct
        self._setter_fct = (lambda x: x) if setter_fct is None else setter_fct
        self._setter_fct_args = tuple() if args is None else args
        self._setter_fct_kwargs = dict() if kwargs is None else kwargs
        self._allow_none = allow_none

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self._getter_fct(instance.__dict__[self._name])

    def __set__(self, instance, val):

        # Bypass None value if it's the case
        if self._allow_none and val is None:
            instance.__dict__[self._name] = val

        # Test match
        else:
            if isinstance(val, self._match):
                match_tuple = val
            else:
                raise DvasError(f'Bad type while assignment of {val}. ' +
                                f'Expected {self._match}. Received {type(val)}')

            # Apply setter function
            try:
                instance.__dict__[self._name] = self._setter_fct(
                        match_tuple, *self._setter_fct_args, **self._setter_fct_kwargs
                )
            except (KeyError, AttributeError) as second_error:
                raise TypeError('Error while apply setter function') from second_error

    def __set_name__(self, instance, name):
        """Attribute name setter"""
        self._name = name


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

    # UTC case
    if utc:
        try:
            out = to_datetime(val).to_pydatetime()
            if out.tzinfo is None:
                raise DvasError(f'tzinfo is None for {val}')

            if out.utcoffset().total_seconds() != 0:
                raise DvasError(f'Non-UTC time zone for {val}')

        except (ValueError, AssertionError) as first_error:
            raise TypeError(f"Bad datetime format for '{val}'") from first_error

    # Non UTC case
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


def get_class_public_attr(obj):
    """Get public attributes from an object

    Returns:
        dict

    """

    out = {
        attr: val for attr, val in getmembers(obj)
        if not attr.startswith('_') and not isroutine(val)
    }

    return out


class AttrDict(dict):
    """Dictionary keys like an attribute

    Note:
        `Source code
        <https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute>`__

    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
