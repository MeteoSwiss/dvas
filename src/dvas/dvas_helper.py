"""
This module contains all helper class and functions used in the package.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import external packages and modules
from pathlib import Path
import re
import platform
from datetime import datetime
from functools import wraps, reduce
from abc import ABC, ABCMeta, abstractmethod
from inspect import getmodule
from operator import getitem
import oschmod
from peewee import PeeweeException


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
        `Source code <https://www.pythonprogramming.in/singleton-class-using-metaclass-in-python.html>`__

    """

    def __init__(cls, name, bases, dic):
        """Constructor"""
        cls.__single_instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        """Class __call__ method"""
        if cls.__single_instance:
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

            if not isinstance(getattr(obj, attr_name), dtype):
                errmsg = (
                    f"required attribute '{attr_name}' bad set in class {obj}"
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


class DBAccess(ContextDecorator):
    """Local SQLite data base context decorator"""

    def __init__(self, db, close_by_exit=True):
        """Constructor

        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object
            close_by_exit (bool): Close DB by exiting context manager.
                Default to True

        """
        super().__init__()
        self._db = db
        self._close_by_exit = close_by_exit
        self._transaction = None

    def __call__(self, func):
        """Overwrite class __call__ method

        Args:
            func (callable): Decorated function

        """
        @wraps(func)
        def decorated(*args, **kwargs):
            with self as transaction:
                try:
                    out = func(*args, **kwargs)
                except PeeweeException:
                    transaction.rollback()
                    out = None
                return out
        return decorated

    def __enter__(self):
        """Class __enter__ method"""
        self._db.connect(reuse_if_open=True)
        self._transaction = self._db.atomic()
        return self._transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""
        if self._close_by_exit:
            self._db.close()


#TODO
# Test this db access context manager to possible speed up data select/insert
class DBAccessQ(ContextDecorator):
    """Data base context decorator"""

    def __init__(self, db):
        """Constructor

        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object

        """
        super().__init__()
        self._db = db

    def __call__(self, func):
        """Overwrite class __call__ method"""
        @wraps(func)
        def decorated(*args, **kwargs):
            with self:
                try:
                    return func(*args, **kwargs)
                except PeeweeException as exc:
                    print(exc)

        return decorated

    def __enter__(self):
        """Class __enter__ method"""
        self._db.start()
        self._db.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Class __exit__ method"""
        self._db.close()
        self._db.stop()


class TypedProperty:
    """Typed property class

    Note:
        `Source code <https://stackoverflow.com/questions/34884947/understanding-a-python-descriptors-example-typedproperty>`__

    """
    def __init__(self, typ, setter_fct=None, args=None, kwargs=None):
        """Constructor

        Args:
            typ (type or tuple of type): Data type
            setter_fct: Function applied before assign value in setter method.
                The function can include special check and raises.
            args (tuple): setter function args
            kwargs (dict): setter function kwargs
        """
        # Set attributes
        self._typ = typ
        self._setter_fct = (lambda x: x) if setter_fct is None else setter_fct
        self._setter_fct_args = tuple() if args is None else args
        self._setter_fct_kwargs = dict() if kwargs is None else kwargs

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__[self._name]

    def __set__(self, instance, val):
        # Test type
        if isinstance(val, self._typ) is False:
            raise TypeError(f'{self._name} <- val bad type')

        # Set val
        instance.__dict__[self._name] = self._setter_fct(
            val, *self._setter_fct_args, **self._setter_fct_kwargs
        )

    def __set_name__(self, instance, name):
        """Attribute name setter"""
        self._name = name


def check_str(val, choices, case_sens=False):
    """Function used to check str

    Args:
        val (str): Value to check
        choices (list of str): Allowed choices for value
        case_sens (bool): Case sensitivity

    Return:
        str

    Raises:
        TypeError if value is not in choises

    """

    if case_sens is False:
        choices_mod = list(map(str.lower, choices))
        val_mod = val.lower()
    else:
        choices_mod = choices
        val_mod = val

    try:
        assert val_mod in choices_mod
        idx = choices_mod.index(val_mod)
    except (StopIteration, AssertionError):
        raise TypeError(f"{val} not in {choices}")

    return choices[idx]


def check_list_str(val, choices=None, case_sens=False):
    """Test and set input argument into list of str.

    Args:
        val (list of str): Input
        choices (list of str):
        case_sens (bool):

    Returns:
        str

    """
    try:
        if choices:
            for arg in val:
                check_str(arg, choices, case_sens=case_sens)
        else:
            assert all([isinstance(arg, str) for arg in val]) is True

    except AssertionError:
        raise TypeError(f"{val} is not a list of str")

    except TypeError as _:
        raise TypeError(f"{val} item are not all in {choices}")

    return val


def get_by_path(root, items):
    """Access a nested object in root by item sequence.

    Args:
        root (dict|list|class): Object to access
        items (list): Item sequence

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

    """
    if isinstance(root, (dict, list)):
        out = reduce(getitem, items, root)
    else:
        out = reduce(getattr, items, root)

    return out


def check_path(value, exist_ok=False):
    """Test and set input argument into pathlib.Path object.

    Args:
        value (`obj`): Argument to be tested
        exist_ok (bool, optional): If True check existence.
            Otherwise create path. Default to False. The user must have
            read and write access to the path.

    Returns:
        pathlib.Path

    Raises:
        - TypeError: In case of path does not exist, or

    """

    # Create or test existence
    if exist_ok is True:
        try:
            assert (out := Path(value)).exists() is True
        except AssertionError:
            raise TypeError(f"Path '{out}' does not exist")

    else:
        try:
            (out := Path(value)).mkdir(parents=True, exist_ok=True)
        except (TypeError, OSError, FileNotFoundError):
            raise TypeError(f"Can not create '{out}'")

    # Set read/write access
    try:
        if platform.system() != 'Windows':
            oschmod.set_mode(out, "u+rw")

    except Exception:
        raise TypeError(f"Can not set '{out}' to read/write access.")

    return out
