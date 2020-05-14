"""
This module contains all helper class and functions used in the package.

"""

# Import external packages and modules
from datetime import datetime
from pathlib import Path
from functools import wraps, reduce
from abc import ABC, ABCMeta, abstractmethod
from inspect import getmodule
from operator import getitem
from peewee import PeeweeException


class SingleInstanceMetaClass(type):
    """Metaclass to create single instance of class"""

    def __init__(cls, name, bases, dic):
        """Constructor"""
        cls.__single_instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        """Class __call method"""
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
    """Use this class as superclass of a context manager to convert it into
    a decorator.

    """

    @abstractmethod
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
    """Code elapsed time calculator context manager/decorator

    """

    def __init__(self, header_msg=''):
        """Constructor

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
        if self._db.is_closed():
            self._db.connect()
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
    """Typed property class"""
    def __init__(self, typ, val=None):
        if not (isinstance(val, typ) or val is None):
            raise TypeError()
        self._value = val
        self._typ = typ

    def __get__(self, instance, owner):
        return self._value

    def __set__(self, instance, val):
        if not isinstance(val, self._typ):
            raise TypeError()
        self._value = val

    def __set_name__(self, instance, name):
        """Attribute name setter"""
        self._name = name


class TypedPropertyPath(TypedProperty):
    """Special typed property class for manage Path attributes"""

    def __init__(self):
        """Constructor"""
        super().__init__((Path, str))

    def __set__(self, instance, val):
        """Overwrite __set__ method

        Args:
            instance (object):
            value (pathlib.Path or str): Directory path.
                Created if create is True.
        """
        if not isinstance(val, self._typ):
            raise TypeError(f"Expected class {self._typ}, got {type(val)}")

        # Convert to pathlib.Path
        val = Path(val)

        # Test OS compatibility
        try:
            val.exists()
        except OSError:
            raise TypeError(f"{val} not valid OS path name")

        self._value = val


def get_by_path(root, items):
    """Access a nested object in root by item sequence.

    Args:
        root (dict or list): Root list or dictionary
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
    return reduce(getitem, items, root)
