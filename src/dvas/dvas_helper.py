"""

"""

# Import external packages and modules
from datetime import datetime
from functools import wraps
from abc import ABC, ABCMeta, abstractmethod
from peewee import PeeweeException
from inspect import getmodule


class SingleInstanceMetaClass(type):
    def __init__(self, name, bases, dic):
        self.__single_instance = None
        super().__init__(name, bases, dic)

    def __call__(cls, *args, **kwargs):
        if cls.__single_instance:
            return cls.__single_instance
        single_obj = cls.__new__(cls)
        single_obj.__init__(*args, **kwargs)
        cls.__single_instance = single_obj
        return single_obj


class RequiredAttrMeta(ABCMeta):
    """Meta class for requiring specific attribute to abstract parent class"""
    REQUIRED_ATTRIBUTES = []

    def __call__(cls, *args, **kwargs):
        obj = super(RequiredAttrMeta, cls).__call__(*args, **kwargs)

        for attr_name, dtype in obj.REQUIRED_ATTRIBUTES.items():

            if not hasattr(obj, attr_name):
                errstr = "required attribute '{}' not defined in class {}"
                errmsg = errstr.format(attr_name, obj)
                raise ValueError(errmsg)

            if not isinstance(getattr(obj, attr_name), dtype):
                errstr = "required attribute '{}' bad set in class {}"
                errmsg = errstr.format(attr_name, obj)
                raise ValueError(errmsg)

        return obj


class ContextDecorator(ABC):
    """Use this class as superclass of a context manager to convert it into
    a decorator.
    """

    @abstractmethod
    def __init__(self):
        self._func = None

    def __call__(self, func):

        @wraps(func)
        def decorated(*args, **kwargs):
            self.func = func
            with self:
                return func(*args, **kwargs)
        return decorated

    @property
    def func(self):
        """callable: Contain the decorated function. Set automatically only
        when class is used as decorated. Default to None.
        """
        return self._func

    @func.setter
    def func(self, val):
        self._func = val

    @abstractmethod
    def __enter__(self):
        """Enter method"""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method"""


class TimeIt(ContextDecorator):
    """Code elapsed time calculator context manager/decorator"""

    def __init__(self, header_msg=''):
        super().__init__()
        self._head_msg = header_msg

    def __enter__(self):
        self._start = datetime.now()

        # Set msg header
        if (self.func is None) and (self._head_msg is ''):
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
        delta = datetime.now() - self._start
        print(f'{self._head_msg}: {delta}', flush=True)


class DBAccess(ContextDecorator):
    """Data base context decorator"""

    def __init__(self, db, close_by_exit=True):
        """
        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object
            close_by_exit (bool): Close by exiting
        """
        super().__init__()
        self._db = db
        self._close_by_exit = close_by_exit

    def __call__(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            with self as transaction:
                try:
                    return func(*args, **kwargs)
                except PeeweeException:
                    transaction.rollback()
                    return
        return decorated

    def __enter__(self):
        if self._db.is_closed():
            self._db.connect()
        self._transaction = self._db.atomic()
        return self._transaction

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._close_by_exit:
            self._db.close()


class DBAccessQ(ContextDecorator):
    """Data base context decorator"""

    def __init__(self, db):
        """
        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object
        """
        super().__init__()
        self._db = db

    def __call__(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            with self:
                try:
                    return func(*args, **kwargs)
                except PeeweeException as e:
                    print(e)

        return decorated

    def __enter__(self):
        self._db.start()
        self._db.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._db.close()
        self._db.stop()
