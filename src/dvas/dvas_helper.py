"""

"""

# Import external packages and modules
from datetime import datetime
from functools import wraps
from abc import ABC, abstractmethod
from peewee import PeeweeException


class ContextDecorator(ABC):
    """Use this class as superclass of a context manager to convert it into
    a decorator.
    """
    def __call__(self, func):
        @wraps(func)
        def decorated(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return decorated

    @abstractmethod
    def __enter__(self):
        """Enter method"""

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit method"""


class TimeIt(ContextDecorator):
    """Code elapsed time calculator context manager/decorator"""

    def __enter__(self):
        self._start = datetime.now()

    def __exit__(self, exc_type, exc_val, exc_tb):
        delta = datetime.now() - self._start
        print(
            'Elapsed time: {} sec'.format(delta.total_seconds()),
            flush=True
        )


class DBAccess(ContextDecorator):
    """Data base context decorator"""

    def __init__(self, db, close_by_exit=True):
        """
        Args:
            db (peewee.SqliteDatabase): PeeWee Sqlite DB object
            close_by_exit (bool): Close by exiting
        """
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
            close_by_exit (bool): Close by exiting
        """
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
