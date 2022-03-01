"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Logging management

"""

# Python external packages and modules import
import logging
from logging import StreamHandler, FileHandler
from datetime import datetime
import inspect
from functools import wraps
from pampy.helpers import Union

# Current package import
from .helper import TypedProperty as TProp
from .helper import TimeIt
from .environ import path_var
from .errors import LogDirError
from . import __name__ as pkg_name


# Define logger names
LOGGER_NAME = [
    'localdb',  # DB stuff
    'rawcsv',  # I/O stuff
    'data',  # Data sub-module
    'plots',  # Plots sub-module
    'tools',  # Tools sub-module
    'general',  # Intended for anything not inside a specific sub-module
    'recipes',  # For high-level dvas-recipes logging
]


class DeltaTimeFormatter(logging.Formatter):
    """Delta time formatter

    Note:
       Adapted from `StackOverflow.
       <https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program>`__
       Author: Keith

    """
    def format(self, record):
        duration = datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S.%f")[:-3]
        return super().format(record)


class LogManager:
    """Class for log management"""

    #: tuple: Allowed logging modes
    _LEVEL_DICT = {
        'DEBUG': 'DEBUG',
        'D': 'DEBUG',
        'INFO': 'INFO',
        'I': 'INFO',
        'WARNING': 'WARNING',
        'WARN': 'WARNING',
        'W': 'WARNING',
        'ERROR': 'ERROR',
        'E': 'ERROR',
    }

    log_mode = TProp(
        Union[bool, int],
        setter_fct=lambda x: int(x) if (0 <= x and x <= 3) or (isinstance(x, bool)) else 0
    )
    """str: Log output mode. Defaults to 1.
        No log: False|0
        Log to file only: True|1
        Log to file + console: 2
        Log to console only: 3
    """

    #: str: Log level. Default to 'INFO'
    log_level = TProp(
        TProp.re_str_choice(list(_LEVEL_DICT.keys()), ignore_case=True),
        setter_fct=lambda x, *args: args[0][x[0].upper()],
        args=(_LEVEL_DICT,)
    )

    def __init__(self, mode, level):
        """Args:
            mode (int): Log output mode.
            level (str): Log level.
        """

        # Set log attr
        self.log_mode = mode
        self.log_level = level

        # Reset logger
        self.clear_log()

    def init_log(self):
        """Function used to initialize logger"""

        # Set handler
        if self.log_mode == 0:
            return  # Skip init
        elif self.log_mode == 1:
            handlers = [self.get_file_handler()]
        elif self.log_mode == 2:
            handlers = [self.get_file_handler(), self.get_console_handler()]
        elif self.log_mode == 3:
            handlers = [self.get_console_handler()]

        # Set datetime formatter
        formatter = DeltaTimeFormatter(
            '%(delta)s|%(levelname)s|%(name)s|%(message)s'
        )
        for hdl in handlers:
            hdl.setFormatter(formatter)

        # Add handler to all logger
        for name in LOGGER_NAME:
            logger = self.get_logger(name)
            logger.setLevel(self.log_level)
            logger.propagate = False
            for hdl in handlers:
                logger.addHandler(hdl)
            logger.disabled = False

        # All done. Let's start logging !
        logger = self.get_logger('general')
        logger.disabled = False
        logger.info(
            'This %s log was started on %s.',
            pkg_name,
            datetime.now().strftime('%Y-%m-%d at %H:%M:%S.%f')
        )

    @staticmethod
    def get_console_handler():
        """Return the console handler"""
        return StreamHandler()

    @staticmethod
    def get_file_handler():
        """Return the log file handler"""

        # Test
        if path_var.output_path is None:
            # TODO
            #  Detail exception
            raise Exception()

        # Set log path
        log_path = path_var.output_path / 'logs'
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            # Set user read/write permission
            log_path.chmod(log_path.stat().st_mode | 0o600)
        except (OSError,) as exc:
            raise LogDirError(f"Error in creating '{log_path}' ({exc})") from exc

        log_file_path = log_path / f"{datetime.utcnow():%Y%m%dT%H%M%SZ}.log"

        # Create stream handler and set level
        return FileHandler(
            log_file_path.as_posix(), mode='w', encoding='utf-8'
        )

    @staticmethod
    def get_logger(name):
        """Get logger"""

        # Test logger name existence
        if name not in LOGGER_NAME:
            raise ValueError("Unknown logger name '{}'".format(name))

        out = logging.getLogger(name)
        out.disabled = True

        return logging.getLogger(name)

    @staticmethod
    def clear_log():
        """Function used to clear log"""

        # Erase handlers
        for name in LOGGER_NAME:
            logger = logging.getLogger(name)
            for hdl in logger.handlers:
                hdl.flush()
                hdl.close()
                logger.removeHandler(hdl)
            logger.disabled = True


def log_func_call(logger, time_it=False):
    """ Intended as a decorator that logs a function call the the log.
    The first part of the message containing the function name is at the 'INFO' level.
    The second part of the message containing the argument values is at the 'DEBUG' level.

    Args:
        logger (str): one of the loggers defined in dvas_logger.py, e.g.: gruan_logger
        time_it (bool, optional): whether to evaluate the decorated function execution time
            (and log it), or not. Default to False.

    Note:
        Adapted from `this post <https://stackoverflow.com/questions/218616>`__ on SO,
        in particular the reply from Kfir Eisner and Peter Mortensen.
        See also `this <https://docs.python.org/3/library/inspect.html#inspect.BoundArguments>`__.

    """

    def deco(func):
        """ This is the actual function decorator. """

        @wraps(func)  # This black magic is required for Sphinx to still pickup the func docstrings.
        def inner_deco(*args, **kwargs):
            """ The core function, where the magic happens. """

            # Extract all the arguments and named-arguments fed to the function.
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Assemble a log message witht he function name ...
            log_msg = 'Executing %s ...' % (func.__name__)
            # ... and log it at the INFO level.
            logger.info(log_msg)

            # Then get extra information about the arguments ...
            log_msg = '... with the following input: %s' % (str(dict(bound_args.arguments)))
            # ... and log it at the DEBUG level.
            logger.debug(log_msg)

            # Launch the actual function
            if time_it:
                with TimeIt(header_msg=func.__name__, logger=logger):
                    out = func(*args, **kwargs)
            else:
                out = func(*args, **kwargs)
            return out
        return inner_deco
    return deco


# Add logger to locals()
#: logging.logger: Local DB logger
localdb = LogManager.get_logger('localdb')

#: logging.logger: Raw CSV data logger
rawcsv = LogManager.get_logger('rawcsv')

#: logging.logger: Data logger
data = LogManager.get_logger('data')

#: logging.logger: plots logger
plots_logger = LogManager.get_logger('plots')

#: logging.logger: tools logger
tools_logger = LogManager.get_logger('tools')

#: logging.logger: general logger
general_logger = LogManager.get_logger('general')

# TODO: I am here defining a logger for the recipes, which strictly speaking live outside
# dvas. I think this is ok ... no ?
#: logging.logger: high-level recipe logger
recipes_logger = LogManager.get_logger('recipes')
