"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Logging management

"""

# Python external packages and modules import
import logging
from logging import StreamHandler, FileHandler
from datetime import datetime
import inspect

# Current package import
from .dvas_environ import log_var, path_var
from .dvas_helper import ContextDecorator


# Define logger names
LOGGER_NAME = [
    'localdb',
    'rawcsv',
    'data',
    'plot',
    'gruan',
]

def get_logger(name):
    """Get logger"""

    # Test logger name existence
    if name not in LOGGER_NAME:
        raise ValueError("Unknown logger name '{}'".format(name))

    out = logging.getLogger(name)
    out.disabled = True

    return logging.getLogger(name)

def log_func_call(logger):
    ''' Intended as a decorator that logs a function call the the log. The message is at the 'DEBUG'
    level.

    Args:
        logger (str): one of the loggers defined in dvas_logger.py, e.g.: gruan_logger

    Note:
        Adapted from
        `this post <https://stackoverflow.com/questions/218616/how-to-get-method-parameter-names>`__
        on SO, in particular the reply from Kfir Eisner and Peter Mortensen.

    '''

    def deco(func):
        ''' This is the actual function decorator. '''

        def inner_deco(*args, **kwargs):
            ''' The core function, where the magic happens. '''

            # Extract all the arguments and named-arguments fed to the function.
            bound_args = inspect.signature(func).bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Assemble a proper log message
            log_msg = 'Executing %s ' % (func.__name__)
            log_msg += 'with the following input: %s' % (str(dict(bound_args.arguments)))

            # Log the message
            logger.debug(log_msg)

            # Launch the actual function
            return func(*args, **kwargs)
        return inner_deco
    return deco


class DeltaTimeFormatter(logging.Formatter):
    """Delta time formatter

    Note:
       Adapted from `StackOverflow. <https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program>`__
       Author: Keith

    """
    def format(self, record):
        duration = datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S.%f")[:-3]
        return super().format(record)


def init_log():
    """Function used to initialize logger"""

    # Reset logger
    clear_log()

    # Select mode
    if log_var.log_mode == 'FILE':

        # Set log path
        log_path = path_var.output_path / 'log'
        try:
            log_path.mkdir(parents=True, exist_ok=True)
            # Set user read/write permission
            log_path.chmod(log_path.stat().st_mode | 0o600)
        except (OSError,) as exc:
            raise LogDirError(f"Error in creating '{log_path}' ({exc})")

        log_file_path = log_path / log_var.log_file_name

        # Create stream handler and set level
        handler = FileHandler(
            log_file_path.as_posix(), mode='w', encoding='utf-8'
        )

    else:
        # Create stream handler and set level
        handler = StreamHandler()

    # Set formatter
    DeltaTimeFormatter()

    formatter = DeltaTimeFormatter(
        '%(delta)s|%(levelname)s|%(name)s|%(message)s'
    )
    handler.setFormatter(formatter)

    # Add handler to all logger
    for name in LOGGER_NAME:
        logger = get_logger(name)
        logger.setLevel(log_var.log_level)
        logger.propagate = False
        logger.addHandler(handler)
        logger.disabled = False


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


class LogManager(ContextDecorator):
    """Logging context manager"""

    def __enter__(self):
        init_log()

    def __exit__(self, typ, value, traceback):
        clear_log()


# Add logger to locals()
#: logging.logger: Local DB logger
localdb = get_logger('localdb')
#: logging.logger: Raw CSV data logger
rawcsv = get_logger('rawcsv')
#: logging.logger: Data logger
data = get_logger('data')
#: logging.logger: Plot logger
plot_logger = get_logger('plot')
#: logging.logger: GRUAN logger
gruan_logger = get_logger('gruan')


class LogDirError(Exception):
    """Exception for error in creating log directory"""
