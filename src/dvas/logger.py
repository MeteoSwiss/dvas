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

# Current package import
from .helper import TypedProperty as TProp
from .helper import TimeIt
from .environ import path_var
from .errors import LogDirError
from . import __name__ as pkg_name

# Define logger names. These should be the neame of the different sub-packages.
LOGGER_NAMES = ['dvas', 'dvas_recipes']


class DeltaTimeFormatter(logging.Formatter):
    """ Delta time formatter

    Note:
       Adapted from `StackOverflow.
       <https://stackoverflow.com/questions/25194864>`__
       Author: Keith

    """

    def format(self, record):
        duration = datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S.%f")[:-3]
        return super().format(record)


class DvasFormatter(logging.Formatter):
    """ The custom logging formatter class for dvas. To handle time deltas AND colors. """

    def __init__(self, colors=False):
        """ Init function.

        Args:
            colors (bool, optional): if True, will add colors to the log message via ANSI codes.
                Defaults to False.
        """

        self._colors = colors

        # Call the super init
        super().__init__()

    def log_msg(self, level=logging.INFO):
        """ Return the dvas log message canvas, possibly with colors and stuff.

        Args:
            level (logging.lvl, optional): the log level, e.g. logging.INFO, logging.ERROR, ...

        Return:
            DeltaTimeFormatter: the formatted log message canvas.
        """

        msg = '%(delta)s|$BOLD$COLOR%(levelname)s$RESET|%(name)s| %(message)s'

        # Cases when I want some colors
        if self._colors:
            if level == logging.DEBUG:
                msg = msg.replace('$COLOR', '\x1b[36;20m')
                msg = msg.replace('$BOLD', '')
                msg = msg.replace('$RESET', '\033[0m')
            elif level == logging.INFO:
                msg = msg.replace('$COLOR', '\x1b[32;20m')
                msg = msg.replace('$BOLD', '')
                msg = msg.replace('$RESET', '\033[0m')
            elif level == logging.WARNING:
                msg = msg.replace('$COLOR', '\x1b[33;20m')
                msg = msg.replace('$BOLD', '')
                msg = msg.replace('$RESET', '\033[0m')
            elif level == logging.ERROR:
                msg = msg.replace('$COLOR', '\x1b[31;20m')
                msg = msg.replace('$BOLD', '')
                msg = msg.replace('$RESET', '\033[0m')
            elif level == logging.CRITICAL:
                msg = msg.replace('$COLOR', '\x1b[31;20m')
                msg = msg.replace('$BOLD', '\033[1m')
                msg = msg.replace('$RESET', '\033[0m')
            else:
                msg = msg.replace('$COLOR', '')
                msg = msg.replace('$BOLD', '')
                msg = msg.replace('$RESET', '')
        else:
            msg = msg.replace('$COLOR', '')
            msg = msg.replace('$BOLD', '')
            msg = msg.replace('$RESET', '')

        return DeltaTimeFormatter(msg)

    def format(self, record):
        """ Format the log message as required """

        out = self.log_msg(level=record.levelno).format(record)
        # Allow users to add colors to the text message only ...
        if self._colors:
            out = out.replace('$SFLASH', '\x1b[35;20m')
            out = out.replace('$EFLASH', '\033[0m')
        else:
            out = out.replace('$SFLASH', '')
            out = out.replace('$EFLASH', '')
        return out


def apply_dvas_formatter(handler, colors=False):
    """ A small routine responsible for apply a custom formatter to a given logging handler. """

    handler.setFormatter(DvasFormatter(colors=colors))

    return handler


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
        'CRITICAL': 'CRITICAL',
        'C': 'CRITICAL'
    }

    log_mode = TProp(bool | int,
                     setter_fct=lambda x: int(x) if (0 <= x <= 3) or (isinstance(x, bool)) else 0)

    """str: Log output mode. Defaults to 1.
        No log: False|0
        Log to file only: True|1
        Log to file + console: 2
        Log to console only: 3
    """

    #: str: Log level. Default to 'INFO'
    log_level = TProp(str, setter_fct=lambda x, *args: args[0][x[0].upper()],
                      args=(_LEVEL_DICT,))

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
        if self.log_mode == 1:
            handlers = [apply_dvas_formatter(self.get_file_handler(), colors=False)]
        elif self.log_mode == 2:
            handlers = [apply_dvas_formatter(self.get_file_handler(), colors=False),
                        apply_dvas_formatter(self.get_console_handler(), colors=True)]
        elif self.log_mode == 3:
            handlers = [apply_dvas_formatter(self.get_console_handler(), colors=True)]
        else:
            return  # Skip init

        # Add handler to all logger
        for name in LOGGER_NAMES:
            logger = self.get_logger(name)
            logger.setLevel(self.log_level)
            logger.propagate = False
            for hdl in handlers:
                logger.addHandler(hdl)
            logger.disabled = False

        # All done. Let's start logging !
        logger = logging.getLogger(__name__)
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
            raise LogDirError('output_path is None.')

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
        if name not in LOGGER_NAMES:
            raise ValueError(f"Unknown logger name: {name}")

        out = logging.getLogger(name)
        out.disabled = True

        return logging.getLogger(name)

    @staticmethod
    def clear_log():
        """Function used to clear log"""

        # Erase handlers
        for name in LOGGER_NAMES:
            logger = logging.getLogger(name)
            for hdl in logger.handlers:
                hdl.flush()
                hdl.close()
                logger.removeHandler(hdl)
            logger.disabled = True


def log_func_call(logger, time_it=False, level='info'):
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
            log_msg = f'Executing {func.__name__} ...'
            # ... and log it at the appropriate level
            getattr(logger, level)(log_msg)

            # Then get extra information about the arguments ...
            log_msg = f'... with the following input: {dict(bound_args.arguments)}'
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
