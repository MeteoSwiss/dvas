"""Logger module

Created May 2020, L. Modolo - mol@meteoswiss.ch

Note:
    Values in LOGGER_NAME can be used as logger into LogManager context
    manager ('.' in LOGGER_NAME value must be replaced by '_')

"""

# Python external packages and modules import
from _datetime import datetime
import logging
from logging import StreamHandler, FileHandler

# Current package import
from .dvas_environ import glob_var, path_var
from .dvas_helper import ContextDecorator


# Define logger names
LOGGER_NAME = [
    'localdb',
    'localdb.insert',
    'localdb.select',
    'rawcsv',
    'rawcsv.load',
    'data',
    'data.calc'
]


def get_logger(name):
    """Get logger"""

    # Test logger name existence
    if not(name in LOGGER_NAME):
        raise ValueError("Unknown logger name '{}'".format(name))

    return logging.getLogger(name)


class DeltaTimeFormatter(logging.Formatter):
    """Delta time formatter

    `Source code`_

    .. _Source code:
        https://stackoverflow.com/questions/25194864/python-logging-time-since-start-of-program

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
    if glob_var.log_output == 'FILE':

        # Set log path
        log_path = path_var.output_path / 'log'
        log_path.mkdir(mode=777, parents=True, exist_ok=True)
        log_file_path = log_path / glob_var.log_file_name

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
        logger = logging.getLogger(name)
        logger.setLevel(glob_var.log_level)
        logger.propagate = False
        logger.addHandler(handler)


def clear_log():
    """Function used to clear log"""

    # Erase handlers
    for name in LOGGER_NAME:
        logger = logging.getLogger(name)
        for hdl in logger.handlers:
            hdl.flush()
        logger.handlers = []


class LogManager(ContextDecorator):
    """Logging context manager"""

    def __enter__(self):
        init_log()

    def __exit__(self, typ, value, traceback):
        clear_log()


# Add logger names to module locals()
for log_nm in LOGGER_NAME:

    # Replace '.' by '_' for camel case syntax
    log_var_nm = log_nm.replace('.', '_')

    # Add to locals()
    locals().update({log_var_nm: get_logger(log_nm)})
