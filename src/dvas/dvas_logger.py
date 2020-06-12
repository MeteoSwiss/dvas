"""Logger module

Created May 2020, L. Modolo - mol@meteoswiss.ch

Note:
    Values in LOGGER_NAME can be used as logger into LogManager context
    manager ('.' in LOGGER_NAME value must be replaced by '_')

"""

# Python external packages and modules import
import logging
from logging import StreamHandler, FileHandler
from datetime import datetime

# Current package import
from .dvas_environ import glob_var, path_var
from .dvas_helper import ContextDecorator


# Define logger names
LOGGER_NAME = [
    'localdb',
    'rawcsv',
    'data',
    'plot',
]


def get_logger(name):
    """Get logger"""

    # Test logger name existence
    if name not in LOGGER_NAME:
        raise ValueError("Unknown logger name '{}'".format(name))

    out = logging.getLogger(name)
    out.disabled = True

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
        logger = get_logger(name)
        logger.setLevel(glob_var.log_level)
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
plot = get_logger('plot')
