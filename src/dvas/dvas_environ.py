"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os
from pathlib import Path
from contextlib import contextmanager

# Import current package's modules
from .dvas_helper import SingleInstanceMetaClass
from .dvas_helper import TypedProperty
from . import __name__ as pkg_name

# Define package path
package_path = Path(__file__).parent


def set_path(value, exist_ok=False):
    """Test and set input argument into pathlib.Path object.

    Args:
        value (`obj`): Argument to be tested
        exist_ok (bool, optional): If True check existence. Otherwise
            create path. Default to False.

    Returns:
        patlib.Path

    """
    try:
        if exist_ok is True:
            assert (out := Path(value)).exists() is True
        else:
            (out := Path(value)).mkdir(mode=777, parents=True, exist_ok=True)

    except AssertionError:
        raise TypeError(f"Path '{out}' does not exist")

    except (TypeError, OSError, FileNotFoundError):
        raise TypeError(f"Path '{out}' is not compatible with system path")

    return out


class GlobalPathVariablesManager(metaclass=SingleInstanceMetaClass):
    """Class to manage package's global directory path variables"""

    # Set class constant attributes
    CST = [
        {'name': 'orig_data_path',
         'default': package_path / 'examples' / 'data',
         'os_nm': 'DVAS_ORIG_DATA_PATH'},
        {'name': 'config_dir_path',
         'default': package_path / 'examples' / 'config',
         'os_nm': 'DVAS_CONFIG_DIR_PATH'},
        {'name': 'local_db_path',
         'default': Path('.') / 'dvas_db',
         'os_nm': 'DVAS_LOCAL_DB_PATH'},
        {'name': 'output_path',
         'default': Path('.') / 'output',
         'os_nm': 'DVAS_OUTPUT_PATH'}
    ]

    #: pathlib.Path: Original data path
    orig_data_path = TypedProperty(
        (Path, str), set_path, kwargs={'exist_ok': True}
    )
    #: pathlib.Path: Config dir path
    config_dir_path = TypedProperty(
        (Path, str), set_path, kwargs={'exist_ok': True}
    )
    #: pathlib.Path: Local db dir path
    local_db_path = TypedProperty(
        (Path, str), set_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: DVAS output dir path
    output_path = TypedProperty(
        (Path, str), set_path, kwargs={'exist_ok': False}
    )

    def __init__(self):
        """Constructor"""
        # Load os environment values
        self.load_os_environ()

    def load_os_environ(self):
        """Load from OS environment variables"""
        for arg in self.CST:
            setattr(
                self,
                arg['name'],
                os.getenv(arg['os_nm'], arg['default'])
            )

    @contextmanager
    def set_many_attr(self, items):
        """Context manager to set temporarily many global variables.

        Args:
            items (dict): Temporarily global variables.
                keys: instance attribute name. values: temporarily set values.

        Examples:
            >>>from dvas.dvas_environ import path_var
            >>>with path_var.set_many_attr({})

        """

        # Set new values and save old values
        old_items = {}
        for key, val in items.items():
            old_items.update({key: getattr(self, key)})
            setattr(self, key, val)
        yield

        # Restore old values
        for key, val in old_items.items():
            setattr(self, key, val)


#: GlobalPathVariablesManager: Global variable containing directory path values
path_var = GlobalPathVariablesManager()


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


class GlobalPackageVariableManager(metaclass=SingleInstanceMetaClass):
    """Class used to manage package global variables"""

    # Set class constant attributes
    CST = [
        {'name': 'log_output',
         'default': 'CONSOLE',
         'os_nm': 'DVAS_LOG_OUTPUT'},
        {'name': 'log_file_name',
         'default': pkg_name + '.log',
         'os_nm': 'DVAS_LOG_FILE_NAME'},
        {'name': 'log_level',
         'default': 'INFO',
         'os_nm': 'DVAS_LOG_LEVEL'},
    ]

    #: str: Logger output type
    log_output = TypedProperty(
        str, check_str, kwargs={'choices': ['FILE', 'CONSOLE']}
    )
    log_file_name = TypedProperty(str)  #TODO add check re \w
    log_level = TypedProperty(
        str, check_str,
        kwargs={'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
    )

    def __init__(self):
        """Constructor"""
        self.load_os_environ()

    def load_os_environ(self):
        """Load from OS environment variables"""
        for arg in self.CST:
            setattr(
                self,
                arg['name'],
                os.getenv(arg['os_nm'], arg['default'])
            )


glob_var = GlobalPackageVariableManager()
