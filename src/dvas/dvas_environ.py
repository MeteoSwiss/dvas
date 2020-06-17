"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os
import platform
from pathlib import Path
from contextlib import contextmanager
import oschmod

# Import current package's modules
from .dvas_helper import SingleInstanceMetaClass
from .dvas_helper import TypedProperty
from .dvas_helper import check_str, check_list_str
from . import __name__ as pkg_name

# Define package path
package_path = Path(__file__).parent


def set_path(value, exist_ok=False):
    """Test and set input argument into pathlib.Path object.

    Args:
        value (pathlib.Path, str): Argument to be tested
        exist_ok (bool, optional): If True check existence. Default to False.

    Returns:
        pathlib.Path

    Raises:
        TypeError: In case if path does not exist falls exist_ok is True

    """

    # Test existence
    if exist_ok is True:
        try:
            assert (out := Path(value)).exists() is True, (
                f"Path '{value}' does not exist"
            )
        except AssertionError as ass:
            raise TypeError(ass)
    else:
        try:
            out = Path(value)
        except (TypeError, OSError):
            raise TypeError(f"Bad path name for '{value}'")

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
            ::

                from dvas.dvas_environ import path_var
                with path_var.set_many_attr({})

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
        {'name': 'config_gen_max',
         'default': 10000},
        {'name': 'config_gen_grp_sep',
         'default': '$'},
        {'name': 'config_file_ext',
         'default': ['yml', 'yaml']}
    ]

    #: str: Log output type, Default to 'CONSOLE'
    log_output = TypedProperty(
        str, check_str, args=(['FILE', 'CONSOLE'],)
    )
    #: str: Log output file name. Default to 'dvas.log'
    log_file_name = TypedProperty(str)  #TODO add check re \w
    #: str: Log level. Default to 'INFO'
    log_level = TypedProperty(
        str, check_str,
        args=(['DEBUG', 'INFO', 'WARNING', 'ERROR'],)
    )
    #: int: Config regexp generator limit. Default to 10000.
    config_gen_max = TypedProperty(int)
    #: str: Config regex generator group function separator. Default to '$'.
    config_gen_grp_sep = TypedProperty(
        str, check_str, args=(['$', '%', '#'],)
    )
    #: list if str: Config file allowed extensions. Default to ['yml', 'yaml']
    config_file_ext = TypedProperty(
        list, check_list_str, kwargs={'choices': ['yml', 'yaml', 'txt']}
    )

    def __init__(self):
        """Constructor"""
        self.load_os_environ()

    def load_os_environ(self):
        """Load from OS environment variables"""
        for arg in self.CST:
            if 'os_nm' in arg.keys():
                setattr(
                    self,
                    arg['name'],
                    os.getenv(arg['os_nm'], arg['default'])
                )
            else:
                setattr(self, arg['name'], arg['default'])


glob_var = GlobalPackageVariableManager()
