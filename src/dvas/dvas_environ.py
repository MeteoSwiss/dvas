"""
This module contains the package's environment variables.

Created February 2020, L. Modolo - mol@meteoswiss.ch

"""

# Import Python packages and module
import os
from pathlib import Path
from re import compile
from contextlib import contextmanager
from pampy.helpers import Union, Iterable

# Import current package's modules
from .dvas_helper import SingleInstanceMetaClass
from .dvas_helper import TypedProperty as TProp
from .dvas_helper import check_path
from . import __name__ as pkg_name

# Define package path
package_path = Path(__file__).parent


class VariableManager():
    """Class to manage variables"""

    CST = []

    def __init__(self):
        """Constructor"""
        # Load os environment values
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

    @contextmanager
    def set_many_attr(self, items):
        """Context manager to set temporarily many global variables.

        Args:
            items (dict): Temporarily global variables.
                keys: instance attribute name. values: temporarily set values.
        except AssertionError as ass:
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


class GlobalPathVariablesManager(VariableManager, metaclass=SingleInstanceMetaClass):
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
    orig_data_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': True}
    )
    #: pathlib.Path: Config dir path
    config_dir_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': True}
    )
    #: pathlib.Path: Local db dir path
    local_db_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: DVAS output dir path
    output_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )


class GlobalLoggingVariableManager(VariableManager, metaclass=SingleInstanceMetaClass):
    """Class used to manage package logging variables"""

    #: tuple: Allowed logging modes
    MODES = ('FILE', 'CONSOLE')

    # Set class constant attributes
    CST = [
        {'name': 'log_mode',
         'default': 'CONSOLE',
         'os_nm': 'DVAS_LOG_MODE'},
        {'name': 'log_file_name',
         'default': pkg_name,
         'os_nm': 'DVAS_LOG_FILE_NAME'},
        {'name': 'log_level',
         'default': 'INFO',
         'os_nm': 'DVAS_LOG_LEVEL'},
    ]

    #: str: Log output mode, Default to 'CONSOLE'
    log_mode = TProp(
        TProp.re_str_choice(MODES, ignore_case=True),
        lambda *x: x[0].upper()
    )
    #: str: Log output file name. Default to 'dvas'
    log_file_name = TProp(compile(r'\w+'), lambda x: x + '.log')
    #: str: Log level. Default to 'INFO'
    log_level = TProp(
        TProp.re_str_choice(
            ['DEBUG', 'INFO', 'WARNING', 'ERROR'], ignore_case=True
        ),
        lambda *x: x[0].upper()
    )


class GlobalPackageVariableManager(VariableManager, metaclass=SingleInstanceMetaClass):
    """Class used to manage package global variables"""

    # Set class constant attributes
    CST = [
        {'name': 'config_gen_max',
         'default': 10000},
        {'name': 'config_gen_grp_sep',
         'default': '$'},
        {'name': 'config_file_ext',
         'default': ['yml', 'yaml']}
    ]

    #: int: Config regexp generator limit. Default to 10000.
    config_gen_max = TProp(int)
    #: str: Config regex generator group function separator. Default to '$'.
    config_gen_grp_sep = TProp(
        TProp.re_str_choice([r'\$', r'\%']),
        lambda *x: x[0]
    )
    #: list of str: Config file allowed extensions. Default to ['yml', 'yaml']
    config_file_ext = TProp(Iterable[str], lambda x: tuple(x))


#: GlobalPathVariablesManager: Global variable containing directory path values
path_var = GlobalPathVariablesManager()
#: GlobalLogingVariableManager: Global variable containing log package variables
log_var = GlobalLoggingVariableManager()
#: GlobalPackageVariableManager: Global variable containing global package variables
glob_var = GlobalPackageVariableManager()
