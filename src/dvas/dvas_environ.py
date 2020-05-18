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

# Define package path
package_path = Path(__file__).parent


def set_path(value):
    """Test and set input argument into pathlib.Path object.

    Args:
        value (`obj`): Argument to be tested

    Returns:
        patlib.Path

    """
    try:
        (out := Path(value)).exists()

    except (TypeError, OSError):
        raise TypeError('Not compatible with system path')

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
    orig_data_path = TypedProperty((Path, str), set_path)
    #: pathlib.Path: Config dir path
    config_dir_path = TypedProperty((Path, str), set_path)
    #: pathlib.Path: Local db dir path
    local_db_path = TypedProperty((Path, str), set_path)
    #: pathlib.Path: DVAS output dir path
    output_path = TypedProperty((Path, str), set_path)

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

path_var = GlobalPathVariablesManager()
