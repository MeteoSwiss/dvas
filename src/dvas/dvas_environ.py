"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import Python packages and module
import os
from pathlib import Path
from re import compile
from abc import ABC, ABCMeta, abstractmethod
from contextlib import contextmanager
from pampy import match as pmatch
from pampy.helpers import Union, Iterable, Any
from yaconfigobject import Config

# Import current package's modules
from . import __name__ as pkg_name
from . import package_path
from .dvas_helper import SingleInstanceMetaClass
from .dvas_helper import TypedProperty as TProp
from .dvas_helper import check_path
from . import __name__ as pkg_name


# Define
CONFIG = Config(
    name='dvas_config.yaml',
    paths=[(package_path / 'etc').as_posix()]
)


class ABCSingleInstanceMeta(ABCMeta, SingleInstanceMetaClass):
    """Child Meteclass from ABCMeta and SingleInstanceMetaClass"""


class VariableManager(ABC, metaclass=ABCSingleInstanceMeta):
    """Class to manage variables"""

    def __init__(self):
        """Constructor"""

        # Check attr_def
        try:
            assert isinstance(self.attr_def, list)
            assert all([
                pmatch({'name': Any, 'default': Any}, arg, True, default=False)
                for arg in self.attr_def
            ])

        except AssertionError:
            "Error in matching 'attr_def' pattern"

        # Set attributes
        self.set_attr()

    @property
    @abstractmethod
    def attr_def(self):
        """Class attributes definition"""
        pass

    def set_attr(self):
        """Set attribute from attr_def. Try first to get attribute value from
        environment. All attributes can be defined in environment variables
        using <package name>_<attribute name> in upper case."""
        for arg in self.attr_def:
            setattr(
                self,
                arg['name'],
                os.getenv(
                    (pkg_name + '_' + arg['name']).upper(),
                    arg['default']
                )
            )

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


class GlobalPathVariablesManager(VariableManager):
    """Class to manage package's global directory path variables"""

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

    @property
    def attr_def(self):
        return [
            {'name': 'orig_data_path',
             'default': package_path / 'examples' / 'data'},
            {'name': 'config_dir_path',
             'default': package_path / 'examples' / 'config'},
            {'name': 'local_db_path',
             'default': Path('.') / 'dvas_db'},
            {'name': 'output_path',
             'default': Path('.') / 'output'}
        ]


class GlobalLoggingVariableManager(VariableManager):
    """Class used to manage package logging variables"""

    #: tuple: Allowed logging modes
    MODES = ('FILE', 'CONSOLE')

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

    @property
    def attr_def(self):
        return [
            {'name': 'log_mode',
             'default': 'CONSOLE'},
            {'name': 'log_file_name',
             'default': pkg_name},
            {'name': 'log_level',
             'default': 'INFO'},
        ]


class GlobalPackageVariableManager(VariableManager):
    """Class used to manage package global variables"""

    #: int: Config regexp generator limit. Default to 10000.
    config_gen_max = TProp(int, lambda x: int(x))
    #: str: Config regex generator group function separator. Default to '$'.
    config_gen_grp_sep = TProp(
        TProp.re_str_choice([r'\$', r'\%']),
        lambda *x: x[0]
    )
    #: list of str: Config file allowed extensions. Default to ['yml', 'yaml']
    config_file_ext = TProp(Iterable[str], lambda x: tuple(x))

    @property
    def attr_def(self):
        return [
            {'name': 'config_gen_max',
             'default': 10000},
            {'name': 'config_gen_grp_sep',
             'default': '$'},
            {'name': 'config_file_ext',
             'default': ['yml', 'yaml']}
        ]


#: GlobalPathVariablesManager: Global variable containing directory path values
path_var = GlobalPathVariablesManager()

#: GlobalLoggingVariableManager: Global variable containing log package variables
log_var = GlobalLoggingVariableManager()

#: GlobalPackageVariableManager: Global variable containing global package variables
glob_var = GlobalPackageVariableManager()
