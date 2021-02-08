"""
Copyright (c) 2020-2021 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import Python packages and module
import os
from pathlib import Path
import re
from abc import ABC, ABCMeta, abstractmethod
from contextlib import contextmanager
from pampy import match as pmatch
from pampy.helpers import Union, Iterable, Any

# Import current package's modules
from .helper import SingleInstanceMetaClass
from .helper import TypedProperty as TProp
from .helper import check_path
from .helper import get_class_public_attr
from .errors import DvasError
from . import __name__ as pkg_name
from .hardcoded import PKG_PATH, PROC_PATH


class ABCSingleInstanceMeta(ABCMeta, SingleInstanceMetaClass):
    """Child Meteclass from ABCMeta and SingleInstanceMetaClass"""


class VariableManager(ABC, metaclass=ABCSingleInstanceMeta):
    """Class to manage variables"""

    def __init__(self):
        """Constructor"""

        # Check _attr_def
        try:
            assert isinstance(self._attr_def, list)
            assert all([
                pmatch(arg, {'name': Any, 'default': Any}, True, default=False)
                for arg in self._attr_def
            ])

        except AssertionError as first_error:
            raise DvasError("Error in matching '_attr_def' pattern") from first_error

        # Set attributes
        self.set_attr()

    def __str__(self):
        return '\n'.join(
            [f"{key}: {val}" for key, val in self.get_attr().items()]
        )

    def get_attr(self):
        """Return current attributes"""
        return get_class_public_attr(self)

    @property
    @abstractmethod
    def _attr_def(self):
        """Class attributes definition"""

    def set_attr(self):
        """Set attribute from _attr_def. Try first to get attribute value from
        environment. All attributes can be defined in environment variables
        using <package name>_<attribute name> in upper case."""
        for arg in self._attr_def:
            setattr(
                self,
                arg['name'],
                os.getenv(
                    (pkg_name + '_' + arg['name']).upper(),
                    arg['default']
                )
            )

    @contextmanager
    def protect(self):
        """Context manager to protect temporarily global variables.
        When existing the context manager, old values are restored.
        """

        # Get current attributes values
        old_attr = self.get_attr()

        # Yield context manager
        yield

        # Restore old values
        for key, val in old_attr.items():
            setattr(self, key, val)


class GlobalPathVariablesManager(VariableManager):
    """Class to manage package's global directory path variables"""

    #: pathlib.Path: Original data path
    orig_data_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: Config dir path
    config_dir_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: Local db dir path
    local_db_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: DVAS output dir path
    output_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: DVAS output dir path for plots
    plot_output_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': False}
    )
    #: pathlib.Path: plot styles dir path
    plot_style_path = TProp(
        Union[Path, str], check_path, kwargs={'exist_ok': True}
    )

    @property
    def _attr_def(self):
        return [
            {'name': 'orig_data_path',
             'default': PROC_PATH / 'data'},
            {'name': 'config_dir_path',
             'default': PROC_PATH / 'config'},
            {'name': 'local_db_path',
             'default': PROC_PATH / 'dvas_db'},
            {'name': 'output_path',
             'default': PROC_PATH / 'output'},
            {'name': 'plot_output_path',
             'default': PROC_PATH / 'output' / 'plots'},
            {'name': 'plot_style_path',
             'default': PKG_PATH / 'plots' / 'mpl_styles'}
        ]


class GlobalPackageVariableManager(VariableManager):
    """Class used to manage package global variables"""

    #: int: Config regexp generator limit. Default to 10000.
    config_gen_max = TProp(int, lambda x: int(x))
    #: list of str: Config file allowed extensions. Default to ['yml', 'yaml']
    config_file_ext = TProp(Iterable[str], lambda x: tuple(x))
    #: str: Event ID pattern use in InfoManager to extract event tag.
    evt_id_pat = TProp(Union[str, re.Pattern], lambda x: re.compile(x))
    #: str: Rig ID pattern use in InfoManager to extract rig tag.
    rig_id_pat = TProp(Union[str, re.Pattern], lambda x: re.compile(x))
    #: str: Product ID pattern use in InfoManager to extract product tag.
    prd_id_pat = TProp(Union[str, re.Pattern], lambda x: re.compile(x))
    #: str: GDP model ID pattern use in InfoManager to extract model tag.
    mdl_id_pat = TProp(Union[str, re.Pattern], lambda x: re.compile(x))

    @property
    def _attr_def(self):
        return [
            {'name': 'config_gen_max',
             'default': 10000},
            {'name': 'config_file_ext',
             'default': ['yml', 'yaml']},
            {'name': 'evt_id_pat',
             'default': r'^e:\w+$'},
            {'name': 'rig_id_pat',
             'default': r'^r:\w+$'},
            {'name': 'prd_id_pat',
             'default': r'^p:\w+$'},
            {'name': 'mdl_id_pat',
             'default': r'^m:\w+$'},
        ]


#: GlobalPathVariablesManager: Global variable containing directory path values
path_var = GlobalPathVariablesManager()

#: GlobalPackageVariableManager: Global variable containing global package variables
glob_var = GlobalPackageVariableManager()
