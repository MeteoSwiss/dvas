"""
Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

"""

# Import Python packages and module
import os
from pathlib import Path
import re
from abc import ABC, ABCMeta, abstractmethod
from contextlib import contextmanager
from collections.abc import Iterable

# Import current package's modules
from .helper import SingleInstanceMetaClass
from .helper import TypedProperty as TProp
from .helper import check_path
from .helper import get_class_public_attr
from .errors import DvasError
from . import __name__ as pkg_name
from .hardcoded import PKG_PATH, MPL_STYLES_PATH
from .hardcoded import CSV_FILE_EXT, FLG_FILE_EXT, CONFIG_FILE_EXT
from .hardcoded import CONFIG_GEN_LIM
from .hardcoded import EID_PAT, RID_PAT, TOD_PAT


class ABCSingleInstanceMeta(ABCMeta, SingleInstanceMetaClass):
    """Child Meteclass from ABCMeta and SingleInstanceMetaClass"""


class VariableManager(ABC, metaclass=ABCSingleInstanceMeta):
    """Class to manage variables

    Note:
        Need to initialize attributes in child __init__(), in order to
        call properly set_default_attr() in parent __init__().

    """

    @abstractmethod
    def __init__(self):

        # Check _attr_def
        try:
            assert isinstance(self._attr_def, dict)

        except AssertionError as first_error:
            raise DvasError("Error in matching '_attr_def' pattern") from first_error

        # Set default attributes
        self.set_default_attr()

    def __str__(self):
        return '\n'.join(
            [f"{key}: {val}" for key, val in self.get_attr().items()]
        )

    def get_attr(self):
        """Return current attributes"""
        return get_class_public_attr(self)

    @property
    def _attr_def(self):
        """dict: Class default attributes value. Dict key: Attribute name.
         Dict value: Default attribute value. Attribute undefined will be left
         untouched"""
        return {}

    def set_default_attr(self):
        """Set attribute from _attr_def. Try first to get attribute value from
        environment. All attributes can be defined in environment variables
        using <package name>_<attribute name> in upper case."""

        # Loop on public attributes
        for attr in self.get_attr():
            # Test
            if attr not in self._attr_def.keys():
                continue

            # Set default val
            setattr(
                self,
                attr,
                os.getenv((pkg_name + '_' + attr).upper(), self._attr_def[attr])
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

    #: pathlib.Path: Original data path. Default to None.
    orig_data_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': False}, allow_none=True
    )
    #: pathlib.Path: Config dir path. Default to None.
    config_dir_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': False}, allow_none=True
    )
    #: pathlib.Path: Local db dir path. Default to None.
    local_db_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': False}, allow_none=True
    )
    #: pathlib.Path: DVAS output dir path. Default to None.
    output_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': False}, allow_none=True
    )
    #: pathlib.Path: DVAS output dir path for plots. Default to None.
    plot_output_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': False}, allow_none=True
    )
    #: pathlib.Path: Plot styles dir path. Default to ./plot/mpl_styles.
    plot_style_path = TProp(
        Path | str, check_path, kwargs={'exist_ok': True}, allow_none=False
    )

    def __init__(self):
        # Init attributes
        self.orig_data_path = None
        self.config_dir_path = None
        self.local_db_path = None
        self.output_path = None
        self.plot_output_path = None
        self.plot_style_path = Path('.')

        # Call super constructor
        super().__init__()

    @property
    def _attr_def(self):
        return {
            'orig_data_path': None,
            'config_dir_path': None,
            'local_db_path': None,
            'output_path': None,
            'plot_output_path': None,
            'plot_style_path': PKG_PATH / MPL_STYLES_PATH
        }


class GlobalPackageVariableManager(VariableManager):
    """Class used to manage package global variables"""

    #: int: Config regexp generator limit. Default to 2000.
    config_gen_max = TProp(int, lambda x: min([int(x), CONFIG_GEN_LIM]))
    #: list of str: CSV file allowed extensions. Default to ['csv', 'txt']
    csv_file_ext = TProp(Iterable, lambda x: tuple(x))
    #: list of str: Flag file allowed extensions. Default to ['flg']
    flg_file_ext = TProp(Iterable, lambda x: tuple(x))
    #: list of str: Config file allowed extensions. Default to ['yml', 'yaml']
    config_file_ext = TProp(Iterable, lambda x: tuple(x))
    #: str: Event ID pattern used in InfoManager to extract event tag.
    eid_pat = TProp(str | re.Pattern, lambda x: re.compile(x))
    #: str: Rig ID pattern used in InfoManager to extract rig tag.
    rid_pat = TProp(str | re.Pattern, lambda x: re.compile(x))
    #: str: TimeOfDay ID pattern used in InfoManager to extract rig tag.
    tod_pat = TProp(str | re.Pattern, lambda x: re.compile(x))

    def __init__(self):
        # Init attributes
        self.config_gen_max = 0
        self.csv_file_ext = ['']
        self.config_file_ext = ['']
        self.flg_file_ext = ['']
        self.eid_pat = re.compile('')
        self.rid_pat = re.compile('')
        self.tod_pat = re.compile('')

        # Call super constructor
        super().__init__()

    @property
    def _attr_def(self):
        return {
            'config_gen_max': 2000,
            'csv_file_ext': CSV_FILE_EXT,
            'flg_file_ext': FLG_FILE_EXT,
            'config_file_ext': CONFIG_FILE_EXT,
            'eid_pat': EID_PAT,
            'rid_pat': RID_PAT,
            'tod_pat': TOD_PAT,
        }


#: GlobalPathVariablesManager: Global variable containing directory path values
path_var = GlobalPathVariablesManager()

#: GlobalPackageVariableManager: Global variable containing global package variables
glob_var = GlobalPackageVariableManager()
