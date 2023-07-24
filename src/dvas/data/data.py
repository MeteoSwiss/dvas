"""
Copyright (c) 2020-2023 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Data management

"""

# Import from external packages
from abc import abstractmethod
import numpy as np
import pandas as pd

# Import from current package
from .strategy.data import Profile, RSProfile, GDPProfile, CWSProfile, DeltaProfile
from .strategy.load import LoadProfileStrategy, LoadRSProfileStrategy, LoadGDPProfileStrategy
from .strategy.load import LoadCWSProfileStrategy, LoadDeltaProfileStrategy
from .strategy.sort import SortProfileStrategy
from .strategy.plot import PlotStrategy, RSPlotStrategy, GDPPlotStrategy
from .strategy.rebase import RebaseStrategy
from .strategy.resample import ResampleStrategy
from .strategy.save import SaveDataStrategy
from ..database.database import DatabaseManager
from ..database.model import Prm as TableParameter
from ..helper import RequiredAttrMetaClass
from ..helper import deepcopy
from ..helper import get_class_public_attr
from ..errors import DBIOError, DvasError
from ..hardcoded import TAG_ORIGINAL, PRF_IDX, PRF_FLG, PRF_VAL

# Loading strategies
load_prf_stgy = LoadProfileStrategy()
load_rsprf_stgy = LoadRSProfileStrategy()
load_gdpprf_stgy = LoadGDPProfileStrategy()
load_cwsprf_stgy = LoadCWSProfileStrategy()
load_dtaprf_stgy = LoadDeltaProfileStrategy()

# Plotting strategies
plt_prf_stgy = PlotStrategy()
plt_rsprf_stgy = RSPlotStrategy()
plt_gdpprf_stgy = GDPPlotStrategy()

# Rebasing strategies
rebase_prf_stgy = RebaseStrategy()

# Resampling strategies
resample_prf_stgy = ResampleStrategy()

# Other strategies
sort_prf_stgy = SortProfileStrategy()
save_prf_stgy = SaveDataStrategy()


class MultiProfileAC(metaclass=RequiredAttrMetaClass):
    """Abstract MultiProfile class"""

    # Specify required attributes
    # _DATA_TYPES:
    # - type (Profile|RSProfile|GDPProfile|DeltaProfile|...)
    REQUIRED_ATTRIBUTES = {
        '_DATA_TYPES': type
    }

    #: type: Data type
    _DATA_TYPES = None

    _DATA_EMPTY = []
    _DB_VAR_EMPTY = {}

    @abstractmethod
    def __init__(self, load_stgy=None, sort_stgy=None, save_stgy=None, plot_stgy=None,
                 rebase_stgy=None):

        # For the class iterator
        self.ind = 0

        # Init attributes
        self._load_stgy = load_stgy
        self._sort_stgy = sort_stgy
        self._save_stgy = save_stgy
        self._plot_stgy = plot_stgy
        self._rebase_stgy = rebase_stgy

        self._profiles = self._DATA_EMPTY
        self._db_variables = self._DB_VAR_EMPTY

    def __len__(self):
        return len(self.profiles)

    def __getitem__(self, i):
        return self.profiles[i]

    @property
    def profiles(self):
        """list of Profile"""
        return self._profiles

    @property
    def db_variables(self):
        """dict: Correspondence between DataFrame and DB parameter"""
        return self._db_variables

    @property
    def var_info(self):
        """dict: Variable informations"""

        # Define
        db_mngr = DatabaseManager()

        # Query parameter info
        qry_res = db_mngr.get_table(
            TableParameter,
            search={
                'where': TableParameter.prm_name.in_(
                    [val for val in self.db_variables.values() if val]
                )
            }
        )

        # Swap db variables dict
        var_db = {val: key for key, val in self.db_variables.items() if val}

        # Set output
        out = {
            var_db[res[TableParameter.prm_name.name]]: {
                TableParameter.prm_name.name: res[TableParameter.prm_name.name],
                TableParameter.prm_plot.name: res[TableParameter.prm_plot.name],
                TableParameter.prm_desc.name: res[TableParameter.prm_desc.name],
                TableParameter.prm_cmt.name: res[TableParameter.prm_cmt.name],
                TableParameter.prm_unit.name: res[TableParameter.prm_unit.name],
            } for res in qry_res
        }

        return out

    @property
    def info(self):
        """ List of ProfileManger info: Data info"""
        return [arg.info for arg in self.profiles]

    @deepcopy
    def rm_info_tags(self, val):
        """ Remove some tags from all info tag lists.

        Args:
            val (str|list of str): Tag value(s) to remove

        """
        for i in range(len(self)):
            self.profiles[i].info.rm_tags(val)

    @deepcopy
    def add_info_tags(self, val):
        """ Add tag from all info tags

        Args:
            val (str|list of str): Tag values to add.

        """
        for i in range(len(self)):
            self.profiles[i].info.add_tags(val)

    def copy(self):
        """ Return a deep copy of the object"""
        obj = self.__class__()
        obj._db_variables = self.db_variables.copy()
        obj._profiles = [arg.copy() for arg in self.profiles]
        return obj

    def extract(self, inds):
        """ Return a new MultiProfile instance with a subset of the Profiles.

        Args:
            inds (int|list of int): indices of the Profiles to extract.

        Return:
            dvas.data.data.MultiProfile: the new instance.
        """

        # Be extra nice and turn ints into lists
        if isinstance(inds, int):
            inds = list[inds]

        new_prfs = self.__class__()
        new_prfs.update(self.db_variables.copy(),
                        [item.copy() for (ind, item) in enumerate(self) if ind in inds])
        return new_prfs

    @deepcopy
    def load_from_db(self, *args, **kwargs):
        """ Load data from the database.

        Args:
            *args: positional arguments
            **kwargs: key word arguments

        """

        # Call the appropriate Data strategy
        data, db_df_keys = self._load_stgy.execute(*args, **kwargs)

        # Test data len
        if not data:
            raise DBIOError('Load empty data')

        # Update
        self.update(db_df_keys, data)

    @deepcopy
    def sort(self):
        """ Sort method

        """

        # Sort
        data = self._sort_stgy.execute(self.profiles)

        # Load
        self.update(self.db_variables, data)

    def save_to_db(self, add_tags=None, rm_tags=None, prms=None):
        """ Save method to store the *entire* content of the Multiprofile
        instance back into the database with an updated set of tags.

        Args:
            add_tags (list of str, optional): list of tags to add to the entity when inserting it
                into the database. Defaults to None.
            rm_tags (list of str, optional): list of *existing* tags to remove
                from the entity before inserting ot into the database. Defaults to None.
            prms (list of str, optional): list of column names to save to the
                database. Defaults to None (= save all possible parameters).

        Notes:
            The TAG_ORIGINAL will always be removed and the 'derived' tag will
            always be added by default when saving anything into the database.

        """

        # Init
        obj = self.copy()

        # Add tags
        if add_tags is not None:
            obj.add_info_tags(add_tags)

        # Remove tag ORIGINAL
        rm_tags = [TAG_ORIGINAL] if rm_tags is None else rm_tags + [TAG_ORIGINAL]

        # Remove tags
        obj.rm_info_tags(rm_tags)

        # Restructure the parameters into a dict, to be consistent with the rest of the class.
        if prms is None:
            prms = list(self.db_variables.keys())

        # Call save strategy
        self._save_stgy.execute(obj, prms)

    # TODO: implement an "export" function that can export specific DataFrame columns back into
    #  the database under new variable names ?
    # This may be confusing. Wouldn't it be sufficient to change the keys in db_variables and then
    # use the existing "save_to_db" method ?

    def update(self, db_df_keys, data):
        """ Update the whole Multiprofile list with new Profiles.

        Args:
            db_df_keys (dict): Relationship between database parameters and
                Profile.data columns.
            data (list of Profile): Data

        """

        # Check input type
        assert isinstance(data, list), f"Was expecting a list, not: {type(data)}"
        # Test data if not empty
        if data:

            # Check input value type
            assert all([isinstance(arg, self._DATA_TYPES) for arg in data]),\
                f"Wrong data type: I need {self._DATA_TYPES} but you gave me " +\
                f"{[type(arg) for arg in data]}"

            # Check db keys
            assert (
                set(db_df_keys.keys()) == set(data[0].get_col_attr() + data[0].get_index_attr())
            ), f"Key {db_df_keys} does not match data columns"

        else:
            data = self._DATA_EMPTY
            db_df_keys = self._DB_VAR_EMPTY

        # Update
        self._db_variables = db_df_keys
        self._profiles = data

    def append(self, db_df_keys, val):
        """ Append method

        Args:
            db_df_keys (dict): Relationship between database parameters and
                Profile.data columns.
            val (Profile): Data

        """

        self.update(db_df_keys, self.profiles + [val])

    def get_prms(self, prm_list=None, mask_flgs=None, request_flgs=None, with_metadata=None,
                 pooled=False):
        """ Convenience getter to extract specific columns from the DataFrames and/or class
        properties of all the Profile instances.

        Args:
            prm_list (str|list of str, optional): names of the column(s) to extract from all the
                Profile DataFrames. Defaults to None (=returns all the columns from the DataFrame).
            mask_flgs (str|list of str, optional): name(s) of the flag(s) to NaN-ify in the
                extraction process. Defaults to None.
            request_flgs(str|list of str, optional): if set, will only return points that have these
                flag values set (AND rule applied, if multiple values are provided).
            with_metadata (str|list, optional): name of the metadata fields to include in the table.
                Defaults to None.
            pooled (bool, optional): if True, all profiles will be gathered together. If False,
                Profiles are kept distinct using a MultiIndex. Defaults to False.

        Returns:
            pd.DataFrame: the requested data as a MultiIndex pandas DataFrame.

        Warning:
            The resulting DataFrame has only ``dvas.hardcoded.PRF_IDX`` (='_idx') as
            an index. Since the values of ``dvas.hardcoded.PRF_TDT`` (='tdt') and
            ``dvas.hardcoded.PRF_ALT`` (='alt') are not necessarily the sames for all
            the Profiles, these cannot be used as common indexes here.

        """

        # Begin with some sanity checks
        if prm_list is None:
            prm_list = list(self.var_info.keys())

        if isinstance(prm_list, str):
            # Be nice/foolish and assume the user forgot to put the key into a list.
            prm_list = [prm_list]

        if mask_flgs is not None:
            if isinstance(mask_flgs, str):
                mask_flgs = [mask_flgs]

        if request_flgs is not None:
            if isinstance(request_flgs, str):
                request_flgs = [request_flgs]

        if with_metadata is not None:
            if isinstance(with_metadata, str):
                with_metadata = [with_metadata]
        else:
            with_metadata = []

        # Let's prepare the data. First, put all the DataFrames into a list
        out = [pd.concat([getattr(prf, prm) for prm in prm_list], axis=1, ignore_index=False)
               for prf in self.profiles]

        # If warranted, let's hide the selected flagged elements
        # Note: here, we also hide the indices of the original DataFrame (i.e. tdt and alt).
        # This is a choice that will remain a good one, until it isn't.
        if mask_flgs is not None:
            for flg in mask_flgs:
                for (p_ind, prf) in enumerate(self.profiles):
                    out[p_ind].loc[prf.has_flg(flg), out[p_ind].columns != PRF_FLG] = np.nan
                    # As of #253, flgs cannot be NaNs, so I treat them separately.
                    out[p_ind].loc[prf.has_flg(flg), PRF_FLG] = 0

        if request_flgs is not None:
            for (p_ind, prf) in enumerate(self.profiles):
                valids = np.array([True] * len(prf))
                for flg in request_flgs:
                    valids *= prf.has_flg(flg).values

                out[p_ind] = out[p_ind][valids]

        # Drop the superfluous index
        out = [df.reset_index(level=[name for name in df.index.names
                                     if name not in [PRF_IDX]],
                              drop=True)
               for df in out]

        # Drop all the columns I do not want to keep
        out = [df[prm_list] for df in out]

        # I may also have been asked to include some metadata as new columns
        for item in with_metadata:
            vals = self.get_info(item)

            # Loop through it and assign the values
            for (prf_id, val) in enumerate(vals):

                # If I am being given a list, make sure it has only 1 element.
                if isinstance(val, list):
                    if len(val) > 1:
                        raise DvasError(f"Metadata field'{item}' for profile #{prf_id} " +
                                        f"contains more than one value ({val})." +
                                        " I am too dumb to handle this. So I give up here.")
                    val = val[0]

                # Actually assign the value to each measurement of the profile.
                out[prf_id] = out[prf_id].assign(**{f'{item}':val})

        # If warranted, pool all the data together
        if pooled:
            for (df_ind, df) in enumerate(out):
                out[df_ind]['profile_index'] = df_ind

            out = [pd.concat(out, axis=0)]

        # Before I combine everything in one big DataFrame, I need to re-organize the columns
        # to avoid collisions.
        # Let's group all columns from one profile under its position in the
        # list (0,1, ...) using pd.MultiIndex()
        for (df_ind, df) in enumerate(out):
            out[df_ind].columns = pd.MultiIndex.from_tuples([(df_ind, item)
                                                             for item in df.columns],
                                                            names=('#', 'prm'))

        # Great, I can now bring everything into one large DataFrame
        out = pd.concat(out, axis=1)

        return out

    def get_info(self, prm=None):
        """ Convenience function to extract Info from all the Profile instances.

        Args:
            prm (str, `optional`): Info attribute to extract. Default to None.

        Returns:
            dict of list: idem to self.profiles, but with only the requested metadata.

        """

        if prm:
            out = [getattr(info, prm) for info in self.info]
        else:
            out = [get_class_public_attr(info) for info in self.info]

        return out

    def has_tag(self, tag):
        """ Convenience method to check if the different Profile each have a specific tag, or not.

        Args:
            tag (str): tag to search for.

        Returns:
            list of bool: one bool for each Profile.
        """

        return [item.has_tag(tag) for item in self]

    def plot(self, **kwargs):
        """ Plot method

        Args:
            **kwargs: Keyword arguments to be passed down to the plotting function.

        Returns:
            None

        """

        self._plot_stgy.execute(self, **kwargs)

    @deepcopy
    def rebase(self, new_lengths, shifts=None):
        """ Rebase method, which allows to map Profiles on new set of integer indices.

        This will move the values around, including the non-integer indices (i.e. anything
        other than '_idx') if applicable.

        Args:
            new_lengths (int|list of int): The length of the DataFrame to rebase upon.
                If specifiying an int, the same length will be applied to all Profiles. Else, the
                list should specify a length for each Profile.
            shifts (int|list of int, optional): row n of the existing data will become row n+shift.
                If specifiying an int, the same shift will be applied to all Profiles. Else, the
                list should specify a shift for each Profile. Defaults to None (=no shift).

        """

        data = self._rebase_stgy.execute(self.profiles, new_lengths, shifts=shifts)

        self.update(self.db_variables, data)


class MultiProfile(MultiProfileAC):
    """ Multi profile base class, designed to handle multiple Profile."""

    #: type: supported Profile Types
    _DATA_TYPES = Profile

    def __init__(self):
        super().__init__(load_stgy=load_prf_stgy, sort_stgy=sort_prf_stgy,
                         save_stgy=save_prf_stgy, plot_stgy=plt_prf_stgy,
                         rebase_stgy=rebase_prf_stgy)


class MultiRSProfileAC(MultiProfileAC):
    """ Abstract MultiRSProfile class"""

    @abstractmethod
    def __init__(self, load_stgy=None, sort_stgy=None, save_stgy=None, plot_stgy=None,
                 rebase_stgy=None, resample_stgy=None):
        super().__init__(load_stgy=load_stgy, sort_stgy=sort_stgy, save_stgy=save_stgy,
                         plot_stgy=plot_stgy, rebase_stgy=rebase_stgy)

        self._resample_stgy = resample_stgy

    @deepcopy
    def resample(self, freq='1s', interp_dist=1, chunk_size=150, n_cpus=1):
        """ Resample the profiles (one-by-one) onto regular timesteps using linear interpolation.

        Args:
            freq (str): see pandas.timedelta_range(). Defaults to '1s'.
            interp_dist(int|float): Distance beyond which to not interpolate, and use NaNs.
                Defaults to 1s.

        Note:
            Will unwrap angles if self.var_info[PRF_VAL]['prm_name'] == 'wdir'.

        """

        if self.var_info[PRF_VAL]['prm_name'] == 'wdir':
            circular = True
        else:
            circular = False

        data = self._resample_stgy.execute(self.profiles, freq=freq, interp_dist=interp_dist,
                                           chunk_size=chunk_size, n_cpus=n_cpus, circular=circular)
        self.update(self.db_variables, data)


class MultiRSProfile(MultiRSProfileAC):
    """ Multi RS profile manager, designed to handle multiple RSProfile instances. """

    _DATA_TYPES = RSProfile

    def __init__(self):
        super().__init__(load_stgy=load_rsprf_stgy, sort_stgy=sort_prf_stgy,
                         save_stgy=save_prf_stgy, plot_stgy=plt_rsprf_stgy,
                         rebase_stgy=rebase_prf_stgy,
                         resample_stgy=resample_prf_stgy)


class MultiGDPProfileAC(MultiRSProfileAC):
    """ Abstract MultiGDPProfile class """

    @abstractmethod
    def __init__(self, load_stgy=None, sort_stgy=None, save_stgy=None, plot_stgy=None,
                 rebase_stgy=None, resample_stgy=None):
        super().__init__(load_stgy=load_stgy, sort_stgy=sort_stgy, save_stgy=save_stgy,
                         plot_stgy=plot_stgy, rebase_stgy=rebase_stgy, resample_stgy=resample_stgy)

    @property
    def uc_tot(self):
        """ Convenience getter to extract the total uncertainty from all the Profile instances.

        Returns:
            list of DataFrame: idem to self.profiles, but with only the requested data.

        """
        return [arg.uc_tot for arg in self.profiles]


class MultiGDPProfile(MultiGDPProfileAC):
    """ Multi GDP profile manager, designed to handle multiple GDPProfile instances. """

    _DATA_TYPES = GDPProfile

    def __init__(self):
        super().__init__(load_stgy=load_gdpprf_stgy, sort_stgy=sort_prf_stgy,
                         save_stgy=save_prf_stgy, plot_stgy=plt_gdpprf_stgy,
                         rebase_stgy=rebase_prf_stgy, resample_stgy=resample_prf_stgy)


class MultiCWSProfile(MultiGDPProfileAC):
    """ Multi CWS profile manager, designed to handle multiple GDPProfile instances. """

    _DATA_TYPES = CWSProfile

    def __init__(self):
        super().__init__(load_stgy=load_cwsprf_stgy, sort_stgy=sort_prf_stgy,
                         save_stgy=save_prf_stgy, plot_stgy=plt_gdpprf_stgy,
                         rebase_stgy=rebase_prf_stgy, resample_stgy=resample_prf_stgy)


class MultiDeltaProfile(MultiGDPProfileAC):
    """ Multi Delta profile manager, designed to handle multiple DeltaProfile instances. """

    _DATA_TYPES = DeltaProfile

    def __init__(self):
        super().__init__(load_stgy=load_dtaprf_stgy, sort_stgy=sort_prf_stgy,
                         save_stgy=save_prf_stgy, plot_stgy=plt_gdpprf_stgy,
                         rebase_stgy=rebase_prf_stgy, resample_stgy=resample_prf_stgy)
