# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains tools and routines related to deltas between profiles and CWS.

"""

# Import from Python
import logging
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd

# Import from this module
from ...logger import log_func_call
from ...hardcoded import PRF_TDT, PRF_FLG, PRF_VAL, PRF_UCS, PRF_UCT, PRF_UCU
from ...data.strategy.data import DeltaProfile
from ...data.data import MultiDeltaProfile
from ...errors import DvasError
from ..tools import fancy_bitwise_or, wrap_angle

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def single_delta(prf, cws, circular=False):
    """ Compute the delta between a (single) error-less Profile|RSProfile and a (single)
    error-full CWS.

    Errors from the CWS are simply passed over to the Delta. Time information is also dropped in the
    process.

    Args:
        prf (Profile|RSProfile): the 'candidate' profile.
        cws (CWSProfile): the `reference` Combined Working Standard profile.
        circular (bool, optional): if True, will wrap delta values in the range [-180;+180[.
            Defaults to False.

    Returns:
        DeltaProfile: the delta profile, i.e. `candidate` - `reference`.
    """

    # First, let's run some preliminary checks, to make sure the data I was fed is appropriate.
    for attr in ['edt', 'rid', 'eid']:
        if getattr(prf.info, attr) != getattr(cws.info, attr):
            raise DvasError(f'prf-cws value mismatch for: {attr}')

    if len(prf) != len(cws):
        raise DvasError(f'prf-cws length mismatch: {len(prf)} vs {len(cws)}')

    # Prepare the InfoManager of the delta Profile. Just get it as a copy of the Profile itself
    dta_info = deepcopy(prf.info)

    # For the data, start by deep copying the cws one with all the uncertainties.
    # The ref_alt parameter, in particular, is taken from the CWS.
    dta_data = deepcopy(cws.data)

    # Next compute the delta itself. Here, let's keep in mind that the index from the cws is
    # **different** from the index of the profile !
    dta_data.loc[:, [PRF_VAL]] = prf.data['val'].values - cws.data[PRF_VAL].values

    # Here, let us hide the uncertainties should the prf not contain any data
    for col_name in [PRF_UCS, PRF_UCT, PRF_UCU]:
        dta_data.loc[prf.data['val'].isna().values, [col_name]] = np.nan

    # Handle the angular_wrap, if warranted. This is used to make sure the wdir delta is never
    # larger than +-180 deg
    if circular:
        dta_data.loc[:, [PRF_VAL]] = dta_data.val.map(wrap_angle)
        if (dta_data.val >= 180).any() or (dta_data.val < -180).any():
            raise DvasError('Angular wrapping failed ?!')

    # For the flags, let's apply a bitwise_or to combine the prf and cws values
    flg_pdf = pd.DataFrame([cws.data['flg'].values, prf.data['flg'].values], dtype=int).T
    dta_data.loc[:, [PRF_FLG]] = fancy_bitwise_or(flg_pdf, axis=1)

    # Create a new DeltaProfile instance
    dta = DeltaProfile(dta_info, dta_data)

    # Add the origin of this DeltaProfile
    dta.info.src = f'dvas single/vector_delta() [{Path(__file__).name}]'

    return dta


@log_func_call(logger)
def vector_delta(prf, cws, dir_prf, dir_cws):
    """ Compute the modulus of the vector difference between a (single) error-less Profile|RSProfile
        and a (single) error-full CWS.

    Time information is dropped in the process.

    Warning:
        Errors from the CWS are propagated over to the Delta. In doing so, we assume that
        uncertainties are uncorrelated ! This is only correct for the wind.

    Args:
        prf (Profile|RSProfile): the 'candidate' profile vector amplitude.
        cws (CWSProfile): the `reference` Combined Working Standard profile vector amplitude.
        dir_prf (Profile|RSProfile): the 'candidate' profile vector direction.
        dir_cws (CWSProfile): the `reference` Combined Working Standard profile vector direction.

    Returns:
        DeltaProfile: the delta profile, i.e. ||vec{`candidate`} - vec{`reference`}||.
    """

    # First, let's run some preliminary checks, to make sure the data I was fed is appropriate.
    for attr in ['edt', 'rid', 'eid']:
        if getattr(prf.info, attr) != getattr(cws.info, attr):
            raise DvasError(f'prf-cws value mismatch for: {attr}')
        if getattr(prf.info, attr) != getattr(dir_prf.info, attr):
            raise DvasError(f'prf-dir_prf value mismatch for: {attr}')
        if getattr(prf.info, attr) != getattr(dir_cws.info, attr):
            raise DvasError(f'prf-dir_cws value mismatch for: {attr}')

    if len(prf) != len(cws):
        raise DvasError(f'prf-cws length mismatch: {len(prf)} vs {len(cws)}')
    if len(prf) != len(dir_cws):
        raise DvasError(f'prf-dir_cws length mismatch: {len(prf)} vs {len(dir_cws)}')
    if len(prf) != len(dir_prf):
        raise DvasError(f'prf-dir_prf length mismatch: {len(prf)} vs {len(dir_prf)}')

    # Prepare the InfoManager of the delta Profile. Just get it as a copy of the Profile itself
    dta_info = deepcopy(prf.info)

    # For the data, start by deep copying the cws one with all the uncertainties.
    # The ref_alt parameter, in particular, is taken from the CWS.
    dta_data = deepcopy(cws.data)

    # Next compute the delta itself. Here, let's keep in mind that the index from the cws is
    # **different** from the index of the profile !
    deltatheta = np.deg2rad(dir_prf.data[PRF_VAL].values - dir_cws.data[PRF_VAL].values)
    dta_data.loc[:, [PRF_VAL]] = \
        (prf.data[PRF_VAL].values**2 + cws.data[PRF_VAL].values**2 -
         2 * prf.data[PRF_VAL].values *
         cws.data[PRF_VAL].values *
         np.cos(deltatheta))**0.5

    # Let us now deal with the uncertainties properly
    if cws.uct.notna().any() or cws.ucs.notna().any():
        logger.error('Correlated CWS uncertainties ignored in vector_delta() !')
    if dir_cws.uct.notna().any() or dir_cws.ucs.notna().any():
        logger.error('Correlated directional CWS uncertainties ignored in vector_delta() !')

    dfdspeed = cws.data[PRF_VAL].values - prf.data[PRF_VAL].values * np.cos(deltatheta)
    dfdspeed /= dta_data.loc[:, PRF_VAL].values

    dfddir = - cws.data[PRF_VAL].values * prf.data[PRF_VAL].values * np.sin(deltatheta)
    dfddir /= dta_data.loc[:, PRF_VAL].values

    # Compute the total uncorrelated uncertainty, not forgetting to convert the directional
    # uncerttainty from degrees to radians in order to propagate them.
    dta_data.loc[:, [PRF_UCU]] = (dfdspeed**2 * cws.data.loc[:, PRF_UCU].values**2 +
                                  dfddir**2 *
                                  np.deg2rad(dir_cws.data.loc[:, PRF_UCU].values)**2)**0.5

    # Here, let us hide the uncertainties should the prf not contain any data
    for col_name in [PRF_UCS, PRF_UCT, PRF_UCU]:
        dta_data.loc[prf.data[PRF_VAL].isna().values, [col_name]] = np.nan
        dta_data.loc[dir_prf.data[PRF_VAL].isna().values, [col_name]] = np.nan

    if not (dta_data.ucu.isna() == dta_data.val.isna()).all():
        logger.error('NaN mismatch for vector dta_data.')

    # For the flags, let's apply a bitwise_or to combine the prf and cws values
    flg_pdf = pd.DataFrame([cws.data['flg'].values, prf.data['flg'].values], dtype=int).T
    dta_data.loc[:, [PRF_FLG]] = fancy_bitwise_or(flg_pdf, axis=1)

    # Create a new DeltaProfile instance
    dta = DeltaProfile(dta_info, dta_data)

    # Add the origin of this DeltaProfile
    dta.info.src = f'dvas single/vector_delta() [{Path(__file__).name}]'

    return dta


@log_func_call(logger)
def compute(prfs, cwss, circular=False, dir_prfs=None, dir_cwss=None):
    """ Compute the deltas between many error-less profiles and error-full cws.

    Args:
        prfs (MultiProfile|MultiRSProfile): the `candidate` profiles. Each will be dealt with
            separately,
        cwss (MultiCWSProfile): the `reference` combined working standard profiles. A 1-to-1 pairing
            with prfs is assumed, unless this contains a single profile, in which case the same
            CWS will be subtracted from all the Profiles.
            I.e. len(cwss) == 1 or len(cwss) == len(prfs).
        circular (bool, optional): if True, will wrap delta values in the range [-180;+180[.
            Defaults to False.
        dir_prfs (MultiProfile|MultiRSProfile): direction profiles, if one wants to compute
            the modulus of the vector difference. Defaults to None.
        dir_cwss (MultiCWSProfile): direction CWS profiles, if one wants to compute the modulus of
            the vector difference. Defaults to None.

    Returns:
        MultiDeltaProfile: the DeltaProfiles.
    """

    do_vector = False
    if dir_prfs is not None and dir_cwss is not None:
        do_vector = True

        if _ := dir_cwss.db_variables[PRF_VAL] != 'wdir':
            logger.error('Directional CWS vector component is not "wdir", but: %s', _)
        if _ := dir_prfs.db_variables[PRF_VAL] != 'wdir':
            logger.error('Directional PRF vector component is not "wdir", but: %s', _)

        if _ := cwss.db_variables[PRF_VAL] != 'wspeed':
            logger.error('Amplitude CWS vector component is not "wspeed", but: %s', _)
        if _ := prfs.db_variables[PRF_VAL] != 'wspeed':
            logger.error('Amplitude PRF vector component is not "wspeed", but: %s', _)

    # First, let's make sure I have a proper length.
    if len(cwss) not in [1, len(prfs)]:
        raise DvasError(f'Incompatible cwss length: {len(cwss)} not in [1, {len(prfs)}]')

    if do_vector and len(dir_cwss) not in [1, len(prfs)]:
        raise DvasError(
            f'Incompatible directional cwss length: {len(dir_cwss)} not in [1, {len(prfs)}]')

    if do_vector and not len(dir_prfs) == len(prfs):
        raise DvasError(f'Directional prfs length mismtach: {len(dir_prfs)} vs {len(prfs)}')

    # Now let's make a list of CWSProfile that I can use in a loop
    cws_prfs = [prf for prf in cwss]
    dir_cws_prfs = []
    if do_vector:
        dir_cws_prfs = [prf for prf in dir_cwss]

    # Make sure it has the correct length !
    if len(cws_prfs) == 1:
        cws_prfs *= len(prfs)
    if do_vector and len(dir_cws_prfs) == 1:
        dir_cws_prfs *= len(prfs)

    # Now loop through these, and assemble the DeltaProfile
    dtas = []
    for (prf_ind, prf) in enumerate(prfs):
        if not do_vector:
            dtas += [single_delta(prf, cws_prfs[prf_ind], circular=circular)]
        else:
            dtas += [vector_delta(prf, cws_prfs[prf_ind], dir_prfs[prf_ind], dir_cws_prfs[prf_ind])]

    # All done, let's pack it all inside a MultiDeltaProfile instance.
    out = MultiDeltaProfile()

    # To do that, I need the db_variables dict ...
    dta_var = deepcopy(cwss.db_variables)
    if do_vector:
        # WARNING: yet even more hardcoded stuff ... sigh ...
        dta_var[PRF_VAL] = 'wvec'
        dta_var[PRF_UCU] = 'wvec_ucu'
        dta_var[PRF_FLG] = 'wvec_flag'

    # ... bearing in mind I need to get rid of 'tdt' if it exist !
    dta_var.pop(PRF_TDT, None)  # Return None if it doesn't exists. Better than an error !
    # With this, let's update the MultiDeltaProfile
    out.update(dta_var, data=dtas)

    return out
