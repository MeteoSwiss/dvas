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
import pandas as pd

# Import from this module
from ...logger import log_func_call
from ...hardcoded import PRF_REF_TDT_NAME, PRF_REF_FLG_NAME, PRF_REF_VAL_NAME
from ...data.strategy.data import DeltaProfile
from ...data.data import MultiDeltaProfile
from ...errors import DvasError
from ..tools import fancy_bitwise_or

# Setup local logger
logger = logging.getLogger(__name__)


@log_func_call(logger)
def single_delta(prf, cws):
    """ Compute the delta between a (single) error-less Profile|RSProfile and a (single)
    error-full CWS.

    Errors from the CWS are simply passed over to the Delta. Time information is also dropped in the
    process.

    Args:
        prf (Profile|RSProfile): the 'candidate' profile.
        cws (CWSProfile): the `reference` Combined Working Standard profile.

    Returns:
        DeltaProfile: the delta profile, i.e. `candidate` - `reference`.
    """

    # First, let's run some preliminary checks, to make sure the data I was fed is appropriate.
    for attr in ['edt', 'rid', 'eid']:
        if getattr(prf.info, attr) != getattr(cws.info, attr):
            raise DvasError(f'Ouch ! prf-cws value mismatch for: {attr}')

    if len(prf) != len(cws):
        raise DvasError(f'Ouch ! prf-cws length mismatch: {len(prf)} vs {len(cws)}')

    # Prepare the InfoManager of the delta Profile. Just get it as a copy of the Profile itself
    dta_info = deepcopy(prf.info)

    # For the data, start by deep copying the cws one with all the uncertainties.
    # The ref_alt parameter, in particular, is taken from the CWS.
    dta_data = deepcopy(cws.data)

    # Next compute the delta itself. Here, let's keep in mind that the index from the cws is
    # **different** from the index of the profile !
    dta_data.loc[:, [PRF_REF_VAL_NAME]] = prf.data['val'].values - cws.data[PRF_REF_VAL_NAME].values

    # For the flags, let's apply a bitwise_or to combine the prf and cws values
    flg_pdf = pd.DataFrame([cws.data['flg'].values, prf.data['flg'].values], dtype='Int64').T
    dta_data.loc[:, [PRF_REF_FLG_NAME]] = fancy_bitwise_or(flg_pdf, axis=1)

    # Create a new DeltaProfile instance
    dta = DeltaProfile(dta_info, dta_data)

    # Add the origin of this DeltaProfile
    dta.info.src = 'dvas single_delta() [{}]'.format(Path(__file__).name)

    return dta


def compute(prfs, cwss):
    """ Compute the deltas between many error-less profiles and error-full cws.

    Args:
        prfs (MultiProfile|MultiRSProfile): the `candidate` profiles. Each will be dealt with
            separately,
        cwss (MultiGDPProfile): the `reference` combined working standard profiles. A 1-to-1 pairing
            with prfs is assumed, unless this contains a single profile, in which case the same
            CWS will be subtracted from all the Profiles.
            I.e. len(cwss) == 1 or len(cwss) == len(prfs).

    Returns:
        MultiDeltaProfile: the DeltaProfiles.
    """

    # First, let's make sure I have a proper length.
    if not len(cwss) in [1, len(prfs)]:
        raise DvasError(f'Ouch ! Incompatible cwss length: {len(cwss)} not in [1, {len(prfs)}]')

    # Now let's make a list of CWSProfile that I can use in a loop
    cws_prfs = [prf for prf in cwss]
    # Make sure it has the correct length !
    if len(cws_prfs) == 1:
        cws_prfs *= len(prfs)

    # Now loop through these, and assemble the DeltaProfile
    dtas = []
    for (prf_ind, prf) in enumerate(prfs):
        dtas += [single_delta(prf, cws_prfs[prf_ind])]

    # All done, let's pack it all inside a MultiDeltaProfile instance.
    out = MultiDeltaProfile()

    # To do that, I need the db_variables dict ...
    dta_var = deepcopy(cwss.db_variables)
    # ... bearing in mind I need to get rid of 'tdt' if it exist !
    dta_var.pop(PRF_REF_TDT_NAME, None)  # Return None if it doesn't exists. Better than an error !
    # With this, let's update the MultiDeltaProfile
    out.update(dta_var, data=dtas)

    return out
