# -*- coding: utf-8 -*-
"""

Copyright (c) 2020-2022 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-correlations-related utilities, including the means to compute
correlation coefficients.

"""

# WARNING: this module should NOT import anything from dvas.data (because it itself is being
# imported there ...), and used in some of the MultiProfile Strategies.

# Import from Python
import logging
import numpy as np

# Import from current package
from ...errors import DvasError

# Setup local logger
logger = logging.getLogger(__name__)


def corr_coeff_matrix(sigma_name, step_ids, oids=None, mids=None, rids=None, eids=None):
    ''' Computes the correlation coefficient(s) between distinct GDP measurement, for a specifc
    uncertainty type.

    Args:
        sigma_name (str): uncertainty type. Must be one of ['ucs', 'uct', 'ucu'].
        step_ids (1D numpy.ndarray of int|float): synchronized time, step, or altitude id of each
            measurement.
        oids (1D numpy.ndarray of int|str, optional): object id from measurement 1.
        mids (1D numpy.ndarray of int|str, optional): GDP model from measurement 1.
        rids (1D numpy.ndarray of int|str, optional): rig id of measurement 1.
        eids (1D numpy.ndarray of int|str, optional): event id of measurement 1.

    Warning:
        - If no oids are specified, the function will assume that the data originates
          **from the exact same radiosonde.** Idem for the GDP model ids, rig ids and event ids.

    Returns:
        numpy.ndarray of float(s): the square correlation coefficient(s) array, in the range [0, 1],
            with shape (len(step_ids), len(step_ids)).

    The supported uncertainty types are:

    - 'ucs': spatial-correlated uncertainty.
             Full correlation between measurements acquired during the same event at the same site,
             irrespective of the time step/altitude, rig, or serial number. No correlation between
             distinct radiosonde models.
    - 'uct': temporal-correlated uncertainty.
             Full correlation between measurements acquired at distinct sites during distinct
             events, with distinct and serial numbers.
             No correlation between distinct radiosonde models.
    - 'ucu': un-correlated uncertainty.
             No correlation whatsoever between distinct measurements.

    '''

    # Begin with some safety checks
    for var in [step_ids, oids, mids, rids, eids]:
        if var is None:
            continue
        if not isinstance(var, np.ndarray):
            raise DvasError(f'I was expecting a numpy.ndarray, not: {type(var)}')
        if var.ndim != 1:
            raise DvasError(f'I need 1D ndarray, not: {var.ndim}D')
        if var.shape != step_ids.shape:
            raise DvasError(f'Inconsistent shape: {var.shape} vs {step_ids.shape}')

    # How many points were specified
    npts = len(step_ids)

    # If all the points are unique, I can save a lot of time. So let's count how many unique points
    # there are.
    all_unique = np.unique(
        np.concatenate([np.array([step_ids]), np.array([oids]), np.array([mids]),
                        np.array([rids]), np.array([eids])], axis=0).T, axis=0)
    all_unique = len(all_unique) == npts

    # Let's anticipate some of the upcoming correlation checks, and do those expensive steps once
    # only. We here rely on the ability of numpy to broadcast arrays, which provides a significant
    # speed gain.
    conds = {'step_ids': step_ids, 'oids': oids, 'mids': mids, 'rids': rids, 'eids': eids}

    for [cond, val] in conds.items():
        # Either no value was specified, or the points are all unique and I can ignore some stuff
        # This is the spot where I save a lot of time, given that the == step is the most consuming.

        # For each sigma_name, skip unused parameters
        check1 = all_unique and sigma_name == 'ucu'
        check2 = all_unique and sigma_name == 'uct' and cond not in ['mids']
        check3 = all_unique and sigma_name == 'ucs' and cond not in ['eids', 'mids']

        if check1 or check2 or check3:
            conds[cond] = None
        elif val is None:  # If I was given no value, assume it is the same for all points
            conds[cond] = True
        else:
            # Ok, I need to check which points share a common value.
            # Let's not forget that we requested 1D arrays ... so we add a dimension to do a .T
            # Save time by adding a copy to the transpose (?)
            conds[cond] = np.array([val]) == np.array([val]).T.copy()

    # If all the point specified are different in some way(s), then the fastest is to
    # initialize a diagonal matrix.
    if all_unique:
        corcoef = np.diagflat(np.ones(npts))
    # Else, we need to multiply many things, to make sure we get all the proper correlations
    # between the identical points
    else:
        corcoef = np.zeros((npts, npts))
        corcoef[conds['step_ids'] * conds['oids'] * conds['mids'] *
                conds['rids'] * conds['eids']] = 1.

    # Now work in the required level of correlation depending on the uncertainty type.
    if sigma_name == 'ucu':
        # Nothing to add in case of uncorrelated uncertainties.
        pass

    elif sigma_name == 'ucs':
        # 1) Full spatial-correlation between measurements acquired in the same event with the same
        #    GDP model, irrespective of the rig, or serial number.
        corcoef[conds['eids'] * conds['mids']] = 1.0

    elif sigma_name == 'uct':
        # 1) Full temporal-correlation between measurements acquired with the same GDP model,
        # irrespective of event, rig, or serial number.
        corcoef[conds['mids']] = 1.0

    else:
        raise DvasError(f"uc_type must be one of ['ucs', 'uct', 'ucu'], not: {sigma_name}")

    return corcoef
