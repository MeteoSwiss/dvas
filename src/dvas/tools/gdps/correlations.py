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
import numpy as np

# Import from current package
from ...logger import log_func_call
from ...logger import tools_logger as logger
from ...errors import DvasError


@log_func_call(logger)
def coeffs(i, j, sigma_name, oid_i=None, oid_j=None, mid_i=None, mid_j=None,
           rid_i=None, rid_j=None, eid_i=None, eid_j=None):
    ''' Computes the correlation coefficient(s), for the different uncertainty types of GRUAN Data
    Products (GDPs), between specific measurements.

    Args:
        i (numpy.ndarray of int|float): synchronized time, step, or altitude of measurement 1.
        j (numpy.ndarray of int|float): synchronized time, step, or altitude of measurement 2.
        sigma_name (str): uncertainty type. Must be one of ['ucr', 'ucs', 'uct', 'ucu'].
        oid_i (numpy.ndarray of int|str, optional): object id from measurement 1.
        oid_j (numpy.ndarray of int|str, optional): object id from measurement 2.
        mid_i (numpy.ndarray of int|str, optional): GDP model from measurement 1.
        mid_j (numpy.ndarray of int|str, optional): GDP model from measurement 2.
        rid_i (numpy.ndarray of int|str, optional): rig id of measurement 1.
        rid_j (numpy.ndarray of int|str, optional): rig id of measurement 2.
        eid_i (numpy.ndarray of int|str, optional): event id of measurement 1.
        eid_j (numpy.ndarray of int|str, optional): event id of measurement 2.

    Warning:
        - If no oids are specified, the function will assume that the data originates
          **from the exact same radiosonde.** Idem for the GDP models, rig id and event id.
        - The profiles are assumed to be synchronized, i.e. if specifying i and j as steps,
          i=j implies that they both have the same altitude.

    Returns:
        numpy.ndarray of float(s): the correlation coefficient(s), in the range [0, 1].

    Note:
        This function returns the pair-wise correlation coefficients,
        **not** the full correlation matrix, i.e::

            shape(corcoef_gdps(i, j, uc_type)) == shape(i) == shape(j)

    The supported uncertainty types are:

    - 'ucr': rig-correlated uncertainty.
             Intended for the so-called "uncorrelated" GRUAN uncertainty.
    - 'ucs': spatial-correlated uncertainty.
             Full correlation between measurements acquired during the same event at the same site,
             irrespective of the time step/altitude, rig, radiosonde model, or serial number.
    - 'uct': temporal-correlated uncertainty.
             Full correlation between measurements acquired at distinct sites during distinct
             events, with distinct radiosondes models and serial numbers.
    - 'ucu': uncorrelated.
             No correlation whatsoever between distinct measurements.

    Todo:
        - Add reference to GRUAN docs & dvas articles in this docstring.

    '''

    # Begin with some safety checks
    for var in [i, j, oid_i, oid_j, mid_i, mid_j, rid_i, rid_j, eid_i, eid_j]:
        if var is None:
            continue
        if not isinstance(var, np.ndarray):
            raise DvasError('Ouch ! I was expecting a numpy.ndarray, not %s' % type(var))
        if np.shape(var) != np.shape(i):
            raise DvasError('Ouch ! All items should have the same shape !')

    # Make sure to return something with the same shape as what came in.
    corcoef = np.zeros_like(i)

    # All variables always correlate fully with themselves
    corcoef[(i == j) * (oid_i == oid_j) * (mid_i == mid_j) *
            (rid_i == rid_j) * (eid_i == eid_j)] = 1.0

    # Now work in the required level of correlation depending on the uncertainty type.
    if sigma_name == 'ucu':
        # Nothing to add in case of uncorrelated uncertainties.
        pass

    elif sigma_name == 'ucr':

        # The so-called "uncorrelated" uncertainties from the GDPs show clear signs of correlations
        # between radiosondes that fly together.
        # This is related to the manner through which this uncertainty is being derived, which is
        # sensitive to short-but-real atmospheric fluctuations, which are common between radiosondes
        # flying together. This also implies that these uncertainties are being underestimated.
        #
        # Here, we choose to entirely ignore any correlation between the ucr components. This helps
        # reduce the impact of this "additional" uncertainty, related to real atmospheric
        # fluctuations, that should not have been present in the first place. In other words, we
        # (partially) correct a wrong with a wrong.
        #
        # See the scientific dvas documentation for details on this aspect.
        pass

    elif sigma_name == 'ucs':
        # 1) Full spatial-correlation between measurements acquired in the same event with the same
        #    GDP model, irrespective of the rig, or serial number.
        corcoef[(eid_i == eid_j) * (mid_i == mid_j)] = 1.0

    elif sigma_name == 'uct':
        # 1) Full temporal-correlation between measurements acquired with the same GDP model,
        # irrespective of event, rig, or serial number.
        corcoef[(mid_i == mid_j)] = 1.0

    else:
        raise DvasError("Ouch ! uc_type must be one of ['ucr', 'ucs', 'uct', 'ucu'], not: %s" %
                        (sigma_name))

    return corcoef
