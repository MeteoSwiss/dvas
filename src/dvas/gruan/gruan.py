"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related routines, including correlation rules for GDP uncertainties.

"""

import numpy as np

def corcoef_gdps(i, j, uc_type,
                 sn_i=None, sn_j=None,
                 rig_i=None, rig_j=None,
                 event_i=None, event_j=None,
                 site_i=None, site_j=None):
    ''' Computes the correlation coefficient(s) for specific uncertainty types between GDP
    measurements.

    The correlations are as follows:

    - sigmar (uncorrelated errors): no correlation between distinct measurements.
    - sigmas (spatial-correlated errors): full correlation between measurements acquired during the
      same event at the same site, irrespective of the altitude or radiosonde model.
    - sigmat (temporal-correlated errors): full correlation between measurements acquired at
      distinct sites during distinct events, with distinct radiosondes models.

    Args:
        i (numpy.ndarray of int or float): time step or altitude of measurement 1.
        j (numpy.ndarray of int or float): time step or altitude of measurement 2.
        uc_type (str): uncertainty type. Must be one of ['sigmar', 'sigmas', 'sigmat'].
        sn_i (numpy.ndarray of int or str, optional): serial number of RS from measurement 1.
        sn_j (numpy.ndarray of int or str, optional): seriel number of RS from measurement 2.
        rig_i (numpy.ndarray of int or str, optional): rig id of measurement 1.
        rig_j (numpy.ndarray of int or str, optional): rig id of measurement 2.
        event_i (numpy.ndarray of int or str, optional): event id of measurement 1.
        event_j (numpy.ndarray of int or str, optional): event id of measurement 2.
        site_i (numpy.ndarray of int or str, optional): site id of measurement 1.
        site_j (numpy.ndarray of int or str, optional): site id of measurement 2.

    Returns:
        numpy.ndarray of float: the correlation coefficient(s).

    Note:
        This function returns the pair-wise correlation coefficients, and
        **not** the full correlation matrix (i.e. `len(corcoef_gdps(i, j, uc_type))ß == len(i)`).

    Todo:
        - Consider refactoring the input using `collections.namedtuple` ?
        - Add reference to GRUAN docs in this docstring.

    '''

    # Begin with some safety checks
    for var in [i, j, sn_i, sn_j, rig_i, rig_j, event_i, event_j, site_i, site_j]:
        if var is None:
            continue
        if not isinstance(var, np.ndarray):
            raise Exception('Ouch ! I was expecting a numpy.ndarray, not %s' % type(var))
        if np.shape(var) != np.shape(i):
            raise Exception('Ouch ! All items should have the same shape !')

    # Make sure to return something with the same shape as what came in.
    corcoef = np.zeros(np.shape(i))

    # All variables always correlate fully with themselves
    corcoef[(i == j) * (sn_i == sn_j) *
            (rig_i == rig_j) * (event_i == event_j) * (site_i == site_j)] = 1.0

    # Now work in the required level of correlation depending on the case.
    if uc_type == 'sigmas':
        # 1) Full spatial-correlation between measurements acquired in the same event and at the
        # same site.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef[(event_i == event_j) * (site_i == site_j)] = 1.0

    elif uc_type == 'sigmat':
        # 1) Full temporal-correlation between measurements acquired in the same event and at the
        # same site.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef[(event_i == event_j) * (site_i == site_j)] = 1.0

        # 2) Full temporal correlation between measurements acquired in distinct events and sites.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef = np.ones_like(corcoef)

    else:
        raise Exception("Ouch ! uc_type must be one of ['sigmar', 'sigmas', 'sigmat']")

    return corcoef
