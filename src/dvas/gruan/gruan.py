# -*- coding: utf-8 -*-
"""

Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

This module contains GRUAN-related routines, including correlation rules for GDP uncertainties.

"""

# Import from Python packages
import numpy as np
import pandas as pd

# Import from current package
from ..dvas_logger import gruan_logger, log_func_call, dvasError

@log_func_call(gruan_logger)
def corcoef_gdps(i, j, uc_type,
                 srn_i=None, srn_j=None,
                 mod_i=None, mod_j=None,
                 rig_i=None, rig_j=None,
                 evt_i=None, evt_j=None,
                 sit_i=None, sit_j=None):
    ''' Computes the correlation coefficient(s), for the specific uncertainty types of GRUAN Data
    Products (GDPs), between measurements.

    Args:
        i (numpy.ndarray of int or float): synchronized time step or altitude of measurement 1.
        j (numpy.ndarray of int or float): synchronized time step or altitude of measurement 2.
        uc_type (str): uncertainty type. Must be one of ['r', 's', 't', 'u'].
        srn_i (numpy.ndarray of int or str, optional): serial number of RS from measurement 1.
        srn_j (numpy.ndarray of int or str, optional): seriel number of RS from measurement 2.
        mod_i (numpy.ndarray of int or str, optional): GDP model from measurement 1.
        mod_j (numpy.ndarray of int or str, optional): GDP model from measurement 2.
        rig_i (numpy.ndarray of int or str, optional): rig id of measurement 1.
        rig_j (numpy.ndarray of int or str, optional): rig id of measurement 2.
        evt_i (numpy.ndarray of int or str, optional): event id of measurement 1.
        evt_j (numpy.ndarray of int or str, optional): event id of measurement 2.
        sit_i (numpy.ndarray of int or str, optional): site id of measurement 1.
        sit_j (numpy.ndarray of int or str, optional): site id of measurement 2.

    Returns:
        numpy.ndarray of float(s): the correlation coefficient(s), in the range [0, 1].

    Note:
        This function returns the pair-wise correlation coefficients,
        **not** the full correlation matrix, i.e::

            len(corcoef_gdps(i, j, uc_type)) == len(i) == len(j)

    The supported uncertainty types are:

    - 'r': rig-correlated uncertainty.
           Intended for the so-called "uncorrelated" GRUAN uncertainty.
    - 's': spatial-correlated uncertainty.
           Full correlation between measurements acquired during the same event at the same site,
           irrespective of the time step/altitude, rig, radiosonde model, or serial number.
    - 't': temporal-correlated uncertainty.
           Full correlation between measurements acquired at distinct sites during distinct events,
           with distinct radiosondes models and serial numbers.
    - 'u': uncorrelated.
           No correlation whatsoever between distinct measurements.

    Todo:
        - Add reference to GRUAN docs & dvas articles in this docstring.

    '''

    # Begin with some safety checks
    for var in [i, j, srn_i, srn_j, mod_i, mod_j, rig_i, rig_j, evt_i, evt_j, sit_i, sit_j]:
        if var is None:
            continue
        if not isinstance(var, np.ndarray):
            raise Exception('Ouch ! I was expecting a numpy.ndarray, not %s' % type(var))
        if np.shape(var) != np.shape(i):
            raise Exception('Ouch ! All items should have the same shape !')

    # Make sure to return something with the same shape as what came in.
    corcoef = np.zeros_like(i)

    # All variables always correlate fully with themselves
    corcoef[(i == j) * (srn_i == srn_j) * (mod_i == mod_j) * (rig_i == rig_j) * (evt_i == evt_j) *
            (sit_i == sit_j)] = 1.0

    # Now work in the required level of correlation depending on the uncertainty type.
    if uc_type == 'u':
        # Nothing to add in case of uncorrelated uncertainties.
        pass

    elif uc_type == 'r':
        gruan_logger.warning('Rig-correlated uncertainties not yet defined.')
        #TODO: specify the rig-correlation rules.

    elif uc_type == 's':
        # 1) Full spatial-correlation between measurements acquired in the same event and at the
        #    same site, irrespective of the rig number, radiosonde model or serial number.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef[(evt_i == evt_j) * (sit_i == sit_j)] = 1.0

    elif uc_type == 't':
        # 1) Full temporal-correlation between measurements acquired in the same event and at the
        # same site.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef[(evt_i == evt_j) * (sit_i == sit_j)] = 1.0

        # 2) Full temporal correlation between measurements acquired in distinct events and sites.
        # TODO: confirm this is correct: incl. for different RS models ?
        corcoef = np.ones_like(corcoef)

    else:
        raise dvasError("Ouch ! uc_type must be one of ['r', 's', 't', 'u'], not: %s" % (uc_type))

    return corcoef

@log_func_call(gruan_logger)
def combine_gdps(gdp_prfs, binning=1, method='weighted mean'):
    ''' Combines and (possibly) rebins GDP profiles, with full error propagation.

    Args:
        gdp_profs (dvas.data.data.MultiGDPProfile): synchronized GDP profiles to combine.
        binning (int, optional): the number of profile steps to put in a bin. Defaults to 1.
        method (str, optional): combination rule. Can be 'weighted mean', or 'delta'.
            Defaults to 'weighted mean'.

    Returns:
        (dvas.data.data.MultiGDPProfile): the combined GDP profile.

    Warnings:
        * If the Serial Numbers are not specified, it will be assumed that they are **distinct** for
          each GDPProfile.
        * For the GDP models, as well as the rigs, events, and sites, the opposite is True. If they
          are not specified for each GDPProfiles, it will be assumed that they are the **same**
          for all.

    '''

    # Some safety checks first of all
    if not isinstance(binning, int):
        raise Exception('Ouch! binning must be of type int, not %s' % (type(binning)))
    if binning <= 0:
        raise Exception('Ouch! binning must be greater or equal to 1 !')

    if method not in ['weighted mean', 'delta']:
        raise Exception('Ouch ! Method %s unsupported.' % (method))

    # How many gdps do we have ?
    n_prf = len(gdp_prfs.profiles)
    # How long are the profiles ?
    # n_steps = len(profiles[0])

    # If I have more than 1 profile, and no SRN was specified, the user most likely forgot to
    # specify it.
    #if n_profiles >= 2 and srns is None:
    #    warn_msg = 'merge_andor_rebin_gdps() received %i profiles' % (n_profiles)
    #    warn_msg += ', but no SRN for them. Assuming they come from different RS.'
    #    gruan_logger.warning(warn_msg)
    #    srns = [np.ones(n_steps) + ind for ind in range(n_profiles)]

    # For a delta, I can only have two profiles
    if method == 'delta' and n_prf != 2:
        raise dvasError('Ouch! I can only make a delta between 2 GDPs, not %i !' % (n_prf))

    # Check that all the profiles/errors have the same length
    for item in gdp_prfs.profiles:
        if len(item.data) != len(gdp_prfs.profiles[0].data):
            raise dvasError('Ouch ! All GDP profiles must have the same length !')

    # let's get started
    if method == 'weighted mean':

        # Let's compute the total error for each profile
        #sigma_tots = [item.uc_tot for item in gdp_prfs.profiles]

        # Let's compute the weights vectors for each profile
        w_ps = pd.DataFrame([1/item.uc_tot**2 for item in gdp_prfs.get_prms('uc_tot')])

        prfs = pd.DataFrame([item.val for item in gdp_prfs.get_prmns()])

        # Let's assemble the arrays of w * x for all profiles
        #wx_ps = np.array([w_ps[p_ind] * profiles[p_ind] for p_ind in range(n_profiles)])

        # Sum these along the number of probes
        #wx_s = np.nansum(wx_ps, axis=0)
        #w_s = np.nansum(w_ps, axis=0)

        # Then sum these along the altitude layers according to the binning
        # Make sure to deal with potential NaN's by using the where function to swap them with 0's
        # This is because I can't find the "np.nanadd.reduceat" function
        #wx_ms = np.add.reduceat(np.where(np.isnan(wx_s), 0, wx_s), range(0, n_steps, binning))
        #w_ms = np.add.reduceat(np.where(np.isnan(w_s), 0, w_s), range(0, n_steps, binning))

        # Compute the weighted mean
        # To avoid some runtime Warning, replace any 0 weight with nan's
        #x_ms = wx_ms / np.where(w_ms == 0, np.nan, w_ms)

        # Compute the weighted mean
        x_ms, G_mat = stats.weighted_mean(prfs, w_ps, binning=binning)

    elif method == 'delta':

        # Compute the difference between the two profiles (full resolution)
        delta_pqs = profiles[0] - profiles[1]

        # Then sum these along the altitude layers according to the binning
        # Make sure to deal with potential NaN's by using the where function to swap them with 0's
        delta_pqm = np.add.reduceat(np.where(np.isnan(delta_pqs), 0, delta_pqs),
                                    range(0, n_steps, binning))

        # Build the mean by normalizing the sum by the number of altitude steps combined
        x_ms = delta_pqm / np.add.reduceat(np.ones(n_steps), range(0, n_steps, binning))

    # What is the length of my new profile ?
    n_steps_new = len(x_ms)

    # Which layer indexes from the original profiles are included in each step ?
    # Watch out for the different syntax between np.split and np.add.reduceat when specifying the
    # slices !
    # Also, turn it into a list of list
    old_inds = [list(item) for item in np.split(range(n_steps), range(binning, n_steps, binning))]

    # I will also compute the pseudo index of the new binned levels
    # Note: I take it as the simple mean of all the levels involved ... EVEN if this is a weighted
    # sum ! These indices are therefore only meant for quick plotting purposes.
    new_inds = np.array([np.mean(old_ind) for old_ind in old_inds])

    # Run some sanity checks before I continue
    if len(new_inds) != n_steps_new or len(old_inds) != n_steps_new:
        raise Exception('Ouch ! This error should be impossible !')

    # Let's get started with the computation of the errors. For now, let's do them one layer k at a
    # time.
    # Note: this is much (!) faster than implementing a full matrix approach, because these get
    # very large but are mostly filled with 0's. However, this technique does not provide any
    # information regarding the correlation of errors between different layer of the merged and
    # resampled spectra. For now, this is not needed, and therefore this remains an appropriate
    # choice. vof - 2020.04.16
    # Wait ... is that statement actually correct ? vof - 2020.06.25

    sigma_us_new = np.zeros_like(x_ms)
    sigma_es_new = np.zeros_like(x_ms)
    sigma_ss_new = np.zeros_like(x_ms)
    sigma_ts_new = np.zeros_like(x_ms)

    # For each layer k, we will arrange all the measurement points in a row, so we can use matrix
    # multiplication with the @ operator. To keep track of things, let's create the 1-D array that
    # identifies the profiles' serial numbers, rigs, events and sites.
    # First, the SN (identical for all measurement points within a profile)
    if srns is None:
        srns = [None] * n_profiles
    srns_inds = np.array([[srns[p_ind]] * n_steps for p_ind in range(n_profiles)]).ravel()
    # idem for the GDP models
    if mods is None:
        mods = [None] * n_profiles
    mods_inds = np.array([[mods[p_ind]] * n_steps for p_ind in range(n_profiles)]).ravel()
    # idem for the rigs
    if rigs is None:
        rigs = [None] * n_profiles
    rigs_inds = np.array([[rigs[p_ind]] * n_steps for p_ind in range(n_profiles)]).ravel()
    # idem for the events
    if evts is None:
        evts = [None] * n_profiles
    evts_inds = np.array([[evts[p_ind]] * n_steps for p_ind in range(n_profiles)]).ravel()
    # idem for the sites
    if sits is None:
        sits = [None] * n_profiles
    sits_inds = np.array([[sits[p_ind]] * n_steps for p_ind in range(n_profiles)]).ravel()

    # Let's also keep track of the original indices of the data
    i_inds = np.array(n_profiles * list(range(n_steps))).ravel()

    # Next, I follow Barlow p.60. Only I do it per bin k, and I only keep the profile points
    # that are included in the construction of the specific bin. This keeps the matrices small by
    # avoiding a lot of 0's.
    # Let's now start looping through all the layers
    for k_ind in range(n_steps_new): # loop through all the new bins

        # First, what are the limiting level indices of this layer ?
        j_min = np.min(old_inds[k_ind])
        j_max = np.max(old_inds[k_ind]) + 1

        # What are the indices of the in-layer points ?
        in_layer = (i_inds >= j_min) * (i_inds < j_max)
        # How many points are included in this layer
        n_in_layer = len(in_layer[in_layer])
        # How thick is the layer actually ?
        n_layer = n_in_layer // n_profiles # This is an integer

        # Quick sanity check for all but the last bin, that I have the correct number of points.
        if n_in_layer != binning * n_profiles and k_ind < n_steps_new - 1:
            raise Exception('Ouch! This error is impossible.')

        # First, build the G matrix
        if method == 'weighted mean':
            G_mat = w_ps.ravel()[in_layer]/w_ms[k_ind]
        elif method == 'delta':
            G_mat = np.append(1/n_layer * np.ones(n_layer), -1/n_layer * np.ones(n_layer))

        # Very well, the covariance matrix V is the sum of the different uncertainty components:
        # uncorrelated, environmental-correlated, spatial-correlated, temporal-correlated
        # Let's assemble the uncertainties of this specific step
        local_sigma_us = np.array([sigma[j_min:j_max] for sigma in sigma_us]).ravel()
        local_sigma_es = np.array([sigma[j_min:j_max] for sigma in sigma_es]).ravel()
        local_sigma_ss = np.array([sigma[j_min:j_max] for sigma in sigma_ss]).ravel()
        local_sigma_ts = np.array([sigma[j_min:j_max] for sigma in sigma_ts]).ravel()

        # Let us now assemble the U matrices, filling all the cross-correlations for the different
        # types of uncertainties
        U_mats = []
        for (sigma_name, sigma_vals) in [('sigma_u', local_sigma_us),
                                         ('sigma_e', local_sigma_es),
                                         ('sigma_s', local_sigma_ss),
                                         ('sigma_t', local_sigma_ts),
                                        ]:
            U_mat = corcoef_gdps(np.tile(i_inds[in_layer], (n_in_layer, 1)), # i
                                 np.tile(i_inds[in_layer], (n_in_layer, 1)).T, # j
                                 sigma_name,
                                 srn_i=np.tile(srns_inds[in_layer], (n_in_layer, 1)),
                                 srn_j=np.tile(srns_inds[in_layer], (n_in_layer, 1)).T,
                                 mod_i=np.tile(mods_inds[in_layer], (n_in_layer, 1)),
                                 mod_j=np.tile(mods_inds[in_layer], (n_in_layer, 1)).T,
                                 rig_i=np.tile(rigs_inds[in_layer], (n_in_layer, 1)),
                                 rig_j=np.tile(rigs_inds[in_layer], (n_in_layer, 1)).T,
                                 evt_i=np.tile(evts_inds[in_layer], (n_in_layer, 1)),
                                 evt_j=np.tile(evts_inds[in_layer], (n_in_layer, 1)).T,
                                 sit_i=np.tile(sits_inds[in_layer], (n_in_layer, 1)),
                                 sit_j=np.tile(sits_inds[in_layer], (n_in_layer, 1)).T,
                                )

            # Include a sanity check: if U_mat is not an identity matrix for sigma_u,
            # something is very wrong.
            if sigma_name == 'sigma_u' and np.any(U_mat != np.identity(n_in_layer)):
                raise Exception('Ouch ! Something is very wrong here.')

            # Implement the multiplication. Mind the structure of these arrays to get the correct
            # mix of Hadamard and dot products where I need them !
            U_mat = np.multiply(U_mat, np.array([sigma_vals]).T @ np.array([sigma_vals]))
            U_mats += [U_mat]

        # I can finally use matrix multiplication (using @) to save me a lot of convoluted sums ...
        # If I only have nan's then that's my result. If I only have partial nan's, then make sure I
        # still get a number out.
        if np.all(np.isnan(G_mat)):
            sigma_us_new[k_ind] = np.nan
            sigma_es_new[k_ind] = np.nan
            sigma_ss_new[k_ind] = np.nan
            sigma_ts_new[k_ind] = np.nan

        else:
            # Replace all the nan's with zeros so they do not intervene in the sums
            sigma_us_new[k_ind] = np.where(np.isnan(G_mat), 0, G_mat) @ \
                                  np.where(np.isnan(U_mats[0]), 0, U_mats[0]) @ \
                                  np.where(np.isnan(G_mat.T), 0, G_mat.T)
            sigma_us_new[k_ind] = np.sqrt(sigma_us_new[k_ind])

            sigma_es_new[k_ind] = np.where(np.isnan(G_mat), 0, G_mat) @ \
                                  np.where(np.isnan(U_mats[1]), 0, U_mats[1]) @ \
                                  np.where(np.isnan(G_mat.T), 0, G_mat.T)
            sigma_es_new[k_ind] = np.sqrt(sigma_es_new[k_ind])

            sigma_ss_new[k_ind] = np.where(np.isnan(G_mat), 0, G_mat) @ \
                                  np.where(np.isnan(U_mats[2]), 0, U_mats[2]) @ \
                                  np.where(np.isnan(G_mat.T), 0, G_mat.T)
            sigma_ss_new[k_ind] = np.sqrt(sigma_ss_new[k_ind])

            sigma_ts_new[k_ind] = np.where(np.isnan(G_mat), 0, G_mat) @ \
                                  np.where(np.isnan(U_mats[3]), 0, U_mats[3]) @ \
                                  np.where(np.isnan(G_mat.T), 0, G_mat.T)
            sigma_ts_new[k_ind] = np.sqrt(sigma_ts_new[k_ind])

    return (x_ms, sigma_us_new, sigma_es_new, sigma_ss_new, sigma_ts_new, old_inds, new_inds)
