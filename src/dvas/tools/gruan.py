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
from .tools import weighted_mean, delta

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
        binning (int, optional): the number of profile steps to put into a bin. Defaults to 1.
        method (str, optional): combination rule. Can be 'weighted mean', 'mean', or 'delta'.
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
    if method not in ['weighted mean', 'mean', 'delta']:
        raise Exception('Ouch ! Method %s unsupported.' % (method))

    # How many gdps do we have ?
    n_prf = len(gdp_prfs)
    # How long are the profiles ?
    len_gdp = len(gdp_prfs.profiles[0].data)

    # For a delta, I can only have two profiles
    if method == 'delta' and n_prf != 2:
        raise dvasError('Ouch! I can only make a delta between 2 GDPs, not %i !' % (n_prf))

    # Check that all the profiles/errors have the same length
    for item in gdp_prfs.profiles:
        if len(item.data) != len_gdp:
            raise dvasError('Ouch ! All GDP profiles must have the same length !')

    # let's get started
    if 'mean' in method:

        # To do this, I need to extract the profile values
        x_ps = pd.DataFrame([item['val'].values for item in gdp_prfs.get_prms('val')]).T

        # TODO: the following "cleaner" line would work if all frames have the same alt and tdt
        # indexes
        # x_ps = pd.concat(gdp_prfs.get_prms('val'), axis=1)

        if method == 'weighted mean':
            # I also need the associated weights, which are computed from the total errors
            w_ps = pd.DataFrame([1/item['uc_tot'].values**2
                                 for item in gdp_prfs.get_prms('uc_tot')]).T
        else:
            # Set all the wights to 1 to make a basic "mean"
            w_ps = pd.DataFrame(np.ones_like(x_ps))

        # Compute the weighted mean
        x_ms, jac_elmts = weighted_mean(x_ps, w_ps, binning=binning)

    elif method == 'delta':

        # To do this, I need to extract the profile values
        x_ps = pd.DataFrame([item['val'].values for item in gdp_prfs.get_prms('val')]).T
        # TODO: the following "cleaner" line would work if all frames have the same alt and tdt
        # indexes
        # x_ps = pd.concat(gdp_prfs.get_prms('val'), axis=1)

        # Compute the weighted mean
        x_ms, jac_elmts = delta(x_ps, binning=binning)

    # TODO: what happens to the 'alt' and 'tdt' variables ?
    # -> Nothing. We will request/assume that the profiles are synchronized and all share the same
    # altitude/tdt bins.

    # What is the length of my combined profile ?
    len_comb = len(x_ms)

    # Which layer indexes from the original profiles are included in each step ?
    # Watch out for the different syntax between np.split and np.add.reduceat when specifying the
    # slices !
    # Also, turn it into a list of list
    old_inds = [list(item) for item in np.split(range(len_gdp), range(binning, len_gdp, binning))]

    # I will also compute the pseudo index of the new binned levels
    # Note: I take it as the simple mean of all the levels involved ... EVEN if this is a weighted
    # sum ! These indices are therefore only meant for quick plotting purposes.
    #new_inds = np.array([np.mean(old_ind) for old_ind in old_inds])

    # Let's get started with the computation of the errors. For now, let's do them one layer k at a
    # time.
    # Note: this is much (!) faster than implementing a full matrix approach, because these get
    # very large but are mostly filled with 0's. However, this technique does not provide any
    # information regarding the correlation of errors between different layer of the merged and
    # resampled spectra. For now, this is not needed, and therefore this remains an appropriate
    # choice. vof - 2020.04.16
    # Wait ... is that statement actually correct ? vof - 2020.06.25

    comb_us = np.zeros_like(x_ms)
    comb_rs = np.zeros_like(x_ms)
    comb_ss = np.zeros_like(x_ms)
    comb_ts = np.zeros_like(x_ms)

    # For each layer k, we will arrange all the measurement points in a row, so we can use matrix
    # multiplication with the @ operator. To keep track and handle the correlations, let's create
    # 1-D array that identify the profiles' serial numbers, rigs, events and sites for each points.

    # First, the SN (identical for all measurement points within a profile)
    srns = gdp_prfs.get_evt_prm('srn')
    # Let us make sure that a single srn is tied to each profile. I would not know how to deal with
    # the correlation matrix of a combined profile!

    if np.any([len(item) > 1 for item in srns]):
        raise dvasError('Ouch! I need a single srn per profile to be combined.')
    # Clean-up all the sublists.
    srns = [item[0] for item in srns]
    # Build the full array of indices
    srns_inds = np.array([[item] * len_gdp for item in srns]).flatten()
    # idem for the GDP models
    # TODO: include these inside the events
    #mods = gdp_prfs.get_evt_prm('mod')
    #mods_inds = np.array([[item] * len_gdp for item in mods]).flatten()
    mods_ind = None

    # I will also need some tags ...
    tags = gdp_prfs.get_evt_prm('tag')
    # TODO: the folllowing is not robust, in the sense that a new tag starting with 'r' would be
    # picked up. Is there any way to select only the rig tags with 100% certainty ?
    # ...including the rig id
    rigs = [tag for sublist in tags for tag in sublist if (tag[0] == 'r' and tag != 'raw')]
    rigs_inds = np.array([[item] * len_gdp for item in rigs]).flatten()
    # idem for the events
    evts = [tag for sublist in tags for tag in sublist if (tag[0] == 'e')]
    evts_inds = np.array([[item] * len_gdp for item in evts]).flatten()
    # idem for the sites
    # TODO: add the sites to the EventManager
    #sits = gdp_prfs.get_evt_prm('sit')
    #sits_inds = np.array([[item] * len_gdp for item in sits]).flatten()
    sits_ind = None

    # Let's also keep track of the original indices of the data
    i_inds = np.array(n_prf * list(range(len_gdp))).ravel()

    # Next, I follow Barlow p.60. Only I do it per bin k, and I only keep the profile points
    # that are included in the construction of thay specific bin in the Jacobian/ U matrices.
    # This keeps the matrices small by avoiding a lot of 0's.

    # Let's now start looping through all the layers
    for k_ind in range(len_comb): # loop through all the new bins

        # First, what are the limiting level indices of this layer ?
        j_min = np.min(old_inds[k_ind])
        j_max = np.max(old_inds[k_ind]) + 1
        # What are the indices of the in-layer points ?
        in_layer = (i_inds >= j_min) * (i_inds < j_max)
        # How many points are included in this layer
        n_in_layer = len(in_layer[in_layer])
        # How thick is the layer actually ?
        n_layer = n_in_layer // n_prf # This is an integer

        # Very well, the covariance matrix V is the sum of the different uncertainty components:
        # uncorrelated, rig-correlated, spatial-correlated, temporal-correlated

        # Let's assemble the uncertainties of this specific step
        # Here I leave pandas in favor of  numpy arrays.
        us_k = np.array([sigma[j_min:j_max].values for sigma in gdp_prfs.get_prms('ucn')]).ravel()
        rs_k = np.array([sigma[j_min:j_max].values for sigma in gdp_prfs.get_prms('ucr')]).ravel()
        ss_k = np.array([sigma[j_min:j_max].values for sigma in gdp_prfs.get_prms('ucs')]).ravel()
        ts_k = np.array([sigma[j_min:j_max].values for sigma in gdp_prfs.get_prms('uct')]).ravel()

        # Let us now assemble the U matrices, filling all the cross-correlations for the different
        # types of uncertainties
        u_mats = []
        for (sigma_name, sigma_vals) in [('u', us_k), ('r', rs_k), ('s', ss_k), ('t', ts_k)]:
            # TODO: add the sites and RS models once available
            u_mat = corcoef_gdps(np.tile(i_inds[in_layer], (n_in_layer, 1)), # i
                                 np.tile(i_inds[in_layer], (n_in_layer, 1)).T, # j
                                 sigma_name,
                                 srn_i=np.tile(srns_inds[in_layer], (n_in_layer, 1)),
                                 srn_j=np.tile(srns_inds[in_layer], (n_in_layer, 1)).T,
                                 #mod_i=np.tile(mods_inds[in_layer], (n_in_layer, 1)),
                                 #mod_j=np.tile(mods_inds[in_layer], (n_in_layer, 1)).T,
                                 rig_i=np.tile(rigs_inds[in_layer], (n_in_layer, 1)),
                                 rig_j=np.tile(rigs_inds[in_layer], (n_in_layer, 1)).T,
                                 evt_i=np.tile(evts_inds[in_layer], (n_in_layer, 1)),
                                 evt_j=np.tile(evts_inds[in_layer], (n_in_layer, 1)).T,
                                 #sit_i=np.tile(sits_inds[in_layer], (n_in_layer, 1)),
                                 #sit_j=np.tile(sits_inds[in_layer], (n_in_layer, 1)).T,
                                )

            # Implement the multiplication. Mind the structure of these arrays to get the correct
            # mix of Hadamard and dot products where I need them !
            u_mat = np.multiply(u_mat, np.array([sigma_vals]).T @ np.array([sigma_vals]))
            u_mats += [u_mat]

        # I can finally use matrix multiplication (using @) to save me a lot of convoluted sums ...
        # If I only have nan's then that's my result. If I only have partial nan's, then make sure I
        # still get a number out.
        if np.all(np.isnan(jac_elmts)):
            comb_us[k_ind] = np.nan
            comb_rs[k_ind] = np.nan
            comb_ss[k_ind] = np.nan
            comb_ts[k_ind] = np.nan

        else:
            # Replace all the nan's with zeros so they do not intervene in the sums
            comb_us[k_ind] = np.where(np.isnan(jac_elmts[k_ind]), 0, jac_elmts[k_ind]) @ \
                                np.where(np.isnan(u_mats[0]), 0, u_mats[0]) @ \
                                np.where(np.isnan(jac_elmts[k_ind].T), 0, jac_elmts[k_ind].T)
            comb_us[k_ind] = np.sqrt(comb_us[k_ind])

            comb_rs[k_ind] = np.where(np.isnan(jac_elmts[k_ind]), 0, jac_elmts[k_ind]) @ \
                                  np.where(np.isnan(u_mats[1]), 0, u_mats[1]) @ \
                                  np.where(np.isnan(jac_elmts[k_ind].T), 0, jac_elmts[k_ind].T)
            comb_rs[k_ind] = np.sqrt(comb_rs[k_ind])

            comb_ss[k_ind] = np.where(np.isnan(jac_elmts[k_ind]), 0, jac_elmts[k_ind]) @ \
                                  np.where(np.isnan(u_mats[2]), 0, u_mats[2]) @ \
                                  np.where(np.isnan(jac_elmts[k_ind].T), 0, jac_elmts[k_ind].T)
            comb_ss[k_ind] = np.sqrt(comb_ss[k_ind])

            comb_ts[k_ind] = np.where(np.isnan(jac_elmts[k_ind]), 0, jac_elmts[k_ind]) @ \
                                  np.where(np.isnan(u_mats[3]), 0, u_mats[3]) @ \
                                  np.where(np.isnan(jac_elmts[k_ind].T), 0, jac_elmts[k_ind].T)
            comb_ts[k_ind] = np.sqrt(comb_ts[k_ind])

    # Very well, I now need to build a MultiProfile with the data that I have.
    # First, I need to build a profile which includes a DataFrame. Let us copy the first one
    # and replace the column as required.
    # TODO: omce x_ms comes as a dataframe with the proper indexes (see previous TODOS),
    # I should be able to do something cleaner than this to assemble the combined DataFrame.
    df_out = gdp_prfs.profiles[0].data.copy(deep=True)
    # Assign the proper values
    df_out['val'] = x_ms.values
    df_out['ucn'] = comb_us
    df_out['ucr'] = comb_rs
    df_out['ucs'] = comb_ss
    df_out['uct'] = comb_ts

    # TODO
    # That next line is not exactly the best ... undoing the index to then set it again in the
    # Profile. Something should be done about that.
    df_out.reset_index(['alt', 'tdt'], inplace=True)

    # Let's prepare the info dict
    # TODO: this needs to be imporved (tags handling, many datetime, etc ... ?)
    info_dict = {'dt': gdp_prfs.events[0].event_dt,
                 'sn': srns,
                 'tag': ['cws']}

    # Let's create a dedicated Profile for the CWS. It's not much different from a GDP.
    cws_prf = ddsd.GDPProfile(info_dict, data=df_out)

    #out.update(gdp_prfs.db_variables, [

    #import pdb
    #pdb.set_trace()

    return (x_ms, comb_us, comb_rs, comb_ss, comb_ts, old_inds)
