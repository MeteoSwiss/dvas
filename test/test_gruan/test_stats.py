# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Testing 'stats' classes and function of the gruan submodule.

"""

# Import from python packages and modules
import numpy as np
import scipy.stats as ss
import pandas as pd

# Import from this package
from dvas.gruan import stats


def test_stats_gdp_ks_test():
    """Function used to test if the KS test between 2 GDPs is behaving ok.

    The function tests:
        - ability to correctly flag points that fail the test

    TODO:
        - generate a test based on actual data ?
        - test the impact of binning ?
    """

    # Create two "profiles" in sigma scale
    # Here, we only consider random errors for now.
    #profile = range(1, 5, 1) * np.sqrt(8)
    #qrofile = np.zeros(4)

    # Create fake errors, such that the "total" error of profile-qrofile is np.sqrt(8)
    #sigma_u = np.ones(4) * 2
    #sigma_e = np.zeros(4)
    #sigma_s = np.zeros(4)
    #sigma_t = np.zeros(4)

    # Note: p-value of 3-sigma level = 0.0027 = (1 - ss.norm.cdf(3))*2
    #(f_pqi, _) = stats.gdp_ks_test([profile, qrofile],
    #                               [sigma_u]*2, [sigma_e]*2, [sigma_s]*2, [sigma_t]*2,
    #                               alpha=(1 - ss.norm.cdf(3))*2, # This corresponds to 3 sigma level
    #                               binning_list=[1, 2, 20], do_plot=False, srns=[1, 2])

    # If I get here, then most likely it is all working fine.
    #assert np.all(f_pqi[0] == np.array([0., 0., 1., 1.]))
    assert True

def test_weighted_mean():
    """ Function used to test the weighted_mean combination of profiles.

    """

    vals = pd.DataFrame(np.ones((10, 3)))
    vals[1] = 2.
    vals[2] = 4.
    weights = pd.DataFrame(np.ones((10, 3)))

    # Can I actually compute a weighted mean ?
    assert np.all(stats.weighted_mean(vals, weights, binning=1)[0] == 7/3)
    assert np.all(stats.weighted_mean(vals, weights, binning=3)[0] == 7/3)

    vals.iloc[0] = np.nan
    vals[0][1] = np.nan
    weights[1][1] = np.nan
    weights.iloc[9] = np.nan

    # If all values for a bin a NaN, result should be NaN
    assert np.isnan(stats.weighted_mean(vals, weights, binning=1)[0][0])
    # But if only some of the bin are NaNs, I should return a number
    assert stats.weighted_mean(vals, weights, binning=1)[0][1] == 4.0
    assert stats.weighted_mean(vals, weights, binning=2)[0][0] == 4.0
    assert stats.weighted_mean(vals, weights, binning=2)[0][1] == 7/3

    # If all weights for a bin are NaN, result should be Nan
    assert np.isnan(stats.weighted_mean(vals, weights, binning=1)[0][9])

    # Check that the last bin is smaller than the others, I still compute it
    assert len(stats.weighted_mean(vals, weights, binning=3)[0]) == 4

def test_delta():
    """ Function used to test the weighted_mean combination of profiles.

    """

    vals = pd.DataFrame(np.ones((10, 2)))
    vals[1] = 2.

    # Can I actually compute a difference ?
    assert np.all(stats.delta(vals, binning=1)[0] == -1)

    vals.iloc[0] = np.nan
    vals[0][9] = np.nan

    # If all values for a bin a NaN, result should be NaN
    assert np.isnan(stats.delta(vals, binning=1)[0][0])

    # If only part of a bin is NaN, then report a number
    assert stats.delta(vals, binning=2)[0][0] == -1

    # If only on number is NaN, result should be Nan
    assert np.isnan(stats.delta(vals, binning=1)[0][9])

    # Check that the last bin is smaller than the others, I still compute it
    assert len(stats.delta(vals, binning=4)[0]) == 3

#def test_stats_gdp_ks_test_2():
    """Function used to test if the KS test between 2 GDPs is behaving ok.

    The function tests:
        - ability to make a good-looking plot

    """
    '''
    # Create two "profiles" in sigma scale
    # Here, we only consider random errors for now.
    profile = np.random.normal(loc=3, scale=2, size=600)
    qrofile = np.random.normal(loc=3, scale=2, size=600)

    # Create fake errors, such that the "total" error of profile-qrofile is np.sqrt(8)
    sigma_u = np.ones(600) * 1
    sigma_e = np.zeros(600)
    sigma_s = np.zeros(600)
    sigma_t = np.zeros(600)

    (f_pqi, _) = stats.gdp_ks_test([profile, qrofile],
                                   [sigma_u]*2, [sigma_e]*2, [sigma_s]*2, [sigma_t]*2,
                                   alpha=0.0027, # This corresponds to a 3 sigma level
                                   binning_list=[1, 2, 10, 40], srns=[1, 2])

    # If I get here, then most likely it is all working fine.
    assert np.all(f_pqi[0] == np.array([0., 0., 1., 1.]))
    '''
